#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import traceback

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import median_filter, gaussian_filter

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import pyvista as pv
from pyvistaqt import QtInteractor

from algoritma.planner import plan_path


def ensure_odd(value: int) -> int:
    """Verilen sayıyı en az 3 olacak şekilde tek sayıya çevirir."""
    value = int(max(3, value))
    return value if value % 2 == 1 else value + 1


def robust_norm_to_uint8(array: np.ndarray) -> np.ndarray:
    """Yükseklik verisini önizleme için 0-255 aralığına ölçekler."""
    normalized = array.copy().astype(np.float32)
    valid_mask = np.isfinite(normalized)

    if not valid_mask.any():
        return np.zeros_like(normalized, dtype=np.uint8)

    p2, p98 = np.percentile(normalized[valid_mask], [2, 98])
    if p98 <= p2:
        p98 = p2 + 1e-6

    normalized = np.clip((normalized - p2) / (p98 - p2), 0, 1)
    normalized[~valid_mask] = 0
    return (normalized * 255).astype(np.uint8)


def fill_nans_iterative(array: np.ndarray, iterations: int = 6) -> np.ndarray:
    """NaN hücreleri komşu değerlerle yaklaşık doldurur."""
    filled = array.astype(np.float32, copy=True)
    nan_mask = ~np.isfinite(filled)

    if not nan_mask.any():
        return filled

    valid_values = filled[np.isfinite(filled)]
    fallback = np.median(valid_values) if valid_values.size else 0.0
    filled[nan_mask] = fallback

    target_mask = nan_mask.copy()
    for _ in range(iterations):
        up = np.roll(filled, -1, axis=0)
        down = np.roll(filled, 1, axis=0)
        left = np.roll(filled, -1, axis=1)
        right = np.roll(filled, 1, axis=1)
        filled[target_mask] = 0.25 * (
            up[target_mask] + down[target_mask] + left[target_mask] + right[target_mask]
        )

    return filled


def destripe_1d_profiles(
    z: np.ndarray,
    row_med_win: int,
    col_med_win: int,
    sigma_row: float,
    sigma_col: float,
) -> np.ndarray:
    """Satır ve sütun profillerindeki bantlaşmayı azaltır."""
    filtered = z.astype(np.float32, copy=True)

    row_profile = np.median(filtered, axis=1)
    row_window = ensure_odd(min(row_med_win, max(3, (len(row_profile) // 2) * 2 - 1)))
    row_trend = median_filter(row_profile, size=row_window, mode="nearest")
    row_residual = row_profile - row_trend
    row_low = gaussian_filter(row_residual, sigma=sigma_row, mode="nearest")
    filtered -= 0.8 * (row_residual - row_low)[:, None]

    col_profile = np.median(filtered, axis=0)
    col_window = ensure_odd(min(col_med_win, max(3, (len(col_profile) // 2) * 2 - 1)))
    col_trend = median_filter(col_profile, size=col_window, mode="nearest")
    col_residual = col_profile - col_trend
    col_low = gaussian_filter(col_residual, sigma=sigma_col, mode="nearest")
    filtered -= 0.8 * (col_residual - col_low)[None, :]

    return filtered


def final_smooth(z: np.ndarray, median_smooth_size: int, gauss_sigma: float) -> np.ndarray:
    """Son yumuşatma adımını uygular."""
    smoothed = median_filter(z, size=median_smooth_size, mode="nearest")
    smoothed = gaussian_filter(smoothed, sigma=gauss_sigma, mode="nearest")
    return smoothed.astype(np.float32)


class MapView(QtWidgets.QGraphicsView):
    """Önizleme rasterı üzerinde ROI, başlangıç ve hedef seçimi yapar."""

    def __init__(self):
        super().__init__()

        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        self.scene_ = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene_)

        self.pix_item = None
        self.mode = "roi"

        self.roi_rect = None
        self.start_pt = None
        self.goal_pt = None

        self.roi_item = None
        self.start_item = None
        self.goal_item = None
        self.shadow_overlay_item = None

        self.dragging = False
        self.drag_start = None

        self.img_w = 0
        self.img_h = 0

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def clear_selection(self) -> None:
        self.roi_rect = None
        self.start_pt = None
        self.goal_pt = None
        self.dragging = False
        self.drag_start = None

        if self.roi_item is not None:
            self.scene_.removeItem(self.roi_item)
            self.roi_item = None

        if self.start_item is not None:
            self.scene_.removeItem(self.start_item)
            self.start_item = None

        if self.goal_item is not None:
            self.scene_.removeItem(self.goal_item)
            self.goal_item = None

        if self.shadow_overlay_item is not None:
            self.scene_.removeItem(self.shadow_overlay_item)
            self.shadow_overlay_item = None

    def set_image(self, pixmap: QtGui.QPixmap) -> None:
        self.scene_.clear()
        self.pix_item = self.scene_.addPixmap(pixmap)
        self.scene_.setSceneRect(0, 0, pixmap.width(), pixmap.height())

        self.img_w = pixmap.width()
        self.img_h = pixmap.height()

        self.clear_selection()
        self.resetTransform()
        self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)

    def set_shadow_overlay(self, roi_rect, shadow_map: np.ndarray) -> None:
        """ROI içine yarı saydam mor shadow overlay çizer."""
        if self.pix_item is None or roi_rect is None or shadow_map is None:
            return

        if self.shadow_overlay_item is not None:
            self.scene_.removeItem(self.shadow_overlay_item)
            self.shadow_overlay_item = None

        x0, y0, x1, y1 = roi_rect
        roi_width = max(1, x1 - x0 + 1)
        roi_height = max(1, y1 - y0 + 1)

        shadow_mask = (shadow_map > 0).astype(np.uint8)

        rgba = np.zeros((shadow_mask.shape[0], shadow_mask.shape[1], 4), dtype=np.uint8)
        rgba[..., 0] = 160
        rgba[..., 1] = 60
        rgba[..., 2] = 200
        rgba[..., 3] = shadow_mask * 90

        qimage = QtGui.QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            4 * rgba.shape[1],
            QtGui.QImage.Format_RGBA8888,
        ).copy()

        pixmap = QtGui.QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(
            roi_width,
            roi_height,
            Qt.IgnoreAspectRatio,
            Qt.FastTransformation,
        )

        self.shadow_overlay_item = self.scene_.addPixmap(pixmap)
        self.shadow_overlay_item.setPos(x0, y0)
        self.shadow_overlay_item.setZValue(5)

    def wheelEvent(self, event):
        if self.pix_item is None:
            return super().wheelEvent(event)

        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if self.pix_item is None:
            return super().mousePressEvent(event)

        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)

        scene_point = self.mapToScene(event.pos())
        x, y = int(scene_point.x()), int(scene_point.y())

        if not (0 <= x < self.img_w and 0 <= y < self.img_h):
            return

        if self.mode == "roi":
            self.dragging = True
            self.drag_start = (x, y)

            if self.roi_item is None:
                self.roi_item = self.scene_.addRect(
                    x, y, 1, 1,
                    QtGui.QPen(QtGui.QColor(255, 0, 0), 2),
                )
            else:
                self.roi_item.setRect(x, y, 1, 1)

        elif self.mode == "start":
            if self.roi_rect is None:
                return

            x0, y0, x1, y1 = self.roi_rect
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                return

            self.start_pt = (x, y)

            if self.start_item is not None:
                self.scene_.removeItem(self.start_item)

            self.start_item = self.scene_.addEllipse(
                x - 4, y - 4, 8, 8,
                QtGui.QPen(QtGui.QColor(0, 255, 0), 2),
                QtGui.QBrush(QtGui.QColor(0, 255, 0)),
            )

        elif self.mode == "goal":
            if self.roi_rect is None:
                return

            x0, y0, x1, y1 = self.roi_rect
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                return

            self.goal_pt = (x, y)

            if self.goal_item is not None:
                self.scene_.removeItem(self.goal_item)

            self.goal_item = self.scene_.addEllipse(
                x - 4, y - 4, 8, 8,
                QtGui.QPen(QtGui.QColor(0, 0, 255), 2),
                QtGui.QBrush(QtGui.QColor(0, 0, 255)),
            )

    def mouseMoveEvent(self, event):
        if self.pix_item is None:
            return super().mouseMoveEvent(event)

        if self.dragging and self.mode == "roi" and self.drag_start is not None:
            scene_point = self.mapToScene(event.pos())
            x = int(np.clip(int(scene_point.x()), 0, self.img_w - 1))
            y = int(np.clip(int(scene_point.y()), 0, self.img_h - 1))

            x0, y0 = self.drag_start
            rx0, ry0 = min(x0, x), min(y0, y)
            rx1, ry1 = max(x0, x), max(y0, y)

            self.roi_rect = (rx0, ry0, rx1, ry1)

            if self.roi_item is not None:
                self.roi_item.setRect(rx0, ry0, max(1, rx1 - rx0), max(1, ry1 - ry0))
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.pix_item is None:
            return super().mouseReleaseEvent(event)

        if event.button() == Qt.LeftButton and self.dragging and self.mode == "roi":
            self.dragging = False

            if self.roi_item is not None:
                rect = self.roi_item.rect()

                x0 = int(np.clip(int(rect.left()), 0, self.img_w - 1))
                x1 = int(np.clip(int(rect.right()), 0, self.img_w - 1))
                y0 = int(np.clip(int(rect.top()), 0, self.img_h - 1))
                y1 = int(np.clip(int(rect.bottom()), 0, self.img_h - 1))

                if x1 > x0 and y1 > y0:
                    self.roi_rect = (x0, y0, x1, y1)
            return

        super().mouseReleaseEvent(event)


class ComputeWorker(QtCore.QObject):
    """Raster okuma, preprocess ve rota hesabını arka planda yapar."""

    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, payload: dict):
        super().__init__()
        self.payload = payload

    @QtCore.pyqtSlot()
    def run(self):
        try:
            params = self.payload
            self.progress.emit("Raster okunuyor...")

            with rasterio.open(params["geotiff_path"]) as src:
                preview_height, preview_width = params["preview_shape"]

                scale_x = src.width / preview_width
                scale_y = src.height / preview_height

                x0p, y0p, x1p, y1p = params["roi_rect"]
                x0 = int(np.floor(x0p * scale_x))
                x1 = int(np.ceil((x1p + 1) * scale_x))
                y0 = int(np.floor(y0p * scale_y))
                y1 = int(np.ceil((y1p + 1) * scale_y))

                x0 = int(np.clip(x0, 0, src.width - 1))
                x1 = int(np.clip(x1, x0 + 1, src.width))
                y0 = int(np.clip(y0, 0, src.height - 1))
                y1 = int(np.clip(y1, y0 + 1, src.height))

                start_x_preview, start_y_preview = params["start_pt"]
                goal_x_preview, goal_y_preview = params["goal_pt"]

                start_col_full = int(np.clip(round(start_x_preview * scale_x), x0, x1 - 1))
                start_row_full = int(np.clip(round(start_y_preview * scale_y), y0, y1 - 1))
                goal_col_full = int(np.clip(round(goal_x_preview * scale_x), x0, x1 - 1))
                goal_row_full = int(np.clip(round(goal_y_preview * scale_y), y0, y1 - 1))

                downsample = max(1, int(params["downsample"]))
                out_height = max(2, int((y1 - y0) / downsample))
                out_width = max(2, int((x1 - x0) / downsample))

                window = Window(x0, y0, x1 - x0, y1 - y0)
                z = src.read(
                    1,
                    window=window,
                    out_shape=(out_height, out_width),
                    resampling=rasterio.enums.Resampling.bilinear,
                ).astype(np.float32)

                if src.nodata is not None:
                    z[np.isclose(z, src.nodata)] = np.nan

            self.progress.emit("Preprocess...")

            z = fill_nans_iterative(z, iterations=int(params["fill_iter"]))
            z = destripe_1d_profiles(
                z=z,
                row_med_win=ensure_odd(int(params["row_med_win"])),
                col_med_win=ensure_odd(int(params["col_med_win"])),
                sigma_row=float(params["sigma_row"]),
                sigma_col=float(params["sigma_col"]),
            )
            z = final_smooth(
                z=z,
                median_smooth_size=ensure_odd(int(params["median_smooth"])),
                gauss_sigma=float(params["gauss_sigma"]),
            )

            start_row = int(np.clip(round((start_row_full - y0) / downsample), 0, z.shape[0] - 1))
            start_col = int(np.clip(round((start_col_full - x0) / downsample), 0, z.shape[1] - 1))
            goal_row = int(np.clip(round((goal_row_full - y0) / downsample), 0, z.shape[0] - 1))
            goal_col = int(np.clip(round((goal_col_full - x0) / downsample), 0, z.shape[1] - 1))

            self.progress.emit("A* çalışıyor...")

            shadow_map = np.zeros_like(z, dtype=np.float32)

            # Test için basit gölge bölgesi:
            # ROI'nin sağ tarafında dikey bir bant gölgeli kabul ediliyor.
            height, width = z.shape
            shadow_start_col = int(width * 0.55)
            shadow_end_col = int(width * 0.75)
            shadow_map[:, shadow_start_col:shadow_end_col] = 1.0

            path, cost = plan_path(
                z=z,
                start_rc=(start_row, start_col),
                goal_rc=(goal_row, goal_col),
                diagonal=bool(params["diagonal"]),
                slope_weight=float(params["slope_weight"]),
                uphill_extra=float(params["uphill_extra"]),
                max_slope=params["max_slope"],
                shadow_map=shadow_map,
                shadow_weight=float(params["shadow_weight"]),
                crater_weight=float(params["crater_weight"]),
            )

            if path is None:
                raise RuntimeError("Yol bulunamadı.")

            self.finished.emit(
                {
                    "z": z,
                    "path": path,
                    "cost": float(cost),
                    "sr": start_row,
                    "sc": start_col,
                    "gr": goal_row,
                    "gc": goal_col,
                    "shadow_map": shadow_map,
                    "roi_rect": params["roi_rect"],
                }
            )

        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class MainWindow(QtWidgets.QMainWindow):
    """Uygulamanın ana arayüzü."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("DTM A* GUI - Refactored")
        self.resize(1650, 950)

        self.geotiff_path = None
        self.preview = None

        self.worker_thread = None
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        container = QtWidgets.QWidget()
        self.setCentralWidget(container)

        root_layout = QtWidgets.QHBoxLayout(container)

        left_layout = QtWidgets.QVBoxLayout()
        root_layout.addLayout(left_layout, stretch=3)

        file_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()

        browse_button = QtWidgets.QPushButton("GeoTIFF Seç")
        browse_button.clicked.connect(self.load_geotiff)

        file_row.addWidget(self.path_edit)
        file_row.addWidget(browse_button)
        left_layout.addLayout(file_row)

        mode_row = QtWidgets.QHBoxLayout()

        roi_button = QtWidgets.QPushButton("ROI")
        start_button = QtWidgets.QPushButton("START")
        goal_button = QtWidgets.QPushButton("GOAL")

        roi_button.clicked.connect(lambda: self.map_view.set_mode("roi"))
        start_button.clicked.connect(lambda: self.map_view.set_mode("start"))
        goal_button.clicked.connect(lambda: self.map_view.set_mode("goal"))

        mode_row.addWidget(roi_button)
        mode_row.addWidget(start_button)
        mode_row.addWidget(goal_button)
        left_layout.addLayout(mode_row)

        self.map_view = MapView()
        left_layout.addWidget(self.map_view)

        right_layout = QtWidgets.QVBoxLayout()
        root_layout.addLayout(right_layout, stretch=2)

        form = QtWidgets.QFormLayout()

        self.ds = QtWidgets.QSpinBox()
        self.ds.setRange(1, 64)
        self.ds.setValue(1)

        self.fill_iter = QtWidgets.QSpinBox()
        self.fill_iter.setRange(1, 100)
        self.fill_iter.setValue(8)

        self.row_win = QtWidgets.QSpinBox()
        self.row_win.setRange(3, 9999)
        self.row_win.setValue(101)

        self.col_win = QtWidgets.QSpinBox()
        self.col_win.setRange(3, 9999)
        self.col_win.setValue(101)

        self.sig_row = QtWidgets.QDoubleSpinBox()
        self.sig_row.setRange(0.1, 1000)
        self.sig_row.setValue(25.0)

        self.sig_col = QtWidgets.QDoubleSpinBox()
        self.sig_col.setRange(0.1, 1000)
        self.sig_col.setValue(25.0)

        self.med = QtWidgets.QSpinBox()
        self.med.setRange(1, 99)
        self.med.setValue(3)

        self.gau = QtWidgets.QDoubleSpinBox()
        self.gau.setRange(0.1, 100)
        self.gau.setValue(1.2)

        self.diag = QtWidgets.QCheckBox()
        self.diag.setChecked(True)

        self.sw = QtWidgets.QDoubleSpinBox()
        self.sw.setRange(0, 1000)
        self.sw.setValue(8.0)

        self.ue = QtWidgets.QDoubleSpinBox()
        self.ue.setRange(0, 1000)
        self.ue.setValue(1.5)

        self.shadow_weight = QtWidgets.QDoubleSpinBox()
        self.shadow_weight.setRange(0, 1000)
        self.shadow_weight.setValue(10.0)

        self.crater_weight = QtWidgets.QDoubleSpinBox()
        self.crater_weight.setRange(0, 1000)
        self.crater_weight.setValue(0.0)

        self.use_ms = QtWidgets.QCheckBox()

        self.ms = QtWidgets.QDoubleSpinBox()
        self.ms.setRange(0.001, 1000)
        self.ms.setValue(1.0)

        form.addRow("Downsample", self.ds)
        form.addRow("Fill Iter", self.fill_iter)
        form.addRow("Row Med Win", self.row_win)
        form.addRow("Col Med Win", self.col_win)
        form.addRow("Sigma Row", self.sig_row)
        form.addRow("Sigma Col", self.sig_col)
        form.addRow("Median Smooth", self.med)
        form.addRow("Gauss Sigma", self.gau)
        form.addRow("Diagonal", self.diag)
        form.addRow("Slope Weight", self.sw)
        form.addRow("Uphill Extra", self.ue)
        form.addRow("Shadow Weight", self.shadow_weight)
        form.addRow("Crater Weight", self.crater_weight)
        form.addRow("Use Max Slope", self.use_ms)
        form.addRow("Max Slope", self.ms)

        right_layout.addLayout(form)

        self.btn_run = QtWidgets.QPushButton("Path Hesapla + 3D Göster")
        self.btn_run.clicked.connect(self.start_compute)

        clear_button = QtWidgets.QPushButton("Seçimleri Temizle")
        clear_button.clicked.connect(self.map_view.clear_selection)

        right_layout.addWidget(self.btn_run)
        right_layout.addWidget(clear_button)

        self.status = QtWidgets.QLabel("Hazır")
        self.status.setWordWrap(True)
        right_layout.addWidget(self.status)

        self.plotter = QtInteractor(self)
        right_layout.addWidget(self.plotter.interactor, stretch=1)

    def load_geotiff(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "GeoTIFF seç",
            "",
            "GeoTIFF (*.tif *.tiff);;All files (*)",
        )

        if not path:
            return

        self.geotiff_path = path
        self.path_edit.setText(path)

        with rasterio.open(path) as src:
            max_preview = 1600
            scale = max(src.width / max_preview, src.height / max_preview, 1.0)

            preview_width = int(src.width / scale)
            preview_height = int(src.height / scale)

            preview = src.read(
                1,
                out_shape=(preview_height, preview_width),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)

            if src.nodata is not None:
                preview[np.isclose(preview, src.nodata)] = np.nan

        self.preview = preview

        image_u8 = robust_norm_to_uint8(preview)
        rgb = np.stack([image_u8, image_u8, image_u8], axis=-1).copy()

        qimage = QtGui.QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            3 * rgb.shape[1],
            QtGui.QImage.Format_RGB888,
        ).copy()

        self.map_view.set_image(QtGui.QPixmap.fromImage(qimage))
        self.status.setText("Raster yüklendi. ROI -> START -> GOAL seç.")

    def start_compute(self):
        if self.geotiff_path is None or self.preview is None:
            self.status.setText("Önce dosya seç.")
            return

        if self.map_view.roi_rect is None:
            self.status.setText("Önce ROI seç.")
            return

        if self.map_view.start_pt is None or self.map_view.goal_pt is None:
            self.status.setText("START ve GOAL seç.")
            return

        if self.worker_thread is not None:
            self.status.setText("İşlem sürüyor...")
            return

        payload = {
            "geotiff_path": self.geotiff_path,
            "preview_shape": self.preview.shape,
            "roi_rect": self.map_view.roi_rect,
            "start_pt": self.map_view.start_pt,
            "goal_pt": self.map_view.goal_pt,
            "downsample": int(self.ds.value()),
            "fill_iter": int(self.fill_iter.value()),
            "row_med_win": int(self.row_win.value()),
            "col_med_win": int(self.col_win.value()),
            "sigma_row": float(self.sig_row.value()),
            "sigma_col": float(self.sig_col.value()),
            "median_smooth": int(self.med.value()),
            "gauss_sigma": float(self.gau.value()),
            "diagonal": bool(self.diag.isChecked()),
            "slope_weight": float(self.sw.value()),
            "uphill_extra": float(self.ue.value()),
            "shadow_weight": float(self.shadow_weight.value()),
            "crater_weight": float(self.crater_weight.value()),
            "max_slope": float(self.ms.value()) if self.use_ms.isChecked() else None,
        }

        self.btn_run.setEnabled(False)

        self.worker_thread = QtCore.QThread()
        self.worker = ComputeWorker(payload)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda message: self.status.setText(message))
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.cleanup_worker)

        self.worker_thread.start()

    def on_finished(self, result: dict):
        try:
            z = result["z"]
            path = result["path"]
            cost = result["cost"]
            shadow_map = result.get("shadow_map")
            roi_rect = result.get("roi_rect")

            start_row = result["sr"]
            start_col = result["sc"]
            goal_row = result["gr"]
            goal_col = result["gc"]

            z_vis = z.astype(np.float32)
            z_vis = np.nan_to_num(z_vis, nan=np.nanmedian(z_vis))
            z_vis = z_vis - np.nanmin(z_vis)

            z_std = float(np.nanstd(z_vis))
            if z_std < 1e-6:
                z_std = 1.0

            z_vis = z_vis / z_std

            height, width = z_vis.shape
            yy, xx = np.mgrid[0:height, 0:width]

            grid = pv.StructuredGrid(
                xx.astype(np.float32),
                yy.astype(np.float32),
                np.zeros_like(xx, dtype=np.float32),
            )
            grid.point_data["elevation"] = z_vis.ravel(order="F")
            grid.set_active_scalars("elevation")

            warp_factor = 50.0
            warped = grid.warp_by_scalar("elevation", factor=warp_factor)

            path_points = np.zeros((len(path), 3), dtype=np.float32)
            for index, (row, col) in enumerate(path):
                path_points[index] = (col, row, z_vis[row, col] * warp_factor + 1.2)

            line = pv.lines_from_points(path_points, close=False)

            self.plotter.clear()
            self.plotter.set_background("#dcdcdc")
            self.plotter.enable_trackball_style()
            self.plotter.remove_all_lights()

            key_light = pv.Light(
                position=(width * 0.2, height * 0.2, 3000),
                focal_point=(width / 2, height / 2, 0),
                color="white",
                intensity=0.9,
            )
            fill_light = pv.Light(
                position=(width * 0.8, height * 0.9, 2000),
                focal_point=(width / 2, height / 2, 0),
                color="white",
                intensity=0.45,
            )

            self.plotter.add_light(key_light)
            self.plotter.add_light(fill_light)

            self.plotter.add_mesh(
                warped,
                color="#d9d9d3",
                smooth_shading=True,
                show_edges=False,
                specular=0.18,
                diffuse=0.88,
                ambient=0.08,
                scalars=None,
            )

            self.plotter.add_mesh(
                line,
                color="#e10000",
                line_width=6,
                render_lines_as_tubes=True,
            )

            self.plotter.add_points(
                np.array([[start_col, start_row, z_vis[start_row, start_col] * warp_factor + 1.8]], dtype=np.float32),
                color="#00aa00",
                point_size=15,
                render_points_as_spheres=True,
            )

            self.plotter.add_points(
                np.array([[goal_col, goal_row, z_vis[goal_row, goal_col] * warp_factor + 1.8]], dtype=np.float32),
                color="#0060ff",
                point_size=15,
                render_points_as_spheres=True,
            )

            self.plotter.add_text(
                "Mouse: Sol=Rotate  Orta=Pan  Teker=Zoom  R=Reset Camera",
                position="upper_left",
                font_size=11,
                color="black",
            )
            self.plotter.add_text(
                f"A* cost = {cost:.2f}",
                position="upper_right",
                font_size=12,
                color="black",
            )

            self.plotter.add_axes(line_width=2)
            self.plotter.show_grid(color="#bdbdbd")
            self.plotter.camera_position = "iso"
            self.plotter.reset_camera()
            self.plotter.render()

            if shadow_map is not None and roi_rect is not None:
                self.map_view.set_shadow_overlay(roi_rect, shadow_map)

            self.status.setText(f"Tamamlandı. node={len(path)} cost={cost:.3f}")

        except Exception as exc:
            self.status.setText(f"Render hatası: {exc}")

    def on_failed(self, error_text: str):
        self.status.setText(f"Hata:\n{error_text}")

    def cleanup_worker(self):
        self.btn_run.setEnabled(True)

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()