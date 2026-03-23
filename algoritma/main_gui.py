#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import heapq
import traceback
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import median_filter, gaussian_filter

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import pyvista as pv
from pyvistaqt import QtInteractor


# =========================
# Core processing
# =========================

def ensure_odd(n: int) -> int:
    n = int(max(3, n))
    return n if n % 2 == 1 else n + 1


def robust_norm_to_uint8(a):
    arr = a.copy().astype(np.float32)
    m = np.isfinite(arr)
    if not m.any():
        return np.zeros_like(arr, dtype=np.uint8)
    p2, p98 = np.percentile(arr[m], [2, 98])
    if p98 <= p2:
        p98 = p2 + 1e-6
    arr = np.clip((arr - p2) / (p98 - p2), 0, 1)
    arr[~m] = 0
    return (arr * 255).astype(np.uint8)


def fill_nans_iterative(a, iterations=6):
    out = a.astype(np.float32, copy=True)
    nanmask = ~np.isfinite(out)
    if not nanmask.any():
        return out
    vals = out[np.isfinite(out)]
    fallback = np.median(vals) if vals.size else 0.0
    out[nanmask] = fallback
    target = nanmask.copy()
    for _ in range(iterations):
        up = np.roll(out, -1, axis=0)
        down = np.roll(out, 1, axis=0)
        left = np.roll(out, -1, axis=1)
        right = np.roll(out, 1, axis=1)
        out[target] = 0.25 * (up[target] + down[target] + left[target] + right[target])
    return out


def destripe_1d_profiles(z, row_med_win, col_med_win, sigma_row, sigma_col):
    zz = z.astype(np.float32, copy=True)

    row_prof = np.median(zz, axis=1)
    row_win = ensure_odd(min(row_med_win, max(3, (len(row_prof)//2)*2-1)))
    row_tr = median_filter(row_prof, size=row_win, mode="nearest")
    row_res = row_prof - row_tr
    row_low = gaussian_filter(row_res, sigma=sigma_row, mode="nearest")
    zz -= 0.8 * (row_res - row_low)[:, None]

    col_prof = np.median(zz, axis=0)
    col_win = ensure_odd(min(col_med_win, max(3, (len(col_prof)//2)*2-1)))
    col_tr = median_filter(col_prof, size=col_win, mode="nearest")
    col_res = col_prof - col_tr
    col_low = gaussian_filter(col_res, sigma=sigma_col, mode="nearest")
    zz -= 0.8 * (col_res - col_low)[None, :]

    return zz


def final_smooth(z, median_smooth_size, gauss_sigma):
    zz = median_filter(z, size=median_smooth_size, mode="nearest")
    zz = gaussian_filter(zz, sigma=gauss_sigma, mode="nearest")
    return zz.astype(np.float32)


def astar_on_grid(z, start_rc, goal_rc, diagonal=True, slope_weight=8.0, uphill_extra=1.5, max_slope=None):
    H, W = z.shape
    sr, sc = start_rc
    gr, gc = goal_rc
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)] if diagonal else [(-1,0),(1,0),(0,-1),(0,1)]

    def h(r, c):
        return np.hypot(gr-r, gc-c)

    g = np.full((H, W), np.inf, dtype=np.float64)
    pr = np.full((H, W), -1, dtype=np.int32)
    pc = np.full((H, W), -1, dtype=np.int32)
    closed = np.zeros((H, W), dtype=bool)

    g[sr, sc] = 0.0
    pq = [(h(sr, sc), 0.0, sr, sc)]

    while pq:
        _, gcur, r, c = heapq.heappop(pq)
        if closed[r, c]:
            continue
        closed[r, c] = True
        if (r, c) == (gr, gc):
            break

        z0 = z[r, c]
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W or closed[rr, cc]:
                continue

            base_dist = np.hypot(dr, dc)
            dz = float(z[rr, cc] - z0)
            slope = abs(dz) / max(base_dist, 1e-6)

            if max_slope is not None and slope > max_slope:
                continue

            uphill_pen = max(0.0, dz) * uphill_extra
            step_cost = base_dist + slope_weight * slope + uphill_pen
            ng = gcur + step_cost

            if ng < g[rr, cc]:
                g[rr, cc] = ng
                pr[rr, cc] = r
                pc[rr, cc] = c
                heapq.heappush(pq, (ng + h(rr, cc), ng, rr, cc))

    if not np.isfinite(g[gr, gc]):
        return None, np.inf

    path = []
    r, c = gr, gc
    while True:
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        rr, cc = pr[r, c], pc[r, c]
        if rr < 0:
            return None, np.inf
        r, c = int(rr), int(cc)
    path.reverse()
    return path, g[gr, gc]


# =========================
# Map selector
# =========================

class MapView(QtWidgets.QGraphicsView):
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

        self.dragging = False
        self.drag_start = None
        self.img_w = 0
        self.img_h = 0

    def set_mode(self, mode):
        self.mode = mode

    def clear_selection(self):
        self.roi_rect = None
        self.start_pt = None
        self.goal_pt = None
        self.dragging = False
        self.drag_start = None

        if self.roi_item is not None:
            self.scene_.removeItem(self.roi_item); self.roi_item = None
        if self.start_item is not None:
            self.scene_.removeItem(self.start_item); self.start_item = None
        if self.goal_item is not None:
            self.scene_.removeItem(self.goal_item); self.goal_item = None

    def set_image(self, pixmap):
        self.scene_.clear()
        self.pix_item = self.scene_.addPixmap(pixmap)
        self.scene_.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.img_w, self.img_h = pixmap.width(), pixmap.height()
        self.clear_selection()
        self.resetTransform()
        self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, e):
        if self.pix_item is None:
            return super().wheelEvent(e)
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def mousePressEvent(self, e):
        if self.pix_item is None:
            return super().mousePressEvent(e)
        if e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)

        p = self.mapToScene(e.pos())
        x, y = int(p.x()), int(p.y())
        if not (0 <= x < self.img_w and 0 <= y < self.img_h):
            return

        if self.mode == "roi":
            self.dragging = True
            self.drag_start = (x, y)
            if self.roi_item is None:
                self.roi_item = self.scene_.addRect(x, y, 1, 1, QtGui.QPen(QtGui.QColor(255, 0, 0), 2))
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
                QtGui.QBrush(QtGui.QColor(0, 255, 0))
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
                QtGui.QBrush(QtGui.QColor(0, 0, 255))
            )

    def mouseMoveEvent(self, e):
        if self.pix_item is None:
            return super().mouseMoveEvent(e)
        if self.dragging and self.mode == "roi" and self.drag_start is not None:
            p = self.mapToScene(e.pos())
            x = int(np.clip(int(p.x()), 0, self.img_w - 1))
            y = int(np.clip(int(p.y()), 0, self.img_h - 1))
            x0, y0 = self.drag_start
            rx0, ry0 = min(x0, x), min(y0, y)
            rx1, ry1 = max(x0, x), max(y0, y)
            self.roi_rect = (rx0, ry0, rx1, ry1)
            if self.roi_item is not None:
                self.roi_item.setRect(rx0, ry0, max(1, rx1 - rx0), max(1, ry1 - ry0))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.pix_item is None:
            return super().mouseReleaseEvent(e)
        if e.button() == Qt.LeftButton and self.dragging and self.mode == "roi":
            self.dragging = False
            if self.roi_item is not None:
                r = self.roi_item.rect()
                x0, y0, x1, y1 = int(r.left()), int(r.top()), int(r.right()), int(r.bottom())
                x0 = int(np.clip(x0, 0, self.img_w - 1)); x1 = int(np.clip(x1, 0, self.img_w - 1))
                y0 = int(np.clip(y0, 0, self.img_h - 1)); y1 = int(np.clip(y1, 0, self.img_h - 1))
                if x1 > x0 and y1 > y0:
                    self.roi_rect = (x0, y0, x1, y1)
            return
        super().mouseReleaseEvent(e)


# =========================
# Worker thread
# =========================

class ComputeWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    @QtCore.pyqtSlot()
    def run(self):
        try:
            p = self.payload
            self.progress.emit("Raster okunuyor...")

            with rasterio.open(p["geotiff_path"]) as src:
                ph, pw = p["preview_shape"]
                fx = src.width / pw
                fy = src.height / ph

                x0p, y0p, x1p, y1p = p["roi_rect"]
                x0 = int(np.floor(x0p * fx)); x1 = int(np.ceil((x1p + 1) * fx))
                y0 = int(np.floor(y0p * fy)); y1 = int(np.ceil((y1p + 1) * fy))
                x0 = int(np.clip(x0, 0, src.width - 1)); x1 = int(np.clip(x1, x0 + 1, src.width))
                y0 = int(np.clip(y0, 0, src.height - 1)); y1 = int(np.clip(y1, y0 + 1, src.height))

                sxp, syp = p["start_pt"]
                gxp, gyp = p["goal_pt"]

                sc_full = int(np.clip(round(sxp * fx), x0, x1 - 1))
                sr_full = int(np.clip(round(syp * fy), y0, y1 - 1))
                gc_full = int(np.clip(round(gxp * fx), x0, x1 - 1))
                gr_full = int(np.clip(round(gyp * fy), y0, y1 - 1))

                ds = max(1, int(p["downsample"]))
                out_h = max(2, int((y1 - y0) / ds))
                out_w = max(2, int((x1 - x0) / ds))

                win = Window(x0, y0, x1 - x0, y1 - y0)
                z = src.read(1, window=win, out_shape=(out_h, out_w),
                             resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
                if src.nodata is not None:
                    z[np.isclose(z, src.nodata)] = np.nan

            self.progress.emit("Preprocess...")
            z = fill_nans_iterative(z, iterations=int(p["fill_iter"]))
            z = destripe_1d_profiles(
                z,
                row_med_win=ensure_odd(int(p["row_med_win"])),
                col_med_win=ensure_odd(int(p["col_med_win"])),
                sigma_row=float(p["sigma_row"]),
                sigma_col=float(p["sigma_col"])
            )
            z = final_smooth(
                z,
                median_smooth_size=ensure_odd(int(p["median_smooth"])),
                gauss_sigma=float(p["gauss_sigma"])
            )

            sr = int(np.clip(round((sr_full - y0) / ds), 0, z.shape[0] - 1))
            sc = int(np.clip(round((sc_full - x0) / ds), 0, z.shape[1] - 1))
            gr = int(np.clip(round((gr_full - y0) / ds), 0, z.shape[0] - 1))
            gc = int(np.clip(round((gc_full - x0) / ds), 0, z.shape[1] - 1))

            self.progress.emit("A* çalışıyor...")
            path, cost = astar_on_grid(
                z, (sr, sc), (gr, gc),
                diagonal=bool(p["diagonal"]),
                slope_weight=float(p["slope_weight"]),
                uphill_extra=float(p["uphill_extra"]),
                max_slope=p["max_slope"]
            )
            if path is None:
                raise RuntimeError("Yol bulunamadı.")

            self.finished.emit({
                "z": z, "path": path, "cost": float(cost),
                "sr": sr, "sc": sc, "gr": gr, "gc": gc
            })

        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# =========================
# Main GUI
# =========================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DTM A* GUI - Shadow Enhanced")
        self.resize(1650, 950)

        self.geotiff_path = None
        self.preview = None
        self.worker_thread = None
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        root = QtWidgets.QHBoxLayout(w)

        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, stretch=3)

        row_file = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        btn_browse = QtWidgets.QPushButton("GeoTIFF Seç")
        btn_browse.clicked.connect(self.load_geotiff)
        row_file.addWidget(self.path_edit)
        row_file.addWidget(btn_browse)
        left.addLayout(row_file)

        row_mode = QtWidgets.QHBoxLayout()
        btn_roi = QtWidgets.QPushButton("ROI")
        btn_start = QtWidgets.QPushButton("START")
        btn_goal = QtWidgets.QPushButton("GOAL")
        btn_roi.clicked.connect(lambda: self.map_view.set_mode("roi"))
        btn_start.clicked.connect(lambda: self.map_view.set_mode("start"))
        btn_goal.clicked.connect(lambda: self.map_view.set_mode("goal"))
        row_mode.addWidget(btn_roi); row_mode.addWidget(btn_start); row_mode.addWidget(btn_goal)
        left.addLayout(row_mode)

        self.map_view = MapView()
        left.addWidget(self.map_view)

        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, stretch=2)

        form = QtWidgets.QFormLayout()
        self.ds = QtWidgets.QSpinBox(); self.ds.setRange(1, 64); self.ds.setValue(1)
        self.fill_iter = QtWidgets.QSpinBox(); self.fill_iter.setRange(1, 100); self.fill_iter.setValue(8)
        self.row_win = QtWidgets.QSpinBox(); self.row_win.setRange(3, 9999); self.row_win.setValue(101)
        self.col_win = QtWidgets.QSpinBox(); self.col_win.setRange(3, 9999); self.col_win.setValue(101)
        self.sig_row = QtWidgets.QDoubleSpinBox(); self.sig_row.setRange(0.1, 1000); self.sig_row.setValue(25.0)
        self.sig_col = QtWidgets.QDoubleSpinBox(); self.sig_col.setRange(0.1, 1000); self.sig_col.setValue(25.0)
        self.med = QtWidgets.QSpinBox(); self.med.setRange(1, 99); self.med.setValue(3)
        self.gau = QtWidgets.QDoubleSpinBox(); self.gau.setRange(0.1, 100); self.gau.setValue(1.2)

        self.diag = QtWidgets.QCheckBox(); self.diag.setChecked(True)
        self.sw = QtWidgets.QDoubleSpinBox(); self.sw.setRange(0, 1000); self.sw.setValue(8.0)
        self.ue = QtWidgets.QDoubleSpinBox(); self.ue.setRange(0, 1000); self.ue.setValue(1.5)
        self.use_ms = QtWidgets.QCheckBox()
        self.ms = QtWidgets.QDoubleSpinBox(); self.ms.setRange(0.001, 1000); self.ms.setValue(1.0)

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
        form.addRow("Use Max Slope", self.use_ms)
        form.addRow("Max Slope", self.ms)
        right.addLayout(form)

        self.btn_run = QtWidgets.QPushButton("Path Hesapla + 3D Göster")
        self.btn_run.clicked.connect(self.start_compute)
        btn_clear = QtWidgets.QPushButton("Seçimleri Temizle")
        btn_clear.clicked.connect(self.map_view.clear_selection)
        right.addWidget(self.btn_run)
        right.addWidget(btn_clear)

        self.status = QtWidgets.QLabel("Hazır")
        self.status.setWordWrap(True)
        right.addWidget(self.status)

        self.plotter = QtInteractor(self)
        right.addWidget(self.plotter.interactor, stretch=1)

    def load_geotiff(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "GeoTIFF seç", "", "GeoTIFF (*.tif *.tiff);;All files (*)"
        )
        if not path:
            return
        self.geotiff_path = path
        self.path_edit.setText(path)

        with rasterio.open(path) as src:
            max_preview = 1600
            scale = max(src.width / max_preview, src.height / max_preview, 1.0)
            p_w = int(src.width / scale)
            p_h = int(src.height / scale)

            arr = src.read(1, out_shape=(p_h, p_w), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
            if src.nodata is not None:
                arr[np.isclose(arr, src.nodata)] = np.nan

        self.preview = arr
        img = robust_norm_to_uint8(arr)
        rgb = np.stack([img, img, img], axis=-1).copy()
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QtGui.QImage.Format_RGB888).copy()
        self.map_view.set_image(QtGui.QPixmap.fromImage(qimg))
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
            "preview_shape": self.preview.shape,  # (h,w)
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
            "max_slope": float(self.ms.value()) if self.use_ms.isChecked() else None
        }

        self.btn_run.setEnabled(False)
        self.worker_thread = QtCore.QThread()
        self.worker = ComputeWorker(payload)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda m: self.status.setText(m))
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.cleanup_worker)
        self.worker_thread.start()

    def on_finished(self, result):
        try:
            z = result["z"]
            path = result["path"]
            cost = result["cost"]
            sr, sc, gr, gc = result["sr"], result["sc"], result["gr"], result["gc"]

            # normalize for visibility
            zv = z.astype(np.float32)
            zv = np.nan_to_num(zv, nan=np.nanmedian(zv))
            zv = zv - np.nanmin(zv)
            zstd = float(np.nanstd(zv))
            if zstd < 1e-6:
                zstd = 1.0
            zv = zv / zstd

            H, W = zv.shape
            yy, xx = np.mgrid[0:H, 0:W]
            grid = pv.StructuredGrid(
                xx.astype(np.float32),
                yy.astype(np.float32),
                np.zeros_like(xx, dtype=np.float32)
            )
            grid.point_data["elevation"] = zv.ravel(order="F")
            grid.set_active_scalars("elevation")

            warp_factor = 50.0
            warped = grid.warp_by_scalar("elevation", factor=warp_factor)

            pts = np.zeros((len(path), 3), dtype=np.float32)
            for i, (r, c) in enumerate(path):
                pts[i] = (c, r, zv[r, c] * warp_factor + 1.2)
            line = pv.lines_from_points(pts, close=False)

            self.plotter.clear()
            self.plotter.set_background("#dcdcdc")
            self.plotter.enable_trackball_style()

            # --- Lighting for crater shadows ---
            self.plotter.remove_all_lights()
            light1 = pv.Light(
                position=(W * 0.2, H * 0.2, 3000),
                focal_point=(W / 2, H / 2, 0),
                color='white',
                intensity=0.9
            )
            light2 = pv.Light(
                position=(W * 0.8, H * 0.9, 2000),
                focal_point=(W / 2, H / 2, 0),
                color='white',
                intensity=0.45
            )
            self.plotter.add_light(light1)
            self.plotter.add_light(light2)

            self.plotter.add_mesh(
                warped,
                color="#d9d9d3",
                smooth_shading=True,
                show_edges=False,
                specular=0.18,
                diffuse=0.88,
                ambient=0.08,
                scalars=None
            )

            self.plotter.add_mesh(line, color="#e10000", line_width=6, render_lines_as_tubes=True)
            self.plotter.add_points(
                np.array([[sc, sr, zv[sr, sc] * warp_factor + 1.8]], dtype=np.float32),
                color="#00aa00", point_size=15, render_points_as_spheres=True
            )
            self.plotter.add_points(
                np.array([[gc, gr, zv[gr, gc] * warp_factor + 1.8]], dtype=np.float32),
                color="#0060ff", point_size=15, render_points_as_spheres=True
            )

            self.plotter.add_text(
                "Mouse: Sol=Rotate  Orta=Pan  Teker=Zoom  R=Reset Camera",
                position="upper_left", font_size=11, color="black"
            )
            self.plotter.add_text(f"A* cost = {cost:.2f}", position="upper_right", font_size=12, color="black")
            self.plotter.add_axes(line_width=2)
            self.plotter.show_grid(color="#bdbdbd")
            self.plotter.camera_position = "iso"
            self.plotter.reset_camera()
            self.plotter.render()

            self.status.setText(f"Tamamlandı. node={len(path)} cost={cost:.3f}")

        except Exception as e:
            self.status.setText(f"Render hatası: {e}")

    def on_failed(self, err):
        self.status.setText(f"Hata:\n{err}")

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
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()