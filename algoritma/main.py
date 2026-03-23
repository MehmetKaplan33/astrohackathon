#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import median_filter, gaussian_filter
import pyvista as pv
import cv2
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

GEOTIFF_PATH = "NAC_DTM_ALDERPEAK.tiff"
PREVIEW_MAX_SIZE = 1600
DOWNSAMPLE_FACTOR = 1

FILL_ITER = 8
ROW_MED_WIN = 101
COL_MED_WIN = 101
LOWFREQ_SIGMA_ROW = 25.0
LOWFREQ_SIGMA_COL = 25.0
MEDIAN_SMOOTH_SIZE = 3
GAUSS_SIGMA = 1.2
VERTICAL_EXAGGERATION = 1.0

DIAGONAL = True
SLOPE_WEIGHT = 8.0
UPHILL_EXTRA = 1.5
MAX_SLOPE = None

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

def destripe_1d_profiles(z):
    zz = z.astype(np.float32, copy=True)

    row_prof = np.median(zz, axis=1)
    row_win = ensure_odd(min(ROW_MED_WIN, max(3, (len(row_prof)//2)*2-1)))
    row_tr = median_filter(row_prof, size=row_win, mode="nearest")
    row_res = row_prof - row_tr
    row_low = gaussian_filter(row_res, sigma=LOWFREQ_SIGMA_ROW, mode="nearest")
    zz -= 0.8 * (row_res - row_low)[:, None]

    col_prof = np.median(zz, axis=0)
    col_win = ensure_odd(min(COL_MED_WIN, max(3, (len(col_prof)//2)*2-1)))
    col_tr = median_filter(col_prof, size=col_win, mode="nearest")
    col_res = col_prof - col_tr
    col_low = gaussian_filter(col_res, sigma=LOWFREQ_SIGMA_COL, mode="nearest")
    zz -= 0.8 * (col_res - col_low)[None, :]

    return zz

def final_smooth(z):
    zz = median_filter(z, size=MEDIAN_SMOOTH_SIZE, mode="nearest")
    zz = gaussian_filter(zz, sigma=GAUSS_SIGMA, mode="nearest")
    return zz.astype(np.float32)

def astar_on_grid(z, start_rc, goal_rc, diagonal=True):
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

            if MAX_SLOPE is not None and slope > MAX_SLOPE:
                continue

            uphill_pen = max(0.0, dz) * UPHILL_EXTRA
            step_cost = base_dist + SLOPE_WEIGHT * slope + uphill_pen
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

def path_polyline_from_rc(path_rc, z, zlift=0.3):
    pts = np.zeros((len(path_rc), 3), dtype=np.float32)
    for i, (r, c) in enumerate(path_rc):
        pts[i] = (c, r, z[r, c] * VERTICAL_EXAGGERATION + zlift)
    return pv.lines_from_points(pts, close=False)

def select_roi_and_points_opencv(src):
    W0, H0 = src.width, src.height
    scale = max(W0 / PREVIEW_MAX_SIZE, H0 / PREVIEW_MAX_SIZE, 1.0)
    p_w = int(W0 / scale)
    p_h = int(H0 / scale)

    preview = src.read(1, out_shape=(p_h, p_w), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
    if src.nodata is not None:
        preview[np.isclose(preview, src.nodata)] = np.nan

    img = robust_norm_to_uint8(preview)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print("\n[ROI] Dikdörtgen seç, ENTER/SPACE onay.")
    roi = cv2.selectROI("ROI sec", img_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("ROI sec")

    x, y, w, h = map(int, roi)
    if w <= 1 or h <= 1:
        raise RuntimeError("ROI seçilmedi veya çok küçük.")

    x0p, y0p, x1p, y1p = x, y, x+w-1, y+h-1
    x0p = int(np.clip(x0p, 0, p_w-1)); x1p = int(np.clip(x1p, 0, p_w-1))
    y0p = int(np.clip(y0p, 0, p_h-1)); y1p = int(np.clip(y1p, 0, p_h-1))

    vis = img_bgr.copy()
    cv2.rectangle(vis, (x0p, y0p), (x1p, y1p), (0, 0, 255), 2)
    cv2.putText(vis, "ROI tamam: START ve GOAL icin 2 tik", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    points = []
    def mouse_cb(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x0p <= mx <= x1p and y0p <= my <= y1p:
                points.append((mx, my))
                col = (0,255,0) if len(points) == 1 else (255,0,0)
                txt = "S" if len(points) == 1 else "G"
                cv2.circle(vis, (mx, my), 5, col, -1)
                cv2.putText(vis, txt, (mx+6, my-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    cv2.namedWindow("Start Goal sec")
    cv2.setMouseCallback("Start Goal sec", mouse_cb)
    while True:
        cv2.imshow("Start Goal sec", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyWindow("Start Goal sec")
            raise RuntimeError("Start/Goal seçimi iptal.")
        if len(points) >= 2:
            break
    cv2.destroyWindow("Start Goal sec")

    sp, gp = points[0], points[1]
    fx, fy = W0 / p_w, H0 / p_h

    x0 = int(np.floor(x0p * fx)); x1 = int(np.ceil((x1p+1) * fx))
    y0 = int(np.floor(y0p * fy)); y1 = int(np.ceil((y1p+1) * fy))
    x0 = int(np.clip(x0, 0, W0-1)); x1 = int(np.clip(x1, x0+1, W0))
    y0 = int(np.clip(y0, 0, H0-1)); y1 = int(np.clip(y1, y0+1, H0))

    sx_full = int(np.clip(round(sp[0]*fx), x0, x1-1))
    sy_full = int(np.clip(round(sp[1]*fy), y0, y1-1))
    gx_full = int(np.clip(round(gp[0]*fx), x0, x1-1))
    gy_full = int(np.clip(round(gp[1]*fy), y0, y1-1))

    return (x0, y0, x1, y1), (sy_full, sx_full), (gy_full, gx_full)

def main():
    with rasterio.open(GEOTIFF_PATH) as src:
        print(f"[INFO] Raster: {GEOTIFF_PATH} ({src.width}x{src.height})")
        (x0, y0, x1, y1), s_full_rc, g_full_rc = select_roi_and_points_opencv(src)

        win = Window(x0, y0, x1-x0, y1-y0)
        ds = max(1, int(DOWNSAMPLE_FACTOR))
        out_h = max(2, int((y1-y0)/ds))
        out_w = max(2, int((x1-x0)/ds))

        z = src.read(1, window=win, out_shape=(out_h, out_w),
                     resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        if src.nodata is not None:
            z[np.isclose(z, src.nodata)] = np.nan

    z = fill_nans_iterative(z, iterations=FILL_ITER)
    z = destripe_1d_profiles(z)
    z = final_smooth(z)

    sr_full, sc_full = s_full_rc
    gr_full, gc_full = g_full_rc
    ds = max(1, int(DOWNSAMPLE_FACTOR))

    sr = int(np.clip(round((sr_full - y0)/ds), 0, z.shape[0]-1))
    sc = int(np.clip(round((sc_full - x0)/ds), 0, z.shape[1]-1))
    gr = int(np.clip(round((gr_full - y0)/ds), 0, z.shape[0]-1))
    gc = int(np.clip(round((gc_full - x0)/ds), 0, z.shape[1]-1))

    path, cost = astar_on_grid(z, (sr, sc), (gr, gc), diagonal=DIAGONAL)
    if path is None:
        raise RuntimeError("Yol bulunamadi. MAX_SLOPE kisitini azalt/kapat.")

    print(f"[INFO] Path bulundu. node={len(path)} cost={cost:.3f}")

    H, W = z.shape
    yy, xx = np.mgrid[0:H, 0:W]
    grid = pv.StructuredGrid(xx.astype(np.float32), yy.astype(np.float32), np.zeros_like(xx, dtype=np.float32))
    grid.point_data["elevation"] = z.ravel(order="F")
    grid.set_active_scalars("elevation")
    warped = grid.warp_by_scalar("elevation", factor=VERTICAL_EXAGGERATION)

    line = path_polyline_from_rc(path, z, zlift=0.35)

    p = pv.Plotter(window_size=(1400, 900))
    p.set_background("white")
    p.enable_trackball_style()   # EN KOLAY KAMERA KONTROLÜ

    p.add_mesh(
        warped,
        color="#cfcfcf",
        show_edges=False,
        smooth_shading=True,
        specular=0.08,
        diffuse=0.9,
        ambient=0.25,
        scalars=None
    )

    p.add_mesh(line, color="red", line_width=5, render_lines_as_tubes=True)

    p.add_points(
        np.array([[sc, sr, z[sr, sc]*VERTICAL_EXAGGERATION + 0.45]], dtype=np.float32),
        color="green", point_size=16, render_points_as_spheres=True
    )
    p.add_points(
        np.array([[gc, gr, z[gr, gc]*VERTICAL_EXAGGERATION + 0.45]], dtype=np.float32),
        color="blue", point_size=16, render_points_as_spheres=True
    )

    p.add_text(
        "Mouse: Sol=Rotate | Orta=Pan | Teker=Zoom | R=Reset Camera",
        position="upper_left",
        font_size=11,
        color="black"
    )
    p.add_text(f"A* cost = {cost:.2f}", position="upper_right", font_size=12, color="black")

    p.add_axes(line_width=2)
    p.show_grid(color="lightgray")
    p.camera_position = "iso"
    p.reset_camera()

    p.show("DTM 3D + A* (Easy Control)")

if __name__ == "__main__":
    main()