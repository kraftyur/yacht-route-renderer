from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from geopy.distance import geodesic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uuid
import os
import math
import requests
from io import BytesIO
from PIL import Image
import json
from shapely.geometry import shape, Point, LineString
from shapely.ops import unary_union
from shapely.prepared import prep

app = FastAPI(title="Yacht Route Map Renderer")

os.makedirs("static/maps", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

TILE_SIZE = 256
OSM_TILE_URL = "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
USER_AGENT = "yacht-route-renderer/0.1"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

LAND_FILE = "data/land_polygons.geojson"

LAND_GEOM = None
LAND_PREP = None


def load_land_mask():
    global LAND_GEOM, LAND_PREP

    if LAND_GEOM is not None:
        return

    if not os.path.exists(LAND_FILE):
        print(f"[land-mask] file not found: {LAND_FILE}")
        return

    with open(LAND_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    geoms = []

    if data.get("type") == "FeatureCollection":
        for feature in data.get("features", []):
            geom = feature.get("geometry")
            if geom:
                geoms.append(shape(geom))
    else:
        geoms.append(shape(data))

    if not geoms:
        print("[land-mask] no geometries loaded")
        return

    LAND_GEOM = unary_union(geoms)
    LAND_PREP = prep(LAND_GEOM)

    print("[land-mask] loaded successfully")

load_land_mask()

class Waypoint(BaseModel):
    name: str
    lat: float
    lon: float
    type: str = "waypoint"


class RouteRequest(BaseModel):
    title: str
    waypoints: List[Waypoint]
    output_format: str = "png"
    style: str = "nautical_schematic"
    show_labels: bool = True
    show_nm_distances: bool = True
    show_route_lines: bool = True
    show_coastline: bool = True

    show_direction_arrows: bool = True

    # auto | fixed | straight
    curve_mode: str = "auto"

    # используется как базовый/предпочтительный изгиб
    route_curvature: float = 0.14


def nm_distance(a: Waypoint, b: Waypoint) -> float:
    km = geodesic((a.lat, a.lon), (b.lat, b.lon)).km
    return km / 1.852


def latlon_to_tile_xy(lat: float, lon: float, zoom: int):
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def estimate_zoom(min_lon, min_lat, max_lon, max_lat, max_tiles=20):
    for z in range(12, 1, -1):
        x_left, y_top = latlon_to_tile_xy(max_lat, min_lon, z)
        x_right, y_bottom = latlon_to_tile_xy(min_lat, max_lon, z)

        x_start = math.floor(x_left)
        x_end = math.floor(x_right)
        y_start = math.floor(y_top)
        y_end = math.floor(y_bottom)

        tiles_count = (x_end - x_start + 1) * (y_end - y_start + 1)
        if tiles_count <= max_tiles:
            return z
    return 3

def fetch_tile(z: int, x: int, y: int) -> Image.Image:
    max_index = 2 ** z
    x = x % max_index

    if y < 0 or y >= max_index:
        return Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (240, 240, 240, 255))

    url = OSM_TILE_URL.format(z=z, x=x, y=y)
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGBA")
    except Exception:
        return Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (240, 240, 240, 255))

def build_osm_background(min_lon, min_lat, max_lon, max_lat):
    zoom = estimate_zoom(min_lon, min_lat, max_lon, max_lat)

    x_left_f, y_top_f = latlon_to_tile_xy(max_lat, min_lon, zoom)
    x_right_f, y_bottom_f = latlon_to_tile_xy(min_lat, max_lon, zoom)

    x_start = math.floor(x_left_f)
    x_end = math.floor(x_right_f)
    y_start = math.floor(y_top_f)
    y_end = math.floor(y_bottom_f)

    tiles_w = x_end - x_start + 1
    tiles_h = y_end - y_start + 1

    stitched = Image.new("RGBA", (tiles_w * TILE_SIZE, tiles_h * TILE_SIZE))

    for ty in range(y_start, y_end + 1):
        for tx in range(x_start, x_end + 1):
            tile = fetch_tile(zoom, tx, ty)
            px = (tx - x_start) * TILE_SIZE
            py = (ty - y_start) * TILE_SIZE
            stitched.paste(tile, (px, py))

    left = int((x_left_f - x_start) * TILE_SIZE)
    top = int((y_top_f - y_start) * TILE_SIZE)
    right = int((x_right_f - x_start) * TILE_SIZE)
    bottom = int((y_bottom_f - y_start) * TILE_SIZE)

    right = max(right, left + 10)
    bottom = max(bottom, top + 10)

    cropped = stitched.crop((left, top, right, bottom))

    meta = {
        "zoom": zoom,
        "x_start": x_start,
        "y_start": y_start,
        "crop_left": left,
        "crop_top": top,
        "width": cropped.size[0],
        "height": cropped.size[1],
    }

    return cropped, meta

def latlon_to_image_px(lat: float, lon: float, meta: dict):
    zoom = meta["zoom"]
    x_tile_f, y_tile_f = latlon_to_tile_xy(lat, lon, zoom)

    x_px_global = x_tile_f * TILE_SIZE
    y_px_global = y_tile_f * TILE_SIZE

    crop_x0 = meta["x_start"] * TILE_SIZE + meta["crop_left"]
    crop_y0 = meta["y_start"] * TILE_SIZE + meta["crop_top"]

    x_px = x_px_global - crop_x0
    y_px = y_px_global - crop_y0

    return x_px, y_px

def build_bezier_curve_lonlat(a_lon, a_lat, b_lon, b_lat, curvature, steps=40):
    """
    Строит квадратичную bezier-дугу между двумя точками в lon/lat.
    curvature > 0 и < 0 задают сторону и силу изгиба.
    """
    mean_lat_rad = math.radians((a_lat + b_lat) / 2)
    scale_x = max(math.cos(mean_lat_rad), 0.25)

    x1, y1 = a_lon * scale_x, a_lat
    x2, y2 = b_lon * scale_x, b_lat

    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    seg_len = math.hypot(dx, dy) or 1.0

    nx, ny = -dy / seg_len, dx / seg_len
    offset = seg_len * curvature

    cx, cy = mx + nx * offset, my + ny * offset

    pts = []
    for i in range(steps + 1):
        t = i / steps
        xt = ((1 - t) ** 2) * x1 + 2 * (1 - t) * t * cx + (t ** 2) * x2
        yt = ((1 - t) ** 2) * y1 + 2 * (1 - t) * t * cy + (t ** 2) * y2
        pts.append((xt / scale_x, yt))

    return pts

def curve_score(curve_lonlat, curvature):
    """
    Меньше score = лучше.
    Сильный штраф за сушу, слабее — за близость к суше, ещё слабее — за слишком сильный изгиб.
    """
    if LAND_GEOM is None or LAND_PREP is None:
        return abs(curvature) * 50

    interior = curve_lonlat[3:-3] if len(curve_lonlat) > 6 else curve_lonlat[1:-1]
    if len(interior) < 2:
        return abs(curvature) * 50

    score = 0.0

    line = LineString(interior)

    # очень большой штраф, если дуга реально идёт по суше
    if LAND_PREP.intersects(line):
        score += 1_000_000

    min_dist = float("inf")
    on_land_points = 0

    for lon, lat in interior:
        p = Point(lon, lat)

        if LAND_PREP.contains(p):
            on_land_points += 1

        d = LAND_GEOM.distance(p)  # в градусах; для визуальной эвристики нам хватает
        if d < min_dist:
            min_dist = d

    score += on_land_points * 100_000

    # мягкий штраф за слишком близкий проход к суше
    near_threshold = 0.01  # примерно ~1 км по широте, для эвристики ок
    if min_dist < near_threshold:
        score += (near_threshold - min_dist) * 40_000

    # лёгкий штраф за слишком театральный изгиб
    score += abs(curvature) * 200

    return score

def choose_curve_lonlat(a_wp, b_wp, curve_mode="auto", preferred_curvature=0.14):
    """
    Возвращает (curve_lonlat, chosen_curvature)
    """
    if curve_mode == "straight":
        curve = build_bezier_curve_lonlat(a_wp.lon, a_wp.lat, b_wp.lon, b_wp.lat, 0.0)
        return curve, 0.0

    if curve_mode == "fixed" or LAND_GEOM is None or LAND_PREP is None:
        curve = build_bezier_curve_lonlat(
            a_wp.lon, a_wp.lat,
            b_wp.lon, b_wp.lat,
            preferred_curvature
        )
        return curve, preferred_curvature

    candidates = [-0.22, -0.14, -0.08, 0.08, 0.14, 0.22]

    best_curve = None
    best_curvature = None
    best_score = None

    for curv in candidates:
        curve = build_bezier_curve_lonlat(
            a_wp.lon, a_wp.lat,
            b_wp.lon, b_wp.lat,
            curv
        )
        score = curve_score(curve, curv)

        if best_score is None or score < best_score:
            best_score = score
            best_curve = curve
            best_curvature = curv

    return best_curve, best_curvature

def point_label_position(i, point_pixels):
    """
    Смещает подпись точки в сторону от линии, чтобы меньше налезало на маршрут.
    """
    x, y = point_pixels[i]
    n = len(point_pixels)

    if n == 1:
        return x + 18, y - 14, "left"

    if i == 0:
        x2, y2 = point_pixels[i + 1]
        dx, dy = x2 - x, y2 - y
    elif i == n - 1:
        x1, y1 = point_pixels[i - 1]
        dx, dy = x - x1, y - y1
    else:
        x1, y1 = point_pixels[i - 1]
        x2, y2 = point_pixels[i + 1]
        dx, dy = x2 - x1, y2 - y1

    seg_len = math.hypot(dx, dy) or 1.0
    tx, ty = dx / seg_len, dy / seg_len
    nx, ny = -ty, tx

    side = 1 if i % 2 == 0 else -1

    label_x = x + nx * 24 * side + tx * 8
    label_y = y + ny * 24 * side + ty * 8

    ha = "left" if label_x >= x else "right"
    return label_x, label_y, ha

@app.get("/")
def root():
    return {"status": "ok", "service": "Yacht Route Map Renderer"}


@app.post("/render-route-map")
def render_route_map(req: RouteRequest):
    if len(req.waypoints) < 2:
        return {"error": "At least two waypoints are required"}

    route_id = str(uuid.uuid4())[:8]
    filename = f"route-{route_id}.png"
    filepath = f"static/maps/{filename}"

    lats = [p.lat for p in req.waypoints]
    lons = [p.lon for p in req.waypoints]

    margin_lat = max(0.2, (max(lats) - min(lats)) * 0.20)
    margin_lon = max(0.2, (max(lons) - min(lons)) * 0.20)

    min_lon = min(lons) - margin_lon
    max_lon = max(lons) + margin_lon
    min_lat = min(lats) - margin_lat
    max_lat = max(lats) + margin_lat

    bg_img, meta = build_osm_background(min_lon, min_lat, max_lon, max_lat)

    width, height = bg_img.size
    fig_w = 12
    fig_h = fig_w * (height / width)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)

    # рисуем картинку в пиксельной системе координат
    ax.imshow(bg_img, zorder=0)

    point_pixels = [latlon_to_image_px(p.lat, p.lon, meta) for p in req.waypoints]

segments = []
for a_wp, b_wp in zip(req.waypoints[:-1], req.waypoints[1:]):
    curve_lonlat, chosen_curvature = choose_curve_lonlat(
        a_wp,
        b_wp,
        curve_mode=req.curve_mode,
        preferred_curvature=req.route_curvature,
    )

    curve_pixels = [latlon_to_image_px(lat, lon, meta) for lon, lat in curve_lonlat]

    segments.append({
        "a_wp": a_wp,
        "b_wp": b_wp,
        "curve_lonlat": curve_lonlat,
        "curve_pixels": curve_pixels,
        "curvature": chosen_curvature,
    })

ROUTE_COLOR = "#1f77b4"

# 1) линии маршрута + стрелки
if req.show_route_lines:
    for seg in segments:
        xs = [pt[0] for pt in seg["curve_pixels"]]
        ys = [pt[1] for pt in seg["curve_pixels"]]

        ax.plot(xs, ys, linewidth=2.4, color=ROUTE_COLOR, zorder=5)

        if req.show_direction_arrows and len(seg["curve_pixels"]) >= 6:
            arrow_idx = int(len(seg["curve_pixels"]) * 0.68)
            arrow_idx = max(2, min(len(seg["curve_pixels"]) - 3, arrow_idx))

            x0, y0 = seg["curve_pixels"][arrow_idx - 1]
            x1, y1 = seg["curve_pixels"][arrow_idx + 1]

            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=2.4,
                    color=ROUTE_COLOR,
                    mutation_scale=14,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=6,
            )

# 2) иконки точек + подписи точек
for i, (p, (x, y)) in enumerate(zip(req.waypoints, point_pixels)):
    if p.type == "anchorage":
        symbol = "⚓"
    else:
        symbol = "⛵"

    ax.text(
        x,
        y,
        symbol,
        fontsize=14,
        ha="center",
        va="center",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="black", alpha=0.95),
    )

    if req.show_labels:
        lx, ly, ha = point_label_position(i, point_pixels)

        ax.text(
            lx,
            ly,
            p.name,
            fontsize=8,
            ha=ha,
            va="center",
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.88),
        )

# 3) подписи расстояний со смещением в сторону дуги
if req.show_nm_distances:
    for seg in segments:
        curve_pixels = seg["curve_pixels"]
        mid_idx = len(curve_pixels) // 2
        i1 = max(0, mid_idx - 1)
        i2 = min(len(curve_pixels) - 1, mid_idx + 1)

        mid_x, mid_y = curve_pixels[mid_idx]

        dx = curve_pixels[i2][0] - curve_pixels[i1][0]
        dy = curve_pixels[i2][1] - curve_pixels[i1][1]
        seg_len = math.hypot(dx, dy) or 1.0

        nx, ny = -dy / seg_len, dx / seg_len
        side = 1 if seg["curvature"] >= 0 else -1

        label_x = mid_x + nx * 16 * side
        label_y = mid_y + ny * 16 * side

        dist = nm_distance(seg["a_wp"], seg["b_wp"])

        ax.text(
            label_x,
            label_y,
            f"{dist:.0f} NM",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.90),
            zorder=8,
        )

    ax.set_title(f"{req.title}", fontsize=14)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # инвертируем Y, чтобы совпадало с изображением
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    base_url = "https://yacht-route-renderer.onrender.com"

    return {
        "image_url": f"{base_url}/static/maps/{filename}",
        "format": "png",
        "zoom": meta["zoom"],
    }
