import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Circle
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from geopy.distance import geodesic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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
import time


app = FastAPI(title="Yacht Route Map Renderer")

os.makedirs("static/maps", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

TILE_SIZE = 256
CARTO_TILE_SUBDOMAINS = ["a", "b", "c", "d"]
VOYAGER_TILE_URL = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OPENSEAMAP_SEAMARK_TILE_URL = "https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png"
USER_AGENT = "yacht-route-renderer/0.1"

def pick_subdomain(x: int, y: int):
    return CARTO_TILE_SUBDOMAINS[(x + y) % len(CARTO_TILE_SUBDOMAINS)]

def format_tile_url(tile_url: str, z: int, x: int, y: int):
    if "{s}" in tile_url:
        s = pick_subdomain(x, y)
        return tile_url.format(s=s, z=z, x=x, y=y)

    return tile_url.format(z=z, x=x, y=y)
    
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

LAND_FILE = "data/land_polygons.geojson"

LAND_GEOM = None
LAND_PREP = None

ANCHOR_ICON_PATH = "static/icons/anchor-64.png"
BOAT_ICON_PATH = "static/icons/boat-64.png"

ANCHOR_ICON = None
BOAT_ICON = None

def load_icons():
    global ANCHOR_ICON, BOAT_ICON

    if os.path.exists(ANCHOR_ICON_PATH):
        ANCHOR_ICON = mpimg.imread(ANCHOR_ICON_PATH)

    if os.path.exists(BOAT_ICON_PATH):
        BOAT_ICON = mpimg.imread(BOAT_ICON_PATH)

load_icons()

def draw_point_icon(ax, x, y, icon_img, circle_radius=13, zoom=0.22):
    # белый круг
    circle = Circle(
        (x, y),
        radius=circle_radius,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.95,
        zorder=7,
    )
    ax.add_patch(circle)

    # сама иконка
    if icon_img is not None:
        imagebox = OffsetImage(icon_img, zoom=zoom)
        ab = AnnotationBbox(
            imagebox,
            (x, y),
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=8,
        )
        ax.add_artist(ab)

def load_land_mask():
    global LAND_GEOM, LAND_PREP

    if LAND_GEOM is not None:
        return

    if not os.path.exists(LAND_FILE):
        print(f"[land-mask] file not found: {LAND_FILE}")
        return

    try:
        with open(LAND_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            print(f"[land-mask] file is empty: {LAND_FILE}")
            return

        if not raw.startswith("{"):
            print(f"[land-mask] file does not look like GeoJSON: {LAND_FILE}")
            print(f"[land-mask] first 120 chars: {raw[:120]}")
            return

        data = json.loads(raw)

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

    except Exception as e:
        print(f"[land-mask] failed to load {LAND_FILE}: {e}")
        LAND_GEOM = None
        LAND_PREP = None

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

    # presentation | route | marine | local
    map_detail: str = "route"

    # Если эти поля явно переданы, они переопределяют map_detail.
    show_labels: bool | None = None
    show_nm_distances: bool | None = None
    show_route_lines: bool = True
    show_coastline: bool = True
    show_seamarks: bool | None = None
    show_direction_arrows: bool = True

    # auto | fixed | straight
    curve_mode: str = "auto"

    # Если явно передан, переопределяет map_detail.
    route_curvature: float | None = None

def apply_map_detail(req: RouteRequest):
    """
    Возвращает словарь с финальными настройками карты.
    Явно переданные параметры имеют приоритет над map_detail.
    """
    presets = {
        "presentation": {
            "show_labels": True,
            "show_nm_distances": True,
            "show_seamarks": False,
            "route_curvature": 0.12,
            "min_margin_deg": 0.12,
            "margin_factor": 0.18,
            "max_zoom": 11,
            "max_tiles": 24,
        },
        "route": {
            "show_labels": True,
            "show_nm_distances": True,
            "show_seamarks": False,
            "route_curvature": 0.14,
            "min_margin_deg": 0.12,
            "margin_factor": 0.20,
            "max_zoom": 12,
            "max_tiles": 24,
        },
        "marine": {
            "show_labels": True,
            "show_nm_distances": True,
            "show_seamarks": True,
            "route_curvature": 0.14,
            "min_margin_deg": 0.12,
            "margin_factor": 0.20,
            "max_zoom": 12,
            "max_tiles": 24,
        },
        "local": {
            "show_labels": True,
            "show_nm_distances": False,
            "show_seamarks": True,
            "route_curvature": 0.06,
            "min_margin_deg": 0.015,
            "margin_factor": 0.80,
            "max_zoom": 15,
            "max_tiles": 64,
        },
    }

    detail = req.map_detail if req.map_detail in presets else "route"
    cfg = presets[detail].copy()

    if req.show_labels is not None:
        cfg["show_labels"] = req.show_labels

    if req.show_nm_distances is not None:
        cfg["show_nm_distances"] = req.show_nm_distances

    if req.show_seamarks is not None:
        cfg["show_seamarks"] = req.show_seamarks

    if req.route_curvature is not None:
        cfg["route_curvature"] = req.route_curvature

    cfg["map_detail"] = detail
    return cfg


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


def estimate_zoom(min_lon, min_lat, max_lon, max_lat, max_tiles=24, max_zoom=12):
    for z in range(max_zoom, 1, -1):
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

import time

def fetch_tile_from_url(tile_url: str, z: int, x: int, y: int, retries=2) -> Image.Image:
    max_index = 2 ** z
    x = x % max_index

    if y < 0 or y >= max_index:
        return Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (240, 240, 240, 0))

    url = format_tile_url(tile_url, z, x, y)
    last_error = None

    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)

            if resp.status_code == 404:
                last_error = f"404 for {url}"
                break

            resp.raise_for_status()

            if not resp.content:
                last_error = f"empty response for {url}"
                continue

            return Image.open(BytesIO(resp.content)).convert("RGBA")

        except Exception as e:
            last_error = e
            time.sleep(0.4 * (attempt + 1))

    print(f"[tile] failed z={z} x={x} y={y} url={url} err={last_error}")
    return Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (240, 240, 240, 0))


def fetch_base_tile(z: int, x: int, y: int, tile_url: str) -> tuple[Image.Image, bool]:
    tile = fetch_tile_from_url(tile_url, z, x, y)

    if tile.getbbox() is None:
        fallback = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (235, 235, 235, 255))
        return fallback, False

    return tile, True
    

def fetch_overlay_tile(tile_url: str, z: int, x: int, y: int) -> Image.Image:
    # overlay должен быть прозрачным, если тайл не загрузился
    return fetch_tile_from_url(tile_url, z, x, y)


def build_background_for_provider(
    min_lon,
    min_lat,
    max_lon,
    max_lat,
    tile_url: str,
    provider_name: str,
    show_seamarks=False,
    max_tiles=24,
    max_zoom=12,
    forced_zoom=None,
):
    zoom = forced_zoom if forced_zoom is not None else estimate_zoom(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        max_tiles=max_tiles,
        max_zoom=max_zoom,
    )

    x_left_f, y_top_f = latlon_to_tile_xy(max_lat, min_lon, zoom)
    x_right_f, y_bottom_f = latlon_to_tile_xy(min_lat, max_lon, zoom)

    x_start = math.floor(x_left_f)
    x_end = math.floor(x_right_f)
    y_start = math.floor(y_top_f)
    y_end = math.floor(y_bottom_f)

    tiles_w = x_end - x_start + 1
    tiles_h = y_end - y_start + 1

    stitched = Image.new("RGBA", (tiles_w * TILE_SIZE, tiles_h * TILE_SIZE))

    failed_tiles = 0
    loaded_tiles = 0

    for ty in range(y_start, y_end + 1):
        for tx in range(x_start, x_end + 1):
            base_tile, ok = fetch_base_tile(zoom, tx, ty, tile_url)

            if ok:
                loaded_tiles += 1
            else:
                failed_tiles += 1

            if show_seamarks:
                seamark_tile = fetch_overlay_tile(OPENSEAMAP_SEAMARK_TILE_URL, zoom, tx, ty)
                base_tile.alpha_composite(seamark_tile)

            px = (tx - x_start) * TILE_SIZE
            py = (ty - y_start) * TILE_SIZE
            stitched.paste(base_tile, (px, py))

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
        "failed_tiles": failed_tiles,
        "loaded_tiles": loaded_tiles,
        "base_provider": provider_name,
    }

    return cropped, meta


def build_osm_background(
    min_lon,
    min_lat,
    max_lon,
    max_lat,
    show_seamarks=False,
    max_tiles=24,
    max_zoom=12,
):
    # Сначала пробуем Voyager
    bg_img, meta = build_background_for_provider(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        tile_url=VOYAGER_TILE_URL,
        provider_name="voyager",
        show_seamarks=show_seamarks,
        max_tiles=max_tiles,
        max_zoom=max_zoom,
    )

    # Если хотя бы один тайл не загрузился — полностью пересобираем в OSM
    if meta["failed_tiles"] > 0:
        print(
            f"[map] voyager incomplete "
            f"(loaded={meta['loaded_tiles']}, failed={meta['failed_tiles']}). "
            f"Rebuilding full map with OSM."
        )

        bg_img, meta = build_background_for_provider(
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            tile_url=OSM_TILE_URL,
            provider_name="osm",
            show_seamarks=show_seamarks,
            max_tiles=max_tiles,
            max_zoom=max_zoom,
            forced_zoom=meta["zoom"],   # тот же zoom
        )

    return bg_img, meta
    

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


def same_waypoint(a: Waypoint, b: Waypoint, tol=0.0001) -> bool:
    return abs(a.lat - b.lat) <= tol and abs(a.lon - b.lon) <= tol


def normalize_route_waypoints(waypoints: list[Waypoint]):
    """
    Если маршрут кольцевой и последняя точка совпадает с первой,
    убираем последнюю из списка отображаемых точек, но помечаем маршрут как closed.
    """
    if len(waypoints) >= 3 and same_waypoint(waypoints[0], waypoints[-1]):
        return waypoints[:-1], True

    return waypoints, False


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Yacht Route Map Renderer",
        "land_mask_loaded": LAND_GEOM is not None
    }

@app.post("/render-route-map")
def render_route_map(req: RouteRequest):
    if len(req.waypoints) < 2:
        return {"error": "At least two waypoints are required"}
    
    render_waypoints, is_closed_route = normalize_route_waypoints(req.waypoints)

    if len(render_waypoints) < 2:
        return {"error": "At least two distinct waypoints are required"}
    
    cfg = apply_map_detail(req)
    
    route_id = str(uuid.uuid4())[:8]
    filename = f"route-{route_id}.png"
    filepath = f"static/maps/{filename}"

    lats = [p.lat for p in render_waypoints]
    lons = [p.lon for p in render_waypoints]

    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    
    margin_factor = cfg["margin_factor"]
    min_margin_deg = cfg["min_margin_deg"]
    
    margin_lat = max(min_margin_deg, lat_span * margin_factor)
    margin_lon = max(min_margin_deg, lon_span * margin_factor)

    min_lon = min(lons) - margin_lon
    max_lon = max(lons) + margin_lon
    min_lat = min(lats) - margin_lat
    max_lat = max(lats) + margin_lat

    bg_img, meta = build_osm_background(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        show_seamarks=cfg["show_seamarks"],
        max_tiles=cfg["max_tiles"],
        max_zoom=cfg["max_zoom"],
    )
    width, height = bg_img.size
    fig_w = 12
    fig_h = fig_w * (height / width)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)

    # рисуем картинку в пиксельной системе координат
    ax.imshow(bg_img, zorder=0)

    point_pixels = [latlon_to_image_px(p.lat, p.lon, meta) for p in render_waypoints]

    segments = []
    
    segment_waypoints = render_waypoints + [render_waypoints[0]] if is_closed_route else render_waypoints
    
    for a_wp, b_wp in zip(segment_waypoints[:-1], segment_waypoints[1:]):
        curve_lonlat, chosen_curvature = choose_curve_lonlat(
            a_wp,
            b_wp,
            curve_mode=req.curve_mode,
            preferred_curvature=cfg["route_curvature"],
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

    # линии маршрута + стрелки
    if req.show_route_lines:
        for seg in segments:
            xs = [pt[0] for pt in seg["curve_pixels"]]
            ys = [pt[1] for pt in seg["curve_pixels"]]
    
            ax.plot(xs, ys, linewidth=2.4, color=ROUTE_COLOR, zorder=5)
    
            seg["arrow_label_pos"] = None
    
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
                        mutation_scale=22,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=6,
                )
    
                # считаем позицию лейбла рядом со стрелкой
                dx = x1 - x0
                dy = y1 - y0
                seg_len = math.hypot(dx, dy) or 1.0
    
                nx = -dy / seg_len
                ny = dx / seg_len
    
                side = 1 if seg["curvature"] >= 0 else -1
    
                label_x = ((x0 + x1) / 2) + nx * 28 * side
                label_y = ((y0 + y1) / 2) + ny * 28 * side
    
                seg["arrow_label_pos"] = (label_x, label_y)

    # иконки точек + подписи точек
    for i, (p, (x, y)) in enumerate(zip(render_waypoints, point_pixels)):
        if p.type == "anchorage":
            draw_point_icon(ax, x, y, ANCHOR_ICON, circle_radius=8, zoom=0.20)
        else:
            draw_point_icon(ax, x, y, BOAT_ICON, circle_radius=8, zoom=0.19)
    
        if cfg["show_labels"]:
            lx, ly, ha = point_label_position(i, point_pixels)
    
            label_text = ax.text(
                lx,
                ly,
                p.name,
                fontsize=10,
                ha=ha,
                va="center",
                zorder=9,
                color="black",
            )
            
            label_text.set_path_effects([
                pe.Stroke(linewidth=3.0, foreground="white"),
                pe.Normal(),
            ])

    # подписи расстояний со смещением в сторону дуги
    if cfg["show_nm_distances"]:
        for seg in segments:
            dist = nm_distance(seg["a_wp"], seg["b_wp"])
    
            if seg.get("arrow_label_pos") is not None:
                label_x, label_y = seg["arrow_label_pos"]
            else:
                curve_pixels = seg["curve_pixels"]
                mid_idx = len(curve_pixels) // 2
                label_x, label_y = curve_pixels[mid_idx]
    
            ax.text(
                label_x,
                label_y,
                f"{dist:.0f}nm",
                fontsize=10,
                ha="center",
                va="center",
                color=ROUTE_COLOR,
                fontweight="bold",
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
        "map_detail": cfg["map_detail"],
        "show_seamarks": cfg["show_seamarks"],
        "show_nm_distances": cfg["show_nm_distances"],
        "loaded_tiles": meta["loaded_tiles"],
        "failed_tiles": meta["failed_tiles"],
        "base_provider": meta["base_provider"],
        "closed_route": is_closed_route,
        "displayed_waypoints": len(render_waypoints),
        "input_waypoints": len(req.waypoints),
    }
