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

app = FastAPI(title="Yacht Route Map Renderer")

os.makedirs("static/maps", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

TILE_SIZE = 256
OSM_TILE_URL = "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
USER_AGENT = "yacht-route-renderer/0.1"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


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

    # x по долготе можно зациклить
    x = x % max_index

    # y по широте не должен выходить за диапазон
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
    return cropped, zoom


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

    bg_img, zoom = build_osm_background(min_lon, min_lat, max_lon, max_lat)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=120)

    # Реальная OSM-подложка
    ax.imshow(
        bg_img,
        extent=[min_lon, max_lon, min_lat, max_lat],
        aspect="auto",
        zorder=0
    )

    if req.show_route_lines:
        ax.plot(lons, lats, linewidth=2.2, marker="o", zorder=5)

    for p in req.waypoints:
        if p.type == "marina":
            symbol = "M"
        elif p.type == "anchorage":
            symbol = "A"
        elif p.type == "harbor":
            symbol = "H"
        else:
            symbol = "•"

        ax.text(
            p.lon,
            p.lat,
            symbol,
            fontsize=12,
            ha="center",
            va="center",
            zorder=6,
            bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec="black", alpha=0.95),
        )

        if req.show_labels:
            ax.text(
                p.lon + 0.03,
                p.lat + 0.02,
                p.name,
                fontsize=8,
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.80),
            )

    if req.show_nm_distances:
        for a, b in zip(req.waypoints[:-1], req.waypoints[1:]):
            mid_lat = (a.lat + b.lat) / 2
            mid_lon = (a.lon + b.lon) / 2
            dist = nm_distance(a, b)

            ax.text(
                mid_lon,
                mid_lat,
                f"{dist:.0f} NM",
                fontsize=8,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
                zorder=7,
            )

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{req.title} (zoom {zoom})", fontsize=14)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.close(fig)

    base_url = "https://yacht-route-renderer.onrender.com"

    return {
        "image_url": f"{base_url}/static/maps/{filename}",
        "bounds": [min_lon, min_lat, max_lon, max_lat],
        "format": "png",
        "zoom": zoom,
    }
