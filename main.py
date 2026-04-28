from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import uuid
import os

app = FastAPI(title="Yacht Route Map Renderer")

os.makedirs("static/maps", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


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

    margin_lat = max(0.3, (max(lats) - min(lats)) * 0.18)
    margin_lon = max(0.3, (max(lons) - min(lons)) * 0.18)

    min_lon = min(lons) - margin_lon
    max_lon = max(lons) + margin_lon
    min_lat = min(lats) - margin_lat
    max_lat = max(lats) + margin_lat

    fig = plt.figure(figsize=(12, 8), dpi=180)
    ax = plt.axes(projection=ccrs.Mercator())

    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    if req.show_route_lines:
        ax.plot(
            lons,
            lats,
            linewidth=2.2,
            marker="o",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    for p in req.waypoints:
        if p.type == "marina":
            symbol = "M"
        elif p.type == "anchorage":
            symbol = "⚓"
        elif p.type == "harbor":
            symbol = "H"
        else:
            symbol = "•"

        ax.text(
            p.lon,
            p.lat,
            symbol,
            transform=ccrs.PlateCarree(),
            fontsize=13,
            ha="center",
            va="center",
            zorder=6,
        )

        if req.show_labels:
            ax.text(
                p.lon + 0.035,
                p.lat + 0.025,
                p.name,
                transform=ccrs.PlateCarree(),
                fontsize=8,
                zorder=6,
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
                transform=ccrs.PlateCarree(),
                fontsize=8,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
                zorder=7,
            )

    plt.title(req.title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.close(fig)

    # ВАЖНО: пока оставляем локальный относительный путь.
    # После деплоя на Render заменим на настоящий домен.
    return {
        "image_url": f"/static/maps/{filename}",
        "bounds": [min_lon, min_lat, max_lon, max_lat],
        "format": "png",
    }
