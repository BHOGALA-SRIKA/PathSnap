"""Trip-local metric road geometry and spatial candidate queries."""
from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .data import TripData


@dataclass(frozen=True)
class RoadGeometry:
    arc_id: int
    geometry: LineString
    length_m: float


@dataclass(frozen=True)
class Candidate:
    arc_id: int
    distance_m: float
    fraction_along_arc: float
    projected_x_m: float
    projected_y_m: float


class TripGeometry:
    """Coordinates are projected to a trip-centered azimuthal-equidistant CRS."""

    def __init__(self, trip: TripData):
        if not trip.track:
            raise ValueError("Cannot project a trip with no GPS observations.")
        lon0 = sum(p.longitude for p in trip.track) / len(trip.track)
        lat0 = sum(p.latitude for p in trip.track) / len(trip.track)
        local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0:.10f} +lon_0={lon0:.10f} "
            "+datum=WGS84 +units=m +no_defs"
        )
        self.to_metric = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
        self.to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
        self.gps_xy = tuple(self.to_metric.transform(p.longitude, p.latitude) for p in trip.track)
        self.nodes_xy = tuple(self.to_metric.transform(n.longitude, n.latitude) for n in trip.nodes)
        self.edges = tuple(
            RoadGeometry(
                arc.arc_id,
                LineString((self.nodes_xy[arc.from_node_id], self.nodes_xy[arc.to_node_id])),
                LineString((self.nodes_xy[arc.from_node_id], self.nodes_xy[arc.to_node_id])).length,
            )
            for arc in trip.arcs
        )
        self._tree = STRtree([edge.geometry for edge in self.edges])

    def candidates(self, observation_index: int, radius_m: float = 80.0) -> tuple[Candidate, ...]:
        """Return every road edge within radius; IDs remain local to this trip."""
        x, y = self.gps_xy[observation_index]
        point = Point(x, y)
        nearby_indices = self._tree.query(point.buffer(radius_m))
        result: list[Candidate] = []
        for index in nearby_indices:
            edge = self.edges[int(index)]
            distance = point.distance(edge.geometry)
            if distance <= radius_m:
                along = edge.geometry.project(point)
                length = edge.length_m
                snapped = edge.geometry.interpolate(along)
                result.append(
                    Candidate(
                        edge.arc_id,
                        float(distance),
                        float(along / length) if length > 0 else 0.0,
                        float(snapped.x),
                        float(snapped.y),
                    )
                )
        return tuple(sorted(result, key=lambda c: (c.distance_m, c.arc_id)))

    def to_latlon(self, x_m: float, y_m: float) -> tuple[float, float]:
        longitude, latitude = self.to_wgs84.transform(x_m, y_m)
        return latitude, longitude
