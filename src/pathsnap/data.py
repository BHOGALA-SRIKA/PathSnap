"""Strict readers and structural checks for one PathSnap trip."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class TrackPoint:
    longitude: float
    latitude: float
    timestamp_s: float


@dataclass(frozen=True)
class RoadNode:
    longitude: float
    latitude: float


@dataclass(frozen=True)
class RoadArc:
    arc_id: int
    from_node_id: int
    to_node_id: int


@dataclass(frozen=True)
class TripData:
    trip_id: str
    partition: str
    track: tuple[TrackPoint, ...]
    nodes: tuple[RoadNode, ...]
    arcs: tuple[RoadArc, ...]
    route: tuple[int, ...]


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    timestamp_gaps_s: tuple[float, ...]
    nonpositive_timestamp_steps: tuple[int, ...]
    invalid_route_transitions: tuple[int, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def _read_numeric_rows(path: Path, expected_columns: int) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != expected_columns:
                raise ValueError(
                    f"{path}: line {line_number} has {len(fields)} columns; "
                    f"expected {expected_columns}."
                )
            try:
                values = [float(field) for field in fields]
            except ValueError as exc:
                raise ValueError(f"{path}: non-numeric value on line {line_number}.") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: non-finite value on line {line_number}.")
            rows.append(values)
    return rows


def read_track(path: Path) -> tuple[TrackPoint, ...]:
    rows = _read_numeric_rows(path, 3)
    points = tuple(TrackPoint(*row) for row in rows)
    for line_number, point in enumerate(points, start=1):
        if not -180 <= point.longitude <= 180 or not -90 <= point.latitude <= 90:
            raise ValueError(f"{path}: invalid longitude/latitude on data row {line_number}.")
    return points


def read_nodes(path: Path) -> tuple[RoadNode, ...]:
    rows = _read_numeric_rows(path, 2)
    nodes = tuple(RoadNode(*row) for row in rows)
    for line_number, node in enumerate(nodes, start=1):
        if not -180 <= node.longitude <= 180 or not -90 <= node.latitude <= 90:
            raise ValueError(f"{path}: invalid longitude/latitude on data row {line_number}.")
    return nodes


def _read_integer_pairs(path: Path) -> tuple[tuple[int, int], ...]:
    rows = _read_numeric_rows(path, 2)
    result: list[tuple[int, int]] = []
    for line_number, row in enumerate(rows, start=1):
        if any(not value.is_integer() for value in row):
            raise ValueError(f"{path}: expected integer indices on data row {line_number}.")
        result.append((int(row[0]), int(row[1])))
    return tuple(result)


def read_arcs(path: Path) -> tuple[RoadArc, ...]:
    return tuple(RoadArc(i, start, end) for i, (start, end) in enumerate(_read_integer_pairs(path)))


def read_route(path: Path) -> tuple[int, ...]:
    rows = _read_numeric_rows(path, 1)
    route: list[int] = []
    for line_number, row in enumerate(rows, start=1):
        if not row[0].is_integer() or row[0] < 0:
            raise ValueError(f"{path}: expected a non-negative integer arc ID on data row {line_number}.")
        route.append(int(row[0]))
    return tuple(route)


def read_trip(trip_dir: Path, partition: str | None = None) -> TripData:
    """Read files named <trip-id>.<type>, retaining trip-local arc IDs."""
    trip_id = trip_dir.name
    return TripData(
        trip_id=trip_id,
        partition=partition or trip_dir.parent.name,
        track=read_track(trip_dir / f"{trip_id}.track"),
        nodes=read_nodes(trip_dir / f"{trip_id}.nodes"),
        arcs=read_arcs(trip_dir / f"{trip_id}.arcs"),
        route=read_route(trip_dir / f"{trip_id}.route"),
    )


def validate_trip(trip: TripData, timestamp_gap_threshold_s: float = 5.0) -> ValidationReport:
    """Return structural errors and non-fatal temporal/route quality findings."""
    errors: list[str] = []
    if not trip.track:
        errors.append("track is empty")
    if not trip.nodes:
        errors.append("nodes is empty")
    if not trip.arcs:
        errors.append("arcs is empty")
    if not trip.route:
        errors.append("route is empty")
    for arc in trip.arcs:
        if not (0 <= arc.from_node_id < len(trip.nodes)):
            errors.append(f"arc {arc.arc_id} has invalid from_node_id {arc.from_node_id}")
        if not (0 <= arc.to_node_id < len(trip.nodes)):
            errors.append(f"arc {arc.arc_id} has invalid to_node_id {arc.to_node_id}")
    for position, arc_id in enumerate(trip.route):
        if not 0 <= arc_id < len(trip.arcs):
            errors.append(f"route position {position} has invalid trip-local arc_id {arc_id}")

    gaps = tuple(b.timestamp_s - a.timestamp_s for a, b in zip(trip.track, trip.track[1:]))
    nonpositive = tuple(i + 1 for i, gap in enumerate(gaps) if gap <= 0)
    breaks = tuple(
        position
        for position, (previous_id, next_id) in enumerate(zip(trip.route, trip.route[1:]))
        if 0 <= previous_id < len(trip.arcs)
        and 0 <= next_id < len(trip.arcs)
        and trip.arcs[previous_id].to_node_id != trip.arcs[next_id].from_node_id
    )
    return ValidationReport(errors, gaps, nonpositive, breaks)
