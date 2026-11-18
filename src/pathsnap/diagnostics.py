"""Training-only Stage 1 diagnostics and Folium maps."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import folium

from .alignment import AlignmentConfig, AlignmentResult, align_trip
from .data import TripData, validate_trip
from .geometry import TripGeometry


def summarize_trip(trip: TripData, geometry: TripGeometry, result: AlignmentResult, config: AlignmentConfig) -> dict:
    report = validate_trip(trip, config.timestamp_gap_threshold_s)
    obs = result.observations
    gaps = report.timestamp_gaps_s
    projected_distances = [row.distance_to_aligned_route_m for row in obs if row.distance_to_aligned_route_m is not None]
    transitions = max(0, len(trip.route) - 1)
    valid_transitions = transitions - len(report.invalid_route_transitions)
    return {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "gps_observations": len(trip.track),
        "nodes": len(trip.nodes),
        "arcs": len(trip.arcs),
        "reference_route_length": len(trip.route),
        "candidate_coverage_pct": round(100 * sum(row.reference_candidate_count > 0 for row in obs) / max(len(obs), 1), 4),
        "unmatched_observations": sum(row.unmatched for row in obs),
        "alignment_ambiguities": sum(row.locally_ambiguous for row in obs),
        "mean_alignment_distance_m": round(sum(projected_distances) / len(projected_distances), 4) if projected_distances else None,
        "median_alignment_distance_m": round(median(projected_distances), 4) if projected_distances else None,
        "max_alignment_distance_m": round(max(projected_distances), 4) if projected_distances else None,
        "timestamp_gap_count": sum(gap > config.timestamp_gap_threshold_s for gap in gaps),
        "timestamp_gap_threshold_s": config.timestamp_gap_threshold_s,
        "max_timestamp_gap_s": max(gaps, default=0.0),
        "nonpositive_timestamp_steps": len(report.nonpositive_timestamp_steps),
        "invalid_route_transitions": len(report.invalid_route_transitions),
        "valid_route_transitions": valid_transitions,
        "observations_crossing_invalid_route_transition": sum(row.crosses_invalid_route_transition for row in obs),
        "nonmonotonic_state_rejections": sum(row.nonmonotonic_state_rejected for row in obs),
        "validation_errors": list(report.errors),
    }


def write_observation_csv(path: Path, trip: TripData, geometry: TripGeometry, result: AlignmentResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(result.observations[0].__dataclass_fields__) if result.observations else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["longitude", "latitude", *fields])
        writer.writeheader()
        for point, (x, y), row in zip(trip.track, geometry.gps_xy, result.observations):
            writer.writerow({"longitude": point.longitude, "latitude": point.latitude, **row.__dict__})


def write_map(path: Path, trip: TripData, geometry: TripGeometry, result: AlignmentResult, config: AlignmentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    center = [sum(p.latitude for p in trip.track) / len(trip.track), sum(p.longitude for p in trip.track) / len(trip.track)]
    fmap = folium.Map(location=center, zoom_start=13, control_scale=True, tiles="OpenStreetMap")
    roads = folium.FeatureGroup(name="Road network", show=True)
    relevant_arc_ids = {candidate.arc_id for batch in result.candidates_by_observation for candidate in batch}
    relevant_arc_ids.update(trip.route)
    for arc_id in relevant_arc_ids:
        edge = geometry.edges[arc_id]
        coords = [geometry.to_latlon(*xy) for xy in edge.geometry.coords]
        is_route = False
        folium.PolyLine(coords, color="#9aa3ad", weight=2, opacity=0.45, tooltip=f"Candidate road arc {arc_id}").add_to(roads)
    roads.add_to(fmap)
    reference = folium.FeatureGroup(name="Reference route (ordered arcs)", show=True)
    breaks = set(validate_trip(trip, config.timestamp_gap_threshold_s).invalid_route_transitions)
    for position, arc_id in enumerate(trip.route):
        edge = geometry.edges[arc_id]
        coords = [geometry.to_latlon(*xy) for xy in edge.geometry.coords]
        color = "#e4572e" if position and position - 1 in breaks else "#198754"
        folium.PolyLine(coords, color=color, weight=4, opacity=0.85, tooltip=f"Route position {position}; trip-local arc {arc_id}").add_to(reference)
    reference.add_to(fmap)
    gps_group = folium.FeatureGroup(name="GPS observations", show=True)
    stride = max(1, len(trip.track) // 500)
    for i in range(0, len(trip.track), stride):
        point = trip.track[i]
        folium.CircleMarker((point.latitude, point.longitude), radius=2, color="#2463eb", fill=True, fill_opacity=0.7, tooltip=f"GPS observation {i}").add_to(gps_group)
    gps_group.add_to(fmap)
    aligned_group = folium.FeatureGroup(name="Monotonic alignment", show=True)
    for row in result.observations[::stride]:
        if row.distance_to_aligned_route_m is None:
            continue
        point = trip.track[row.observation_index]
        color = "#f0ad00" if row.locally_ambiguous else "#7c3aed"
        folium.CircleMarker((point.latitude, point.longitude), radius=3, color=color, fill=True, fill_opacity=0.8, tooltip=f"Aligned arc {row.aligned_arc_id}; distance {row.distance_to_aligned_route_m:.1f} m; route position {row.route_position}").add_to(aligned_group)
    aligned_group.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(p.latitude for p in trip.track), min(p.longitude for p in trip.track)], [max(p.latitude for p in trip.track), max(p.longitude for p in trip.track)]])
    fmap.save(str(path))


def run_training_diagnostics(
    data_root: Path,
    output_root: Path,
    config: AlignmentConfig = AlignmentConfig(),
    map_count: int = 3,
) -> list[dict]:
    """Process train only; student_test and absent evaluator data are never read."""
    train_dir = data_root / "train"
    trips = sorted(path for path in train_dir.iterdir() if path.is_dir())
    if len(trips) != 80:
        raise ValueError(f"Expected 80 training trips; found {len(trips)} in {train_dir}.")
    student_dir = data_root / "student_test"
    student_trip_ids = sorted(path.name for path in student_dir.iterdir() if path.is_dir()) if student_dir.exists() else []
    if len(student_trip_ids) != 10:
        raise ValueError(
            "Expected 10 student_test trip folders in the supplied dataset; "
            f"found {len(student_trip_ids)}. The evaluator partition is never substituted."
        )
    train_ids = {path.name for path in trips}
    overlap = sorted(train_ids.intersection(student_trip_ids))
    if overlap:
        raise ValueError(f"Trip IDs occur in both train and student_test: {overlap}")
    out = output_root
    observations_dir = out / "observations"
    summaries: list[dict] = []
    cached: list[tuple[TripData, TripGeometry, AlignmentResult, float]] = []
    for i, trip_dir in enumerate(trips, start=1):
        from .data import read_trip
        trip = read_trip(trip_dir, partition="train")
        validation = validate_trip(trip, config.timestamp_gap_threshold_s)
        if validation.errors:
            raise ValueError(f"Invalid training trip {trip.trip_id}: {validation.errors}")
        geometry = TripGeometry(trip)
        result = align_trip(trip, geometry, validation, config)
        summaries.append(summarize_trip(trip, geometry, result, config))
        write_observation_csv(observations_dir / f"{trip.trip_id}.csv", trip, geometry, result)
        cached.append((trip, geometry, result, summaries[-1]["candidate_coverage_pct"]))
        # Retain at most the lowest-coverage examples plus one well-covered
        # representative, so maps highlight real failure cases without holding
        # every trip's road graph in memory.
        low_coverage = sorted(
            (item for item in cached if item[3] < 99.0),
            key=lambda item: (item[3], item[0].trip_id),
        )[: max(0, map_count - 1)]
        well_covered = [item for item in cached if item[3] >= 99.0]
        representative = max(well_covered, key=lambda item: (len(item[0].track), item[0].trip_id), default=None)
        cached = low_coverage + ([representative] if representative is not None else [])
        cached = cached[:max(0, map_count)]
        print(f"[{i}/80] {trip.trip_id}: {len(trip.track)} observations; {summaries[-1]['candidate_coverage_pct']:.1f}% candidate coverage")

    out.mkdir(parents=True, exist_ok=True)
    partition_manifest = {
        "training_trip_count": len(trips),
        "student_test_trip_count": len(student_trip_ids),
        "training_student_test_id_overlap": overlap,
        "reserved_evaluator_partition_present": (data_root / "evaluator_test").exists(),
        "reserved_evaluator_partition_processed": False,
    }
    (out / "partition_manifest.json").write_text(json.dumps(partition_manifest, indent=2), encoding="utf-8")
    (out / "trip_diagnostics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (out / "trip_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [key for key in summaries[0] if key != "validation_errors"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row[key] for key in fields})

    map_dir = out / "maps"
    selected = cached[:map_count]
    map_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = map_dir / "index.json"
    old_map_ids: list[str] = []
    if manifest_path.exists():
        try:
            old_map_ids = json.loads(manifest_path.read_text(encoding="utf-8")).get("trip_ids", [])
        except (json.JSONDecodeError, AttributeError):
            old_map_ids = []
    selected_ids = [trip.trip_id for trip, _geometry, _result, _coverage in selected]
    for old_trip_id in old_map_ids:
        if old_trip_id not in selected_ids:
            (map_dir / f"{old_trip_id}.html").unlink(missing_ok=True)
    for trip, geometry, result, _coverage in selected:
        write_map(map_dir / f"{trip.trip_id}.html", trip, geometry, result, config)
    manifest_path.write_text(json.dumps({"trip_ids": selected_ids}, indent=2), encoding="utf-8")
    return summaries
