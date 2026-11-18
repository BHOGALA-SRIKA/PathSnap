from pathlib import Path

import pytest

from pathsnap.alignment import (
    AlignmentConfig,
    _State,
    _crosses_route_break,
    _decode_monotonic,
    _state_sets,
    _transition_cost,
    align_trip,
)
from pathsnap.data import (
    RoadArc,
    RoadNode,
    TrackPoint,
    TripData,
    read_arcs,
    read_nodes,
    read_route,
    read_track,
    validate_trip,
)
from pathsnap.diagnostics import write_map
from pathsnap.geometry import Candidate, TripGeometry


def sample_trip(*, far_observation: bool = False, broken_route: bool = False) -> TripData:
    nodes = (
        RoadNode(8.0, 50.0),
        RoadNode(8.001, 50.0),
        RoadNode(8.002, 50.0),
        RoadNode(8.003, 50.0),
    )
    arcs = (RoadArc(0, 0, 1), RoadArc(1, 2 if broken_route else 1, 3 if broken_route else 2))
    point = (8.0015, 50.00001) if not far_observation else (8.01, 50.01)
    return TripData(
        "trip-local",
        "train",
        (TrackPoint(8.0002, 50.0, 1), TrackPoint(*point, 7)),
        nodes,
        arcs,
        (0, 1),
    )


def test_headerless_readers_and_local_arc_ids(tmp_path: Path) -> None:
    track = tmp_path / "a.track"
    nodes = tmp_path / "a.nodes"
    arcs = tmp_path / "a.arcs"
    route = tmp_path / "a.route"
    track.write_text("8.0 50.0 1\n8.1 50.1 2\n", encoding="utf-8")
    nodes.write_text("8.0 50.0\n8.1 50.1\n", encoding="utf-8")
    arcs.write_text("0 1\n1 0\n", encoding="utf-8")
    route.write_text("1\n0\n", encoding="utf-8")
    assert len(read_track(track)) == 2
    assert len(read_nodes(nodes)) == 2
    assert [arc.arc_id for arc in read_arcs(arcs)] == [0, 1]
    assert read_route(route) == (1, 0)


def test_reader_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.track"
    path.write_text("8.0 50.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 3"):
        read_track(path)


def test_geometry_maps_arc_to_its_two_node_coordinates() -> None:
    trip = sample_trip()
    geometry = TripGeometry(trip)
    arc = geometry.edges[1]
    assert arc.arc_id == 1
    assert arc.geometry.coords[0] == geometry.nodes_xy[1]
    assert arc.geometry.coords[-1] == geometry.nodes_xy[2]
    assert arc.length_m > 0


def test_timestamp_gap_and_nonpositive_step_detection() -> None:
    trip = sample_trip()
    report = validate_trip(trip)
    assert report.timestamp_gaps_s == (6,)
    assert report.nonpositive_timestamp_steps == ()
    points = (trip.track[0], TrackPoint(trip.track[1].longitude, trip.track[1].latitude, 1))
    report = validate_trip(TripData(trip.trip_id, trip.partition, points, trip.nodes, trip.arcs, trip.route))
    assert report.nonpositive_timestamp_steps == (1,)


def test_alignment_is_monotonic_and_records_gap() -> None:
    trip = sample_trip()
    geometry = TripGeometry(trip)
    result = align_trip(trip, geometry, validate_trip(trip), AlignmentConfig(candidate_radius_m=100))
    positions = [row.route_position for row in result.observations if row.route_position is not None]
    assert positions == sorted(positions)
    assert result.observations[1].timestamp_gap_before_s == 6


def test_missing_candidates_produce_explicit_unmatched_state() -> None:
    trip = sample_trip(far_observation=True)
    geometry = TripGeometry(trip)
    result = align_trip(trip, geometry, validate_trip(trip), AlignmentConfig(candidate_radius_m=10))
    assert result.observations[1].candidate_count == 0
    assert result.observations[1].candidate_miss
    assert result.observations[1].unmatched
    assert result.observations[1].aligned_arc_id is None


def test_ambiguous_parallel_candidate_edges_are_flagged() -> None:
    trip = TripData(
        "parallel",
        "train",
        (TrackPoint(8.0005, 50.0, 1),),
        (RoadNode(8.0, 50.0), RoadNode(8.001, 50.0), RoadNode(8.0, 50.00001), RoadNode(8.001, 50.00001)),
        (RoadArc(0, 0, 1), RoadArc(1, 2, 3)),
        (0, 1),
    )
    geometry = TripGeometry(trip)
    result = align_trip(trip, geometry, validate_trip(trip), AlignmentConfig(candidate_radius_m=100))
    assert result.observations[0].candidate_count >= 2
    assert result.observations[0].locally_ambiguous


def test_invalid_route_transition_is_reported_without_repair() -> None:
    trip = sample_trip(broken_route=True)
    report = validate_trip(trip)
    assert report.invalid_route_transitions == (0,)
    assert trip.route == (0, 1)


def test_state_cap_keeps_all_occurrences_of_selected_repeated_arc() -> None:
    nodes = tuple(
        RoadNode(8.0 + arc_id * 0.001 + end * 0.0001, 50.0)
        for arc_id in range(21)
        for end in range(2)
    )
    trip = TripData(
        "repeated",
        "train",
        (TrackPoint(8.0005, 50.0, 1),),
        nodes,
        tuple(RoadArc(i, 2 * i, 2 * i + 1) for i in range(21)),
        (*range(20), 20, 20),
    )
    geometry = TripGeometry(trip)
    candidates = tuple(
        Candidate(i, float(i), 0.5, *geometry.gps_xy[0]) for i in range(20)
    ) + (Candidate(20, 100.0, 0.5, *geometry.gps_xy[0]),)
    states, candidate_counts = _state_sets(
        trip,
        geometry,
        (candidates,),
        AlignmentConfig(max_route_candidate_arcs_per_observation=20),
    )
    assert candidate_counts == [21]
    repeated_positions = [state.route_position for state in states[0] if state.arc_id == 20]
    assert repeated_positions == [20, 21]


def test_transition_never_accepts_backward_route_occurrence_or_chainage() -> None:
    config = AlignmentConfig()
    first = _State(0, 3, 10.0, 0.0, 0.0)
    backward_same_edge = _State(0, 3, 9.0, 0.0, 0.0)
    backward_route_position = _State(1, 2, 10.0, 0.0, 0.0)
    assert _transition_cost(first, backward_same_edge, 0.0, 1.0, config) == float("inf")
    assert _transition_cost(first, backward_route_position, 0.0, 1.0, config) == float("inf")


def test_long_or_invalid_timestamp_gap_weakens_distance_only_not_order() -> None:
    config = AlignmentConfig(timestamp_gap_threshold_s=5.0)
    first = _State(0, 0, 0.0, 0.0, 0.0)
    later = _State(1, 1, 1000.0, 0.0, 0.0)
    regular_cost = _transition_cost(first, later, 0.0, 1.0, config)
    assert regular_cost > 0
    assert _transition_cost(first, later, 0.0, 6.0, config) == 0.0
    assert _transition_cost(first, later, 0.0, 153064.0, config) == 0.0
    assert _transition_cost(first, later, 0.0, 0.0, config) == 0.0
    assert _transition_cost(later, first, 0.0, 153064.0, config) == float("inf")


def test_route_break_blocks_direct_transition_and_is_visible_after_unmatched_row() -> None:
    first = _State(0, 0, 0.0, 0.0, 0.0)
    after_break = _State(2, 2, 100.0, 0.0, 0.0)
    config = AlignmentConfig()
    assert _transition_cost(first, after_break, 100.0, 1.0, config, {0}) == float("inf")
    assert _crosses_route_break(0, 2, {0})

    base = sample_trip()
    trip = TripData(
        base.trip_id,
        base.partition,
        (*base.track, TrackPoint(base.track[-1].longitude, base.track[-1].latitude, 13)),
        base.nodes,
        base.arcs,
        base.route,
    )
    geometry = TripGeometry(trip)
    selected, rejected = _decode_monotonic(
        trip,
        geometry,
        [[first], [], [after_break]],
        config,
        route_breaks={0},
    )
    assert selected == [0, None, 0]
    assert not rejected


def test_decoder_rejects_backward_reentry_after_unmatched_state() -> None:
    base = sample_trip()
    trip = TripData(
        base.trip_id,
        base.partition,
        (*base.track, TrackPoint(base.track[-1].longitude, base.track[-1].latitude, 13)),
        base.nodes,
        base.arcs,
        base.route,
    )
    geometry = TripGeometry(trip)
    later = _State(1, 2, 100.0, 0.0, 0.0)
    earlier = _State(0, 0, 0.0, 0.0, 0.0)
    forward_alternative = _State(1, 3, 110.0, 5.0, 0.5)
    selected, rejected = _decode_monotonic(
        trip,
        geometry,
        [[later], [], [earlier, forward_alternative]],
        AlignmentConfig(),
    )
    assert selected == [0, None, 1]
    assert rejected == {2}


def test_alignment_marks_route_break_crossed_across_an_unmatched_observation() -> None:
    trip = TripData(
        "break-gap",
        "train",
        (
            TrackPoint(8.00005, 50.0, 1),
            TrackPoint(8.005, 50.0, 2),
            TrackPoint(8.0109, 50.0, 3),
        ),
        (
            RoadNode(8.0, 50.0),
            RoadNode(8.0001, 50.0),
            RoadNode(8.01, 50.0),
            RoadNode(8.0101, 50.0),
            RoadNode(8.0111, 50.0),
        ),
        (RoadArc(0, 0, 1), RoadArc(1, 2, 3), RoadArc(2, 3, 4)),
        (0, 1, 2),
    )
    report = validate_trip(trip)
    assert report.invalid_route_transitions == (0,)
    result = align_trip(
        trip,
        TripGeometry(trip),
        report,
        AlignmentConfig(candidate_radius_m=10),
    )
    assert result.observations[0].route_position == 0
    assert result.observations[1].unmatched
    assert result.observations[2].route_position == 2
    assert result.observations[2].crosses_invalid_route_transition


def test_folium_diagnostic_map_contains_layers(tmp_path: Path) -> None:
    trip = sample_trip()
    geometry = TripGeometry(trip)
    config = AlignmentConfig(candidate_radius_m=100)
    result = align_trip(trip, geometry, validate_trip(trip), config)
    target = tmp_path / "trip.html"
    write_map(target, trip, geometry, result, config)
    html = target.read_text(encoding="utf-8")
    assert "L.control.layers" in html
    assert "GPS observations" in html
    assert "Reference route" in html
