"""Monotonic GPS-to-reference-route alignment with explicit null states."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .data import TripData, ValidationReport
from .geometry import Candidate, TripGeometry


@dataclass(frozen=True)
class AlignmentConfig:
    candidate_radius_m: float = 80.0
    timestamp_gap_threshold_s: float = 5.0
    ambiguity_margin_m: float = 5.0
    max_route_candidate_arcs_per_observation: int = 20
    unmatched_state_cost: float = 8.0
    emission_sigma_m: float = 20.0


@dataclass(frozen=True)
class ObservationAlignment:
    observation_index: int
    timestamp_s: float
    candidate_count: int
    reference_candidate_count: int
    candidate_miss: bool
    aligned_arc_id: int | None
    route_position: int | None
    distance_to_aligned_route_m: float | None
    locally_ambiguous: bool
    unmatched: bool
    timestamp_gap_before_s: float | None
    timestamp_order_violation_before: bool
    crosses_invalid_route_transition: bool
    nonmonotonic_state_rejected: bool


@dataclass(frozen=True)
class AlignmentResult:
    observations: tuple[ObservationAlignment, ...]
    candidates_by_observation: tuple[tuple[Candidate, ...], ...]


@dataclass(frozen=True)
class _State:
    arc_id: int
    route_position: int
    chainage_m: float
    distance_m: float
    emission_cost: float


def _state_sets(
    trip: TripData,
    geometry: TripGeometry,
    all_candidates: tuple[tuple[Candidate, ...], ...],
    config: AlignmentConfig,
) -> tuple[list[list[_State]], list[int]]:
    route_positions: dict[int, list[int]] = {}
    for route_position, arc_id in enumerate(trip.route):
        route_positions.setdefault(arc_id, []).append(route_position)
    prefix = [0.0]
    for arc_id in trip.route:
        prefix.append(prefix[-1] + geometry.edges[arc_id].length_m)

    states_by_observation: list[list[_State]] = []
    ref_counts: list[int] = []
    for candidates in all_candidates:
        route_candidates = [c for c in candidates if c.arc_id in route_positions]
        ref_counts.append(len(route_candidates))
        # Bound ordinary candidate roads, not route-occurrence states. Any
        # repeated route arc among the spatial candidates is retained regardless
        # of rank so none of its valid route occurrences is discarded.
        selected_arc_ids = {
            candidate.arc_id
            for candidate in route_candidates
            if len(route_positions[candidate.arc_id]) > 1
        }
        for candidate in route_candidates:
            if len(selected_arc_ids) >= config.max_route_candidate_arcs_per_observation:
                break
            selected_arc_ids.add(candidate.arc_id)
        states: list[_State] = []
        for candidate in route_candidates:
            if candidate.arc_id not in selected_arc_ids:
                continue
            for route_position in route_positions[candidate.arc_id]:
                edge = geometry.edges[candidate.arc_id]
                chainage = prefix[route_position] + candidate.fraction_along_arc * edge.length_m
                states.append(
                    _State(
                        candidate.arc_id,
                        route_position,
                        chainage,
                        candidate.distance_m,
                        0.5 * (candidate.distance_m / config.emission_sigma_m) ** 2,
                    )
                )
        states.sort(key=lambda state: (state.emission_cost, state.route_position))
        states_by_observation.append(states)
    return states_by_observation, ref_counts


def _crosses_route_break(previous_position: int, current_position: int, route_breaks: set[int]) -> bool:
    """Whether forward route-position movement spans a broken arc boundary."""
    return any(previous_position <= boundary < current_position for boundary in route_breaks)


def _transition_cost(
    previous: _State | None,
    current: _State | None,
    gps_step_m: float,
    dt_s: float,
    config: AlignmentConfig,
    route_breaks: set[int] | None = None,
) -> float:
    """Score a transition; gaps remove displacement evidence, never route order."""
    if previous is None or current is None:
        return 1.5
    if current.route_position < previous.route_position:
        return math.inf
    if current.route_position == previous.route_position and current.chainage_m < previous.chainage_m - 1e-6:
        return math.inf
    if _crosses_route_break(previous.route_position, current.route_position, route_breaks or set()):
        return math.inf
    # A nonpositive timestamp or a long gap makes the observed displacement
    # unreliable for transition scoring. Keep order constraints, but do not
    # extrapolate the old unbounded linear uncertainty formula.
    if dt_s <= 0 or dt_s > config.timestamp_gap_threshold_s:
        return 0.0
    uncertainty = 20.0 + 5.0 * max(dt_s, 1.0)
    progress = max(current.chainage_m - previous.chainage_m, 0.0)
    return min(abs(progress - gps_step_m) / uncertainty, 12.0)


def _decode_monotonic(
    trip: TripData,
    geometry: TripGeometry,
    states: list[list[_State]],
    config: AlignmentConfig,
    route_breaks: set[int] | None = None,
) -> tuple[list[int | None], set[int]]:
    """Viterbi over route-occurrence candidates plus an explicit unmatched state."""
    costs: list[list[float]] = []
    backpointers: list[list[int | None]] = []
    for i, current_options in enumerate(states):
        options: list[_State | None] = [*current_options, None]
        row_costs: list[float] = []
        row_back: list[int | None] = []
        for current in options:
            emission = config.unmatched_state_cost if current is None else current.emission_cost
            if i == 0:
                row_costs.append(emission)
                row_back.append(None)
                continue
            best_cost = math.inf
            best_previous: int | None = None
            dt = trip.track[i].timestamp_s - trip.track[i - 1].timestamp_s
            x0, y0 = geometry.gps_xy[i - 1]
            x1, y1 = geometry.gps_xy[i]
            gps_step_m = math.hypot(x1 - x0, y1 - y0)
            previous_options: list[_State | None] = [*states[i - 1], None]
            for j, previous in enumerate(previous_options):
                transition = _transition_cost(
                    previous, current, gps_step_m, dt, config, route_breaks
                )
                if math.isinf(transition):
                    continue
                candidate_cost = costs[-1][j] + transition + emission
                if candidate_cost < best_cost:
                    best_cost, best_previous = candidate_cost, j
            row_costs.append(best_cost)
            row_back.append(best_previous)
        costs.append(row_costs)
        backpointers.append(row_back)

    selected: list[int | None] = [None] * len(states)
    if not states:
        return selected, set()
    index = min(range(len(costs[-1])), key=costs[-1].__getitem__)
    for i in range(len(states) - 1, -1, -1):
        # Each observation's final state is an explicit unmatched/null state.
        if index < len(states[i]):
            selected[i] = index
        else:
            selected[i] = None
        prior = backpointers[i][index]
        if prior is not None:
            index = prior

    # The single unmatched DP state can otherwise forget the last matched
    # route position and re-enter earlier in the route. Re-select the best
    # locally available forward state when possible; otherwise leave the
    # observation explicitly unmatched. Never publish a backward route label.
    rejected: set[int] = set()
    last_state: _State | None = None
    last_match_index: int | None = None
    for i, state_index in enumerate(selected):
        if state_index is None:
            continue
        current = states[i][state_index]
        def legal(candidate: _State) -> bool:
            if last_state is None:
                return True
            if candidate.route_position < last_state.route_position:
                return False
            if (
                candidate.route_position == last_state.route_position
                and candidate.chainage_m < last_state.chainage_m - 1e-6
            ):
                return False
            crosses_break = _crosses_route_break(
                last_state.route_position, candidate.route_position, route_breaks or set()
            )
            return not (crosses_break and last_match_index == i - 1)

        if not legal(current):
            options = [candidate for candidate in states[i] if legal(candidate)]
            if options:
                if last_state is None or last_match_index != i - 1:
                    current = min(options, key=lambda candidate: (candidate.emission_cost, candidate.route_position))
                else:
                    dt = trip.track[i].timestamp_s - trip.track[i - 1].timestamp_s
                    x0, y0 = geometry.gps_xy[i - 1]
                    x1, y1 = geometry.gps_xy[i]
                    gps_step_m = math.hypot(x1 - x0, y1 - y0)
                    current = min(
                        options,
                        key=lambda candidate: candidate.emission_cost
                        + _transition_cost(last_state, candidate, gps_step_m, dt, config, route_breaks),
                    )
                selected[i] = states[i].index(current)
            else:
                selected[i] = None
                rejected.add(i)
                continue
            rejected.add(i)
        last_state = current
        last_match_index = i
    return selected, rejected


def align_trip(
    trip: TripData,
    geometry: TripGeometry,
    validation: ValidationReport,
    config: AlignmentConfig = AlignmentConfig(),
) -> AlignmentResult:
    all_candidates = tuple(
        geometry.candidates(i, config.candidate_radius_m) for i in range(len(trip.track))
    )
    states, reference_counts = _state_sets(trip, geometry, all_candidates, config)
    route_breaks = set(validation.invalid_route_transitions)
    selected, rejected = _decode_monotonic(trip, geometry, states, config, route_breaks)
    aligned: list[ObservationAlignment] = []
    last_aligned_route_position: int | None = None
    for i, (point, candidates, local_states, state_index) in enumerate(
        zip(trip.track, all_candidates, states, selected)
    ):
        chosen = local_states[state_index] if state_index is not None else None
        distances = sorted(state.distance_m for state in local_states)
        ambiguous = len(distances) > 1 and distances[1] - distances[0] <= config.ambiguity_margin_m
        gap = None if i == 0 else point.timestamp_s - trip.track[i - 1].timestamp_s
        crosses_break = bool(
            chosen is not None
            and last_aligned_route_position is not None
            and _crosses_route_break(last_aligned_route_position, chosen.route_position, route_breaks)
        )
        aligned.append(
            ObservationAlignment(
                i,
                point.timestamp_s,
                len(candidates),
                reference_counts[i],
                reference_counts[i] == 0,
                chosen.arc_id if chosen else None,
                chosen.route_position if chosen else None,
                chosen.distance_m if chosen else None,
                ambiguous,
                chosen is None,
                gap if gap is not None and gap > config.timestamp_gap_threshold_s else None,
                gap is not None and gap <= 0,
                crosses_break,
                i in rejected,
            )
        )
        if chosen is not None:
            last_aligned_route_position = chosen.route_position
    return AlignmentResult(tuple(aligned), all_candidates)
