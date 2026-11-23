"""Read and write the simulator state of the shared environment as a small tuple.

Used only by the exhaustive verifier, to enumerate game states. Training and evaluation
never touch this module: their episodes always start from the environment's own reset.
"""

from __future__ import annotations

SCREEN_WIDTH = 288
PIPE_HEIGHT = 320
GAP_TOPS = (100, 110, 120, 130, 140, 150, 160, 170)  # the environment's eight gap heights
UPPER_YS = tuple(top - PIPE_HEIGHT for top in GAP_TOPS)
UNDECIDED = UPPER_YS[0]  # placeholder stored for a pipe that is still off-screen
# (bird y, bird velocity, bird rotation, then x and upper-pipe y of the three pipes by x)
State = tuple


def read_state(raw) -> State:
    """Canonical state of an unwrapped environment; off-screen gaps are left undecided."""
    pipes = sorted(
        (p["x"], p["y"] if p["x"] <= SCREEN_WIDTH else UNDECIDED) for p in raw._upper_pipes
    )
    return (
        raw._player_y,
        raw._player_vel_y,
        raw._player_rot,
        pipes[0][0],
        pipes[0][1],
        pipes[1][0],
        pipes[1][1],
        pipes[2][0],
        pipes[2][1],
    )


def write_state(raw, state: State, *, reveal: tuple[int, int] | None = None) -> None:
    """Load ``state`` into an unwrapped environment.

    ``reveal=(slot, upper_y)`` decides the gap of one pipe that is still off-screen.
    """
    raw._player_y, raw._player_vel_y, raw._player_rot = state[0], state[1], state[2]
    gap = raw._pipe_gap
    for i in range(3):
        x, upper_y = state[3 + 2 * i], state[4 + 2 * i]
        if reveal is not None and reveal[0] == i:
            upper_y = reveal[1]
        raw._upper_pipes[i]["x"] = x
        raw._upper_pipes[i]["y"] = upper_y
        raw._lower_pipes[i]["x"] = x
        raw._lower_pipes[i]["y"] = upper_y + PIPE_HEIGHT + gap


def entering_pipe(state: State) -> int | None:
    """Slot of the pipe that comes into view on the next step, if any."""
    for i in range(3):
        x = state[3 + 2 * i]
        if x > SCREEN_WIDTH and x + -4 <= SCREEN_WIDTH:
            return i
    return None
