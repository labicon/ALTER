"""Observation contract for the retained current-frame hardware policies."""


def require_current_frame_hardware(stats):
    """Reject retired history models before loading networks or starting inference.

    Older current-frame stats may omit these fields. Keep accepting their
    defaults, including the redundant side/history metadata saved by variant A.
    Simulation's shared temporal conditioning is unaffected by this contract.
    """
    if (
        stats.get("variant") == "B"
        or tuple(stats.get("frame_offsets", (0,))) != (0,)
        or tuple(stats.get("side_frame_offsets", (0,))) != (0,)
        or tuple(stats.get("head_history_seconds", (0,))) != (0,)
    ):
        raise ValueError(
            "Hardware camera-history variant B is retired; use a current-frame "
            "hardware model or the original experiment snapshot for reproduction."
        )
