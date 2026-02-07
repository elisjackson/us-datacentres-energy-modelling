"""
Optimisation logic triggered by the Optimise button.
"""
import time
import random


def run_optimisation(**kwargs):
    """
    Placeholder: simulates an optimisation run (1–2 seconds), then returns a result.
    Replace with real logic; kwargs can receive form state (toggles, sliders, tiers).
    """
    duration = random.uniform(1.0, 2.0)
    time.sleep(duration)
    return {
        "status": "ok",
        "message": f"Optimisation complete (simulated {duration:.1f}s).",
        "duration_seconds": round(duration, 2),
    }
