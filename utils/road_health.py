
def assess_road_health(small, medium, large):
    """
    Estimate road health using pothole severity counts.

    Small   = 1 risk point
    Medium  = 3 risk points
    Large   = 6 risk points
    """

    small = max(0, int(small))
    medium = max(0, int(medium))
    large = max(0, int(large))

    total = small + medium + large

    # Empty detection result
    if total == 0:
        return {
            "health": "GOOD",
            "health_color": "#22C55E",
            "risk_score": 0,
            "repair": "No Major Repair",
            "repair_color": "#22C55E",
            "repair_message": (
                "No potholes were detected. "
                "The inspected image or video appears "
                "to have good road condition."
            )
        }

    # Severity-based risk calculation
    risk_score = (
        (small * 1) +
        (medium * 3) +
        (large * 6)
    )

    # Road health classification
    if large >= 3 or risk_score >= 15:

        health = "BAD"
        health_color = "#EF4444"
        repair = "Full Repair Assessment"
        repair_color = "#EF4444"

        repair_message = (
            "Significant road damage detected. "
            "A detailed engineering inspection and "
            "possible full-road repair assessment "
            "are recommended."
        )

    elif risk_score >= 5 or medium >= 2 or large >= 1:

        health = "MODERATE"
        health_color = "#F59E0B"
        repair = "Patch Repair Recommended"
        repair_color = "#F59E0B"

        repair_message = (
            "Moderate road damage detected. "
            "Localized pothole patching and "
            "maintenance are recommended."
        )

    else:

        health = "GOOD"
        health_color = "#22C55E"
        repair = "Minor Maintenance"
        repair_color = "#22C55E"

        repair_message = (
            "Road condition appears relatively good. "
            "Only minor maintenance may be required."
        )

    return {
        "health": health,
        "health_color": health_color,
        "risk_score": risk_score,
        "repair": repair,
        "repair_color": repair_color,
        "repair_message": repair_message
    }
