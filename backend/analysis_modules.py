ANALYSIS_MODULES = {
    "vehicle": {
        "title": "Vehicle Detection",
        "description": "Locate and classify road vehicles in the scene.",
    },
    "license_plate": {
        "title": "License Plate Recognition",
        "description": "Detect license plates and read visible registration text.",
    },
    "helmet": {
        "title": "Helmet Compliance",
        "description": "Assess whether detected riders are wearing helmets.",
    },
    "seatbelt": {
        "title": "Seat Belt Compliance",
        "description": "Review seat belt use in supported vehicle regions.",
    },
    "redlight": {
        "title": "Traffic Signal & Red-Light",
        "description": "Detect signal state and, with vehicles enabled, assess stop-line crossing.",
    },
}


def normalize_modules(value: str) -> list[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not requested:
        raise ValueError("Select at least one analysis module")
    if "all" in requested:
        return list(ANALYSIS_MODULES)

    unknown = sorted(set(requested) - set(ANALYSIS_MODULES))
    if unknown:
        raise ValueError(f"Unsupported analysis module(s): {', '.join(unknown)}")
    return [key for key in ANALYSIS_MODULES if key in requested]
