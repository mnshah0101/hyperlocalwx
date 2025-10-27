# scripts/regions.py
REGIONS = {
    "sf_coast_inland": {
        "icao": ["KSFO", "KOAK", "KSJC", "KHWD", "KLVK", "KCCR", "KSUU", "KSAC", "KMHR", "KSTS", "KAPC", "KCVH"],
        "bbox": (-123.2, 36.8, -120.5, 38.8),
        "years": [2024, 2025],
    },
    "co_front_range": {
        "icao": ["KBJC", "KBDU", "KDEN", "KAPA", "KEIK", "KLMO", "KGXY", "KFTG", "KFNL", "KLIC"],
        "bbox": (-106.0, 39.4, -103.5, 40.5),
        "years": [2024, 2025],
    },
    "wa_rainshadow": {
        "icao": ["KSEA", "KBFI", "KPAE", "KOLM", "KTIW", "KOMK", "KYKM", "KELN", "KPSC", "KMWH"],
        "bbox": (-123.6, 45.5, -118.0, 48.0),
        "years": [2024, 2025],
    },
    "lake_michigan": {
        "icao": ["KMDW", "KORD", "KGYY", "KENW", "KMKE", "KUES", "KGRR", "KMKG", "KAZO", "KBEH"],
        "bbox": (-88.5, 41.2, -85.3, 43.7),
        "years": [2024, 2025],
    },
    "phoenix_uhi": {
        "icao": ["KPHX", "KSDL", "KFFZ", "KIWA", "KDVT", "KBXK", "KGEU", "KLUF", "KPAN", "KPRC"],
        "bbox": (-113.0, 32.7, -110.5, 34.8),
        "years": [2024, 2025],
    },
    "cape_cod": {
        "icao": ["KBOS", "KBED", "KPYM", "KGHG", "KEWB", "KHYA", "KFMH", "KCQX", "KACK", "KBID", "KMVY"],
        "bbox": (-71.6, 41.0, -69.6, 42.8),
        "years": [2024, 2025],
    }
}


def get_available_regions():
    """
    Returns a list of all available region names.

    Returns:
        list: A list of region names (strings)
    """
    return list(REGIONS.keys())
