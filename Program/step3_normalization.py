"""
Step 3 — Entity Normalization
==============================
Canonicalizes O-ring engineering terminology so that equivalent terms
from different documents (e.g. "Nitrile", "NBR", "NBR70") resolve to
a single consistent label in the knowledge graph.

Extend NORMALIZATION_MAP to cover additional materials, units, or
application-specific terminology as your document set grows.
"""

import re


NORMALIZATION_MAP = {
    # Materials
    r'\bnitrile\b|\bnbr\b|\bnbr70\b':                 "NBR (Nitrile)",
    r'\bviton\b|\bfkm\b|\bfluorocarbon\b':            "FKM (Viton)",
    r'\bsilicone\b|\bvmq\b|\bpvmq\b':                 "Silicone (VMQ)",
    r'\bepdm\b':                                       "EPDM",
    r'\bneoprene\b|\bcr\b':                            "Neoprene (CR)",
    r'\bptfe\b|\bteflon\b':                            "PTFE",
    # Dimension labels
    r'\binside dia\.?\b|\binner dia\.?\b|\bid\b':      "ID",
    r'\boutside dia\.?\b|\bod\b':                      "OD",
    r'\bcross.?section\b|\bwidth\b|\bw\b|\bcs\b':     "Cross-section",
    # Pressure units
    r'\bpsi\b':                                        "PSI",
    r'\bmpa\b':                                        "MPa",
    r'\bbar\b':                                        "bar",
    # Temperature units
    r'\b°?f\b|\bfahrenheit\b':                        "°F",
    r'\b°?c\b|\bcelsius\b|\bcentigrade\b':            "°C",
}


def normalize(text):
    """Apply canonical normalization to a string."""
    t = text.lower().strip()
    for pattern, canonical in NORMALIZATION_MAP.items():
        if re.search(pattern, t, re.IGNORECASE):
            return canonical
    return text.strip()
