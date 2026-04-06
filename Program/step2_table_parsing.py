"""
Step 2 — Table Parsing (O-Ring Specific)
=========================================
Converts raw pdfplumber table output (list of lists) into a matrix
structure that preserves column-level metadata (header name, unit)
and the two-dimensional row/column relationships.

Improvements over the original:
  - Multi-row header support: if the second row looks like a sub-header
    (non-numeric, non-dash values), it is merged with the first row.
  - Vertical span filling: None cells in data rows inherit the value
    from the cell above them (common in merged data cells).
  - Repeating-header detection: rows that duplicate the header row are
    removed (pdfplumber page-break artifacts are caught in step1 too,
    but this catches any that slip through).
  - Improved unit parsing with more engineering unit tokens.

Output format
-------------
{
    "columns": [
        {"index": 0, "header": "Dash No.", "unit": None},
        {"index": 1, "header": "ID",       "unit": "in"},
        ...
    ],
    "rows": [
        {
            "index": 0,
            "cells": [
                {"col_index": 0, "header": "Dash No.", "unit": None,
                 "value": "-001", "raw": "-001*"},
                ...
            ]
        },
        ...
    ]
}
"""

import re


_UNIT_TOKENS = {
    # Length
    "in", "inch", "inches", "mm", "cm", "m", "ft",
    # Pressure
    "psi", "kpa", "mpa", "bar", "atm", "ksi",
    # Temperature
    "°f", "°c", "f", "c", "k",
    # Mass / force
    "lbs", "lb", "kg", "g", "n", "kn",
    # Area / volume
    "in²", "mm²", "cm²", "in³", "mm³", "cm³",
    # Misc
    "%", "rpm", "hz", "mhz",
}

_UNIT_PATTERNS = [
    re.compile(r'^(.+?)\s*\(([^)]+)\)\s*$'),      # "ID (in)"
    re.compile(r'^(.+?)\s*\[([^\]]+)\]\s*$'),      # "Pressure [MPa]"
    re.compile(r'^(.+?)\s*/\s*([^\s]+)\s*$'),      # "Cross-section/mm"
    re.compile(r'^(.+?)\s*[—–]\s*([^\s]+)\s*$'),   # "Width—mm"  (em/en dash only)
    re.compile(r'^(.+?)\s+(°[CF])\s*$'),           # "Temp. °C"
]

# A cell looks like a data value (not a header) if it matches these
_DATA_VALUE_RE = re.compile(
    r'^-?\d|^[A-Z]{2,5}\d|^[-+]|'   # numbers, part codes, signed values
    r'^\d{1,3}[,/]\d|'              # fractions, ranges like "3/4"
    r'^(NBR|FKM|EPDM|VMQ|PTFE|CR)\b', re.IGNORECASE
)


def parse_oring_table(raw_table):
    """
    Parse a raw pdfplumber table into a matrix structure with full
    column metadata (header + unit) and per-cell row/column context.
    Returns an empty dict if the table has fewer than 2 rows.
    """
    if not raw_table or len(raw_table) < 2:
        return {}

    raw_table = [list(row) for row in raw_table]  # make mutable

    # --- Detect multi-row headers ---
    # If row[1] looks like a sub-header (no data values), merge it with row[0]
    header_rows, data_rows = _split_header_and_data(raw_table)

    # Build merged header from all header rows
    merged_headers = _merge_header_rows(header_rows)
    raw_headers = _forward_fill(merged_headers)

    columns = []
    for i, raw_h in enumerate(raw_headers):
        cleaned = _clean_cell(raw_h)
        header, unit = _parse_header_unit(cleaned)
        columns.append({"index": i, "header": header, "unit": unit})

    header_sig = [str(c or "").strip() for c in raw_headers]

    # --- Build rows with vertical-span filling ---
    prev_values = [""] * len(columns)
    rows = []

    for row_idx, raw_row in enumerate(data_rows):
        if not any(raw_row):
            continue

        # Skip rows that duplicate the header (repeating header artifact)
        row_sig = [str(c or "").strip() for c in raw_row]
        if row_sig == header_sig:
            continue

        cells = []
        for col_idx, raw_val in enumerate(raw_row):
            if col_idx >= len(columns):
                break
            col = columns[col_idx]

            # Vertical span: empty cell inherits value from above
            raw_str = (raw_val or "").strip()
            if raw_str == "" and prev_values[col_idx]:
                raw_str = prev_values[col_idx]
                raw_val = raw_str

            cleaned_val = _clean_cell(raw_val)
            prev_values[col_idx] = cleaned_val

            cells.append({
                "col_index": col_idx,
                "header":    col["header"],
                "unit":      col["unit"],
                "value":     cleaned_val,
                "raw":       raw_str,
            })

        if any(c["value"] for c in cells):
            rows.append({"index": row_idx, "cells": cells})

    return {"columns": columns, "rows": rows}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_header_and_data(raw_table):
    """
    Separate header rows from data rows.
    A row is considered a header if none of its non-empty cells look
    like data values (numbers, part codes, material abbreviations).
    At most the first 3 rows are checked.
    """
    header_rows = [raw_table[0]]
    data_start = 1

    for i in range(1, min(3, len(raw_table))):
        row = raw_table[i]
        non_empty = [str(c or "").strip() for c in row if (c or "").strip()]
        if not non_empty:
            continue
        looks_like_data = any(_DATA_VALUE_RE.match(v) for v in non_empty)
        if not looks_like_data:
            header_rows.append(row)
            data_start = i + 1
        else:
            break

    return header_rows, raw_table[data_start:]


def _merge_header_rows(header_rows):
    """
    Merge multiple header rows into one by concatenating non-empty
    cells from each row. The first row takes precedence for cells
    that are non-empty in both rows.
    """
    if len(header_rows) == 1:
        return header_rows[0]

    max_cols = max(len(r) for r in header_rows)
    merged = []
    for col_idx in range(max_cols):
        parts = []
        for row in header_rows:
            val = (row[col_idx] if col_idx < len(row) else None) or ""
            val = str(val).strip()
            if val and val not in parts:
                parts.append(val)
        merged.append(" ".join(parts) if parts else "")
    return merged


def _parse_header_unit(header_text):
    """Split a header string into (name, unit)."""
    if not header_text:
        return ("", None)

    for pattern in _UNIT_PATTERNS:
        m = pattern.match(header_text)
        if m:
            return (m.group(1).strip(), m.group(2).strip())

    parts = header_text.rsplit(None, 1)
    if len(parts) == 2 and parts[1].lower() in _UNIT_TOKENS:
        return (parts[0].strip(), parts[1].strip())

    return (header_text, None)


def _forward_fill(row):
    """Replace None/empty cells with the last seen non-empty value."""
    filled, last = [], ""
    for cell in row:
        val = (cell or "").strip()
        if val:
            last = val
        filled.append(last)
    return filled


def _clean_cell(value):
    """Strip whitespace, newlines, and trailing footnote markers."""
    if value is None:
        return ""
    value = str(value).replace("\n", " ").strip()
    value = re.sub(r'[*†‡§]+$', '', value).strip()
    return value
