"""
Step 1 — PDF Extraction
=======================
Extracts every table from a digital PDF along with surrounding context,
plus a per-page prose description used to populate Page nodes in the KG.

Changes from user's version:
  - Integrates step1d block classifier to label each page's text blocks
  - Adds `_pages` to the result: {page_num: {"prose": str, "blocks": [...]}}
  - Keeps all user's TOC detection, dual-strategy, and metadata logic
"""

import re
import pdfplumber

from App.console_progress import status_line, status_line_done
from Program.step1d_block_classifier import classify_page_blocks


# ── ToC detection ────────────────────────────────────────────────────────────

_TOC_ENTRY_RE   = re.compile(r'^(.+?)[\s.·•\-]{2,}\s*(\d+)\s*$')
_TOC_HEADING_RE = re.compile(r'^\s*(table\s+of\s+)?contents?\s*$', re.IGNORECASE)


def detect_toc(pdf_path, max_pages=20):
    """
    Scan the first max_pages pages for a Table of Contents.
    Returns {"toc_page_numbers": set, "entries": [{"title":..,"page":..}]}
    """
    toc_page_numbers = set()
    entries = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_scan = min(max_pages, len(pdf.pages))
            for page_num in range(pages_to_scan):
                page = pdf.pages[page_num]
                text = page.extract_text() or ""
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                if not lines:
                    continue
                top_lines = lines[:max(3, len(lines) // 4)]
                if not any(_TOC_HEADING_RE.match(l) for l in top_lines):
                    continue
                toc_page_numbers.add(page_num + 1)
                for line in lines:
                    m = _TOC_ENTRY_RE.match(line)
                    if m:
                        title    = m.group(1).strip(" .")
                        page_ref = int(m.group(2))
                        if title and page_ref > 0:
                            entries.append({"title": title, "page": page_ref})
    except Exception:
        pass
    return {"toc_page_numbers": toc_page_numbers, "entries": entries}


# ── Multi-column detection ────────────────────────────────────────────────────

def _detect_columns(page) -> int:
    """Detect whether a page uses a multi-column layout (returns 1 or 2)."""
    try:
        words = page.extract_words()
    except Exception:
        return 1
    if not words or len(words) < 20:
        return 1
    mid_x = page.width / 2
    margin = page.width * 0.08
    center_words = [
        w for w in words
        if (mid_x - margin <= w["x0"] <= mid_x + margin
            or mid_x - margin <= w["x1"] <= mid_x + margin)
    ]
    center_ratio = len(center_words) / len(words)
    if center_ratio < 0.05:
        left_words  = [w for w in words if w["x1"] < mid_x]
        right_words = [w for w in words if w["x0"] > mid_x]
        left_ratio  = len(left_words) / len(words)
        right_ratio = len(right_words) / len(words)
        if left_ratio > 0.25 and right_ratio > 0.25:
            return 2
    return 1


def _extract_column_text(page, num_columns: int) -> str:
    """Extract text respecting column layout to avoid interleaving."""
    if num_columns <= 1:
        return page.extract_text() or ""
    mid_x = page.width / 2
    left  = page.crop((0, 0, mid_x, page.height)).extract_text() or ""
    right = page.crop((mid_x, 0, page.width, page.height)).extract_text() or ""
    return left.rstrip() + "\n\n" + right.lstrip()


# ── Table extraction ─────────────────────────────────────────────────────────

def extract_tables_with_context(pdf_path, margin=50, heading_size_threshold=11, use_generative=False):
    """
    Extract every table with surrounding context, plus per-page prose blocks.

    Returns a list where:
      - Most entries are table dicts (page, bbox, section_heading, …)
      - results[0]["_toc"]   = TOC metadata
      - results[0]["_pages"] = {page_num: {"prose": str, "blocks": [...]}}
    """
    toc = detect_toc(pdf_path)
    results = []
    pages_data: dict[int, dict] = {}   # page_num → prose + blocks

    try:
        pdf_handle = pdfplumber.open(pdf_path)
    except OSError as e:
        raise OSError(
            f"Could not open PDF file.\n"
            f"  Path passed to pdfplumber: {pdf_path!r}\n"
            f"  Underlying error: {e}"
        ) from e

    with pdf_handle as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            pct = round((page_num + 1) / total_pages * 100)
            status_line(f"  Extracting... {pct}% (page {page_num + 1}/{total_pages})")

            # Skip ToC pages
            if (page_num + 1) in toc["toc_page_numbers"]:
                continue

            # ── Detect multi-column layout ────────────────────────────────
            num_columns = _detect_columns(page)

            # ── Classify all text blocks on this page ──────────────────────
            blocks = classify_page_blocks(page, use_generative=use_generative)
            if num_columns > 1:
                prose_text = _extract_column_text(page, num_columns)
            else:
                prose_parts = [
                    b["text"] for b in blocks
                    if b["block_type"] in ("description", "summary")
                ]
                prose_text = " ".join(prose_parts).strip()
            pages_data[page_num + 1] = {
                "prose":  prose_text,
                "blocks": blocks,
            }

            # ── Extract tables ─────────────────────────────────────────────
            tables = _find_tables_dual_strategy(page)
            for table in tables:
                bbox = table.bbox

                above_top = max(0, bbox[1] - margin)
                above_bot = max(above_top, bbox[1])
                if above_bot - above_top > 1:
                    above_text = (page.crop((0, above_top, page.width, above_bot))
                                  .extract_text() or "").strip()
                else:
                    above_text = ""

                below_top = max(0, bbox[3])
                below_bot = min(page.height, bbox[3] + margin)
                if below_bot - below_top > 1:
                    below_text = (page.crop((0, below_top, page.width, below_bot))
                                  .extract_text() or "").strip()
                else:
                    below_text = ""

                heading = _find_section_heading(page, bbox, heading_size_threshold)

                if bbox[1] < 100 and page_num > 0:
                    prev_page = pdf.pages[page_num - 1]
                    heading = heading or _find_section_heading(
                        prev_page,
                        (0, 0, prev_page.width, prev_page.height),
                        heading_size_threshold,
                    )

                if not heading:
                    heading = _toc_section_for_page(toc, page_num + 1)

                raw = table.extract()
                if not raw or len(raw) < 2:
                    continue
                if _looks_like_toc_table(raw):
                    continue

                results.append({
                    "page":            page_num + 1,
                    "bbox":            bbox,
                    "section_heading": heading,
                    "caption_above":   above_text,
                    "table":           raw,
                    "notes_below":     below_text,
                    "footnotes":       _parse_footnotes(below_text),
                })

    status_line_done(f"  Extracting... 100% ({total_pages}/{total_pages} pages) — done.")

    # Attach metadata to first result (or as standalone if no tables found)
    meta = {"_toc": toc, "_pages": pages_data}
    if results:
        results[0].update(meta)
    else:
        results.append(meta)

    return results


# ── Internal helpers ─────────────────────────────────────────────────────────

def _find_tables_dual_strategy(page):
    """Try bordered (lines) strategy first, fall back to text-alignment."""
    tables = page.find_tables({
        "vertical_strategy":   "lines",
        "horizontal_strategy": "lines",
    })
    if tables:
        return tables
    return page.find_tables({
        "vertical_strategy":    "text",
        "horizontal_strategy":  "text",
        "min_words_vertical":   2,
        "min_words_horizontal": 2,
    })


def _looks_like_toc_table(raw_table):
    """True if the last column is mostly page numbers (ToC artifact)."""
    if not raw_table or len(raw_table) < 3:
        return False
    last_col  = [row[-1] for row in raw_table if row and row[-1]]
    numeric   = sum(1 for v in last_col if re.match(r'^\d+$', (v or "").strip()))
    return numeric >= len(last_col) * 0.6


def _toc_section_for_page(toc, page_num):
    """Return the ToC section title whose page number ≤ page_num."""
    best = None
    for entry in toc.get("entries", []):
        if entry["page"] <= page_num:
            best = entry["title"]
        else:
            break
    return best


def _find_section_heading(page, table_bbox, size_threshold):
    """Return the nearest heading-level text above the table bbox."""
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
    except Exception:
        return None

    above = [w for w in words if w["bottom"] < table_bbox[1]]
    above_sorted = sorted(above, key=lambda w: w["bottom"], reverse=True)

    for word in above_sorted:
        if word.get("size", 0) >= size_threshold:
            same_line = [
                w["text"] for w in above_sorted
                if abs(w["top"] - word["top"]) < 3
            ]
            return " ".join(same_line)
    return None


def _parse_footnotes(text):
    """Parse footnote markers and their descriptions from below-table text."""
    pattern = r'([*†‡§]|\d+[).]|[a-z][).])[ \t]+(.+?)(?=\n[*†‡§\d]|\n[a-z][).]|$)'
    return {
        m.group(1).strip(): m.group(2).strip()
        for m in re.finditer(pattern, text, re.DOTALL)
    }
