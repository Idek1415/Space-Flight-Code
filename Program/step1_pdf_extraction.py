"""
Step 1 — PDF Extraction
=======================
Extracts structured content from a PDF, preserving reading order and
logical document structure.  Pure extraction — no ML models.

Features
--------
- Multi-column detection via horizontal gutter analysis
- Header detection using font metadata (size, bold weight, uppercase)
- Cross-page section continuity: a section that starts on page N and
  continues to page N+1 carries the same heading across both pages so
  step 4 can stitch it into a single Section node.
- Sentence-level prose segmentation (common abbreviations protected)
- Table extraction via pdfplumber find_tables() with caption capture
- Reference / bibliography section detection and exclusion

Return format
-------------
{
    "title": str,
    "pages": {
        page_num (int): {
            "sections": [
                {
                    "heading":   str,   # empty for body text before first heading
                    "level":     int,   # 1, 2, or 3
                    "sentences": [str],
                    "page":      int,
                }
            ],
            "tables": [
                {
                    "raw":     list[list],   # pdfplumber raw table output
                    "caption": str,          # text immediately above the table
                    "bbox":    tuple,        # (x0, y0, x1, y1) in points
                    "page":    int,
                }
            ]
        }
    }
}
"""

import re
from collections import Counter

import pdfplumber


# ── Text utilities ────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Normalize whitespace and resolve PDF hyphenated line-breaks.
    Example: "meth-  od" -> "method"
    """
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)   # de-hyphenate across lines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


_ABBREV_RE = re.compile(
    r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g|'
    r'Fig|Tab|Eq|Sec|Ref|No|Vol|pp|al|approx|dept|est|'
    r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.',
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    """
    Split a block of prose into sentences.

    Strategy:
      1. Temporarily mask common abbreviation dots so they are not treated
         as sentence boundaries.
      2. Split on sentence-terminal punctuation followed by whitespace and
         an uppercase letter, open quote, or open parenthesis.
      3. Restore masked dots and drop trivially short fragments (<12 chars).
    """
    text = _ABBREV_RE.sub(lambda m: m.group().replace('.', '<DOT>'), text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\(\[])', text)
    result = []
    for p in parts:
        p = p.replace('<DOT>', '.').strip()
        if len(p) > 12:
            result.append(p)
    # If splitting produced nothing useful, return the whole block as one sentence
    return result or ([text.replace('<DOT>', '.').strip()]
                      if text.strip() else [])


# ── Column detection via horizontal gutter analysis ──────────────────────────

def _detect_columns(page) -> int:
    """
    Detect whether a page has a two-column layout by looking for a
    near-empty vertical strip (gutter) flanked by dense word regions.

    Returns 1 (single column) or 2 (two-column).
    """
    try:
        words = page.extract_words()
    except Exception:
        return 1

    if not words or len(words) < 20:
        return 1

    n_buckets = 30
    bucket_w  = page.width / n_buckets
    counts    = [0] * n_buckets

    for w in words:
        b = min(int(w["x0"] / bucket_w), n_buckets - 1)
        counts[b] += 1

    # Skip the outermost 2 buckets (page margins)
    for i in range(2, n_buckets - 2):
        if counts[i] <= 1:                          # near-empty bucket = gutter
            left_density  = sum(counts[max(0, i - 4) : i])
            right_density = sum(counts[i + 1 : min(n_buckets, i + 5)])
            if left_density >= 6 and right_density >= 6:
                return 2
    return 1


# ── Font metadata extraction ──────────────────────────────────────────────────

def _get_line_metadata(page_or_crop) -> list[dict]:
    """
    Group words into lines and return per-line font attributes.

    Works on a pdfplumber Page or cropped Page.

    Returns list of dicts with keys:
        text  : str    -- joined word texts for the line
        size  : float  -- average font size
        bold  : bool   -- any word uses a bold font variant
        upper : bool   -- entire text is uppercase and contains letters
        y_top : float  -- topmost y-coordinate of the line
        y_bot : float  -- bottommost y-coordinate
    """
    try:
        words = page_or_crop.extract_words(extra_attrs=["size", "fontname"])
    except Exception:
        return []
    if not words:
        return []

    # Sort by (quantised top, x) for stable reading order within a line
    words = sorted(words, key=lambda w: (round(w["top"] / 2) * 2, w["x0"]))

    lines: list[list[dict]] = []
    cur:   list[dict]       = []
    prev_top = None

    for w in words:
        if prev_top is None or abs(w["top"] - prev_top) <= 4:
            cur.append(w)
        else:
            if cur:
                lines.append(cur)
            cur = [w]
        prev_top = w["top"]
    if cur:
        lines.append(cur)

    result = []
    for line in lines:
        text = " ".join(w["text"] for w in line).strip()
        if not text:
            continue
        sizes    = [w.get("size") or 0.0 for w in line]
        avg_size = sum(sizes) / len(sizes) if sizes else 0.0
        is_bold  = any("bold" in (w.get("fontname") or "").lower() for w in line)
        is_upper = text.upper() == text and any(c.isalpha() for c in text)
        result.append({
            "text":  text,
            "size":  avg_size,
            "bold":  is_bold,
            "upper": is_upper,
            "y_top": min(w["top"]    for w in line),
            "y_bot": max(w["bottom"] for w in line),
        })
    return result


def _modal_size(line_metas: list[dict]) -> float:
    """
    Return the modal font size across all lines.
    This is the body text size — used as the reference for heading thresholds.
    """
    sizes = [round(l["size"], 1) for l in line_metas if l["size"] > 0]
    if not sizes:
        return 10.0
    return Counter(sizes).most_common(1)[0][0]


def _heading_level(line: dict, body_size: float) -> int | None:
    """
    Classify a line as heading level 1, 2, or 3, or None (body text).

    Thresholds (tuned for academic, technical, and regulatory PDFs):

        H1 -- very large font (body + 4pt) OR large + bold (body + 2pt)
        H2 -- moderately larger (body + 1.5pt) AND (bold OR all-caps)
        H3 -- bold AND all-caps at approximately body size

    Single characters and purely numeric strings are always rejected
    (page numbers, footnote markers, list bullets).
    """
    s = line["size"]
    b = line["bold"]
    u = line["upper"]
    t = line["text"]

    stripped = t.replace(" ", "")
    if len(stripped) <= 2 or stripped.isdigit():
        return None

    if s >= body_size + 4.0:
        return 1
    if s >= body_size + 2.0 and b:
        return 1
    if s >= body_size + 1.5 and (b or u):
        return 2
    if b and u and s >= body_size - 0.5:
        return 3
    return None


# ── Caption extraction ────────────────────────────────────────────────────────

_CAPTION_LOOK_ABOVE = 80   # points above table top to scan for caption text


def _extract_caption(page, table_bbox: tuple, n_lines: int = 3) -> str:
    """
    Return up to n_lines of text immediately above the table bounding box.
    These lines are the most likely candidates for a table caption or title.
    """
    top = max(0.0, table_bbox[1] - _CAPTION_LOOK_ABOVE)
    bot = table_bbox[1]
    if bot - top < 2:
        return ""
    try:
        crop = page.crop((0, top, page.width, bot))
        text = crop.extract_text() or ""
    except Exception:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return " ".join(lines[-n_lines:])


# ── Title inference ───────────────────────────────────────────────────────────

def _infer_title(pdf) -> str:
    """
    Infer document title using two strategies in order:
      1. pdfplumber PDF metadata 'Title' key, if present and non-trivial.
      2. The largest-font line in the top 40% of page 1.
    """
    meta_title = (pdf.metadata or {}).get("Title", "").strip()
    if meta_title and len(meta_title) > 4:
        return meta_title

    if not pdf.pages:
        return ""

    lines = _get_line_metadata(pdf.pages[0])
    if not lines:
        return ""

    top_region = [l for l in lines if l["y_top"] < pdf.pages[0].height * 0.40]
    pool = top_region if top_region else lines
    return max(pool, key=lambda l: l["size"])["text"]


# ── Reference section guard ───────────────────────────────────────────────────

_REF_HEADING_RE = re.compile(
    r'^\s*(references?|bibliography|works\s+cited|'
    r'citations?|acknowledgements?|appendix)\s*$',
    re.IGNORECASE,
)


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_pdf(pdf_path: str) -> dict:
    """
    Parse a PDF and return its structured content.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    dict -- see module docstring for full schema.

    Raises
    ------
    OSError -- if the PDF cannot be opened.
    """
    try:
        pdf_handle = pdfplumber.open(pdf_path)
    except OSError as e:
        raise OSError(f"Cannot open PDF: {pdf_path!r}\n{e}") from e

    doc: dict = {"title": "", "pages": {}}

    with pdf_handle as pdf:
        doc["title"] = _infer_title(pdf)

        # ── Cross-page section state ───────────────────────────────────────
        # heading and level persist across page boundaries so that
        # continuation text is correctly attributed in step 4.
        current_heading     : str       = ""
        current_level       : int       = 1
        current_text_buffer : list[str] = []
        in_references       : bool      = False

        for page_num, page in enumerate(pdf.pages, start=1):

            # ── Detect column layout ───────────────────────────────────────
            num_cols = _detect_columns(page)

            if num_cols == 1:
                columns = [page]
            else:
                mid_x   = page.width / 2
                columns = [
                    page.crop((0,     0, mid_x,      page.height)),
                    page.crop((mid_x, 0, page.width, page.height)),
                ]

            # ── Extract tables ─────────────────────────────────────────────
            page_tables   : list[dict]                = []
            table_y_ranges: list[tuple[float, float]] = []

            try:
                for ft in page.find_tables():
                    raw = ft.extract()
                    if not raw or len(raw) < 2:
                        continue
                    bbox    = ft.bbox
                    caption = _extract_caption(page, bbox)
                    page_tables.append({
                        "raw":     raw,
                        "caption": caption,
                        "bbox":    bbox,
                        "page":    page_num,
                    })
                    table_y_ranges.append((bbox[1], bbox[3]))
            except Exception:
                pass

            def _in_table_area(y_mid: float) -> bool:
                return any(y0 <= y_mid <= y1 for y0, y1 in table_y_ranges)

            # ── Extract prose in reading order (column by column) ──────────
            # Each column's lines are already top-to-bottom.  Processing
            # left column then right column preserves two-column reading order.
            page_sections  : list[dict] = []
            col_line_metas : list[dict] = []

            for col in columns:
                col_lines = _get_line_metadata(col)
                body_size = _modal_size(col_lines)
                for line in col_lines:
                    line["_body_size"] = body_size
                col_line_metas.extend(col_lines)

            for line in col_line_metas:
                body_size = line.pop("_body_size")

                # Skip lines that fall inside a table bounding box
                y_mid = (line["y_top"] + line["y_bot"]) / 2
                if _in_table_area(y_mid):
                    continue

                level = _heading_level(line, body_size)

                if level is not None:
                    # ── Flush accumulated prose as a completed section ─────
                    if current_text_buffer:
                        joined  = " ".join(current_text_buffer)
                        cleaned = _clean_text(joined)
                        sents   = _split_sentences(cleaned)
                        if sents:
                            page_sections.append({
                                "heading":   current_heading,
                                "level":     current_level,
                                "sentences": sents,
                                "page":      page_num,
                            })
                        current_text_buffer = []

                    # ── Detect reference / bibliography heading ────────────
                    if _REF_HEADING_RE.match(line["text"]):
                        in_references = True

                    current_heading = line["text"]
                    current_level   = level

                else:
                    if in_references:
                        continue
                    current_text_buffer.append(line["text"])

            # ── End-of-page flush ──────────────────────────────────────────
            # Flush whatever is in the buffer as a page-boundary section.
            # current_heading / current_level intentionally persist so that
            # the first prose block on the next page is attributed to the
            # same section heading (cross-page continuity).
            if current_text_buffer:
                joined  = " ".join(current_text_buffer)
                cleaned = _clean_text(joined)
                sents   = _split_sentences(cleaned)
                if sents:
                    page_sections.append({
                        "heading":   current_heading,
                        "level":     current_level,
                        "sentences": sents,
                        "page":      page_num,
                    })
                current_text_buffer = []

            doc["pages"][page_num] = {
                "sections": page_sections,
                "tables":   page_tables,
            }

    return doc
