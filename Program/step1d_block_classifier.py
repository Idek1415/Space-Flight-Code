"""
Step 1d — Block Classifier
===========================
Lightweight, training-free classifier that labels text blocks extracted
from a PDF page as one of:

    "table"       — structured data in rows/columns
    "description" — flowing prose (product info, explanation)
    "summary"     — concise paragraph summarising a section or table
    "caption"     — short label above/below a figure or table
    "heading"     — section or subsection title

Five features (derived without a neural model):
    1. numeric_ratio       — fraction of tokens that are purely numeric
    2. short_line_ratio    — fraction of lines under 30 characters
    3. x_stdev             — std dev of word x-coords (low = columnar = table)
    4. line_height_stdev   — std dev of inter-line spacing (low = uniform rows)
    5. sentence_score      — fraction of lines ending with sentence punctuation

Research basis: ablation studies (VTLayout) show spatial features (x_stdev,
line_height_stdev) carry most discriminative power for tables vs figures,
while text features resolve heading/paragraph ambiguity.
"""

import re
import statistics
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Feature container
# ---------------------------------------------------------------------------

@dataclass
class BlockFeatures:
    text:             str         = ""
    lines:            list        = field(default_factory=list)
    word_xs:          list        = field(default_factory=list)
    line_heights:     list        = field(default_factory=list)
    context_above:    str         = ""
    context_below:    str         = ""
    numeric_ratio:    float       = 0.0
    short_line_ratio: float       = 0.0
    x_stdev:          float       = 999.0   # sentinel = unavailable
    line_height_stdev:float       = 999.0
    avg_line_len:     float       = 0.0
    sentence_score:   float       = 0.0
    has_context_kw:   bool        = False


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_NUMERIC_RE      = re.compile(r'^-?\d[\d.,/°%±×]*$')
_SENTENCE_END_RE = re.compile(r'[.!?]["\']?\s*$')
_TABLE_CTX_RE    = re.compile(
    r'\b(table|tbl\.?|fig\.?|figure|note:|notes?:|see\s+table)\b',
    re.IGNORECASE,
)

# Thresholds (tuned for engineering datasheets)
_T_NUMERIC   = 0.22   # >22% numeric tokens → table signal
_T_SHORT     = 0.55   # >55% short lines    → table signal
_T_X_MAX     = 55.0   # low x-variance      → columnar
_T_LH_MAX    = 7.0    # uniform row heights → table
_S_MAX_LINES = 5      # short paragraph     → summary candidate
_S_SENT_MIN  = 0.50   # mostly sentences    → summary / description
_H_MAX_CHARS = 90
_H_MAX_LINES = 2
_C_MAX_CHARS = 130
_C_MAX_LINES = 3


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    text: str,
    *,
    word_xs: list | None = None,
    line_heights: list | None = None,
    context_above: str = "",
    context_below: str = "",
) -> BlockFeatures:
    """
    Compute classification features from a text block.

    Args:
        text:          Raw text of the block.
        word_xs:       X0 coordinates of each word on the page (pdfplumber
                       extract_words). None if unavailable — spatial signals
                       will be skipped and only text signals used.
        line_heights:  Y-distances between consecutive text baselines.
        context_above: Text immediately above this block.
        context_below: Text immediately below this block.
    """
    f = BlockFeatures(
        text=text,
        word_xs=word_xs or [],
        line_heights=line_heights or [],
        context_above=context_above,
        context_below=context_below,
    )

    lines = [l for l in text.splitlines() if l.strip()]
    f.lines = lines
    if not lines:
        return f

    tokens = text.split()
    n_tok  = len(tokens)

    # Numeric ratio
    if n_tok:
        f.numeric_ratio = sum(
            1 for t in tokens if _NUMERIC_RE.match(t.strip("()[],."))
        ) / n_tok

    # Short line ratio
    f.short_line_ratio = sum(1 for l in lines if len(l.strip()) < 30) / len(lines)

    # X-coordinate std dev (spatial: low = column-aligned)
    if len(f.word_xs) >= 4:
        try:
            f.x_stdev = statistics.stdev(f.word_xs)
        except statistics.StatisticsError:
            f.x_stdev = 999.0

    # Line height std dev (spatial: low = uniform rows)
    if len(f.line_heights) >= 3:
        try:
            f.line_height_stdev = statistics.stdev(f.line_heights)
        except statistics.StatisticsError:
            f.line_height_stdev = 999.0

    # Average line length
    f.avg_line_len = sum(len(l.strip()) for l in lines) / len(lines)

    # Sentence score
    f.sentence_score = (
        sum(1 for l in lines if _SENTENCE_END_RE.search(l)) / len(lines)
    )

    # Context keyword (nearby text says "Table 3" or "Figure 2")
    ctx = (context_above + " " + context_below).strip()
    f.has_context_kw = bool(_TABLE_CTX_RE.search(ctx) or _TABLE_CTX_RE.search(text[:60]))

    return f


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_block(f: BlockFeatures) -> str:
    """
    Classify a BlockFeatures instance.

    Returns one of: "table", "description", "summary", "caption", "heading"
    """
    lines    = f.lines
    n_lines  = len(lines)
    n_chars  = sum(len(l.strip()) for l in lines)

    if not lines:
        return "description"

    # ── Heading ──────────────────────────────────────────────────────────────
    # Very short, no sentence punctuation, low numeric density
    if (
        n_lines  <= _H_MAX_LINES
        and n_chars  <= _H_MAX_CHARS
        and f.sentence_score < 0.25
        and f.numeric_ratio  < 0.08
    ):
        return "heading"

    # ── Caption ──────────────────────────────────────────────────────────────
    # Short, and the text or context contains "Table X" / "Fig. X"
    if (
        n_lines  <= _C_MAX_LINES
        and n_chars  <= _C_MAX_CHARS
        and f.has_context_kw
    ):
        return "caption"

    # ── Table ─────────────────────────────────────────────────────────────────
    # Vote-based: each spatial/text signal contributes.
    votes = 0
    if f.numeric_ratio       >= _T_NUMERIC:             votes += 2
    if f.short_line_ratio    >= _T_SHORT:               votes += 1
    if f.x_stdev             <  _T_X_MAX  and f.x_stdev != 999.0:     votes += 2
    if f.line_height_stdev   <  _T_LH_MAX and f.line_height_stdev != 999.0: votes += 1
    if f.has_context_kw:                                votes += 1

    if votes >= 3:
        return "table"

    # ── Summary ──────────────────────────────────────────────────────────────
    # Short paragraph composed mostly of complete sentences
    if (
        n_lines        <= _S_MAX_LINES
        and f.sentence_score >= _S_SENT_MIN
        and f.numeric_ratio  <  0.12
    ):
        return "summary"

    # ── Default: description ─────────────────────────────────────────────────
    return "description"


def classify_text(
    text: str,
    *,
    context_above: str = "",
    context_below: str = "",
) -> str:
    """Classify a raw string with no spatial data (text-only signals)."""
    f = extract_features(text, context_above=context_above, context_below=context_below)
    return classify_block(f)


# ---------------------------------------------------------------------------
# Page-level batch classification
# ---------------------------------------------------------------------------

def classify_page_blocks(page, use_generative: bool = False) -> list[dict]:
    """
    Extract and classify all text blocks on a pdfplumber Page.

    Groups words into blocks by vertical proximity, then classifies each.

    Returns:
        List of dicts:
        [{"text": "...", "block_type": "description", "bbox": (x0,y0,x1,y1)}, ...]
    """
    V_GAP    = 6    # max y-gap within a line (points)
    B_GAP    = 18   # min y-gap to start a new block

    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
    except Exception:
        return []

    if not words:
        return []

    words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    # Group words into lines by top-y proximity
    lines: list[list[dict]] = []
    cur_line: list[dict] = []
    prev_top = None

    for w in words:
        if prev_top is None or abs(w["top"] - prev_top) <= V_GAP:
            cur_line.append(w)
        else:
            if cur_line:
                lines.append(cur_line)
            cur_line = [w]
        prev_top = w["top"]
    if cur_line:
        lines.append(cur_line)

    # Group lines into blocks by bottom-of-line → top-of-next-line gap
    blocks: list[list[list[dict]]] = []
    cur_block: list[list[dict]] = []
    prev_bottom = None

    for line in lines:
        line_top = min(w["top"] for w in line)
        if prev_bottom is None or (line_top - prev_bottom) <= B_GAP:
            cur_block.append(line)
        else:
            if cur_block:
                blocks.append(cur_block)
            cur_block = [line]
        prev_bottom = max(w["bottom"] for w in line)
    if cur_block:
        blocks.append(cur_block)

    # Classify each block
    results = []
    for b_idx, block_lines in enumerate(blocks):
        all_words = [w for line in block_lines for w in line]
        text = " ".join(w["text"] for w in all_words).strip()
        if not text:
            continue

        xs = [w["x0"] for w in all_words]

        # Line heights: distance between consecutive line tops
        line_tops = [min(w["top"] for w in ln) for ln in block_lines]
        lhs = [line_tops[i+1] - line_tops[i] for i in range(len(line_tops)-1)]

        x0 = min(w["x0"]    for w in all_words)
        y0 = min(w["top"]   for w in all_words)
        x1 = max(w["x1"]    for w in all_words)
        y1 = max(w["bottom"]for w in all_words)

        ctx_above_words = [w for ln in (blocks[b_idx-1] if b_idx > 0 else []) for w in ln]
        ctx_below_words = [w for ln in (blocks[b_idx+1] if b_idx < len(blocks)-1 else []) for w in ln]
        ctx_above = " ".join(w["text"] for w in ctx_above_words)
        ctx_below = " ".join(w["text"] for w in ctx_below_words)

        f = extract_features(
            text,
            word_xs=xs,
            line_heights=lhs,
            context_above=ctx_above,
            context_below=ctx_below,
        )
        spatial_class = classify_block(f)

        # Generative second-opinion for ambiguous cases only
        # (avoids loading the model for clear-cut classifications).
        # Borderline = votes==2 (just below table threshold) OR
        # spatial_class=="description" with suspiciously short avg line.
        if use_generative and spatial_class in ("description", "summary"):
            try:
                from step1e_page_description import classify_block_gen
                # Check if this block looks borderline
                _votes = 0
                if f.numeric_ratio >= _T_NUMERIC:   _votes += 2
                if f.short_line_ratio >= _T_SHORT:  _votes += 1
                if f.x_stdev < _T_X_MAX and f.x_stdev != 999.0: _votes += 2
                if f.line_height_stdev < _T_LH_MAX and f.line_height_stdev != 999.0: _votes += 1
                if _votes == 2:   # borderline — ask Flan-T5
                    spatial_class = classify_block_gen(text, spatial_class)
            except Exception:
                pass  # generative assist is best-effort

        results.append({
            "text":       text,
            "block_type": spatial_class,
            "bbox":       (x0, y0, x1, y1),
        })

    return results
