"""
Step 1e — Generative Page Description, Block Classification Assist,
          and HyPE (Hypothetical Prompt Embeddings) Index Enrichment
====================================================================
Three functions served by Flan-T5-small (77M params, ~300 MB):

1.  generate_page_descriptions(G)
    Generates a 2-3 sentence engineering summary for each Page node.
    Input is assembled from connected subnodes with importance weighting.

2.  classify_block_gen(text, spatial_class)
    Called from step1d for borderline (vote==2) blocks.
    Returns "table", "summary", "description", "heading", or "caption".

3.  generate_hype_queries(G)
    HyPE — Hypothetical Prompt Embeddings (Vake et al. 2025).
    For each Row node, generates 2 hypothetical search queries a user
    might ask if they were looking for that row's content.  These
    queries are appended to the node's "text" field BEFORE the NLP index
    is built, enriching the embedding representation.

    Research basis: HyPE improves retrieval precision by up to 42pp and
    recall by up to 45pp vs standard retrieval, with zero query-time cost
    because the generation happens at index time (Vake et al. 2025).
    The key insight: a document chunk is better represented by the
    questions it answers than by the text it contains, because it closes
    the query-document distributional gap.
"""

from __future__ import annotations

_GEN_MODELS = {
    "small": "google/flan-t5-small",
    "base":  "google/flan-t5-base",
}
_GEN_MODEL_NAME = _GEN_MODELS["small"]
_gen_tok   = None
_gen_model = None


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def configure_gen_model(size: str = "small") -> str:
    global _GEN_MODEL_NAME, _gen_tok, _gen_model
    key = size.strip().lower()
    if key not in _GEN_MODELS:
        raise ValueError(f"Unknown gen model size: {size!r}")
    _GEN_MODEL_NAME = _GEN_MODELS[key]
    _gen_tok   = None
    _gen_model = None
    return _GEN_MODEL_NAME


def _get_gen_model():
    global _gen_tok, _gen_model
    if _gen_model is None:
        try:
            import torch
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Loading generative model ({_GEN_MODEL_NAME}) on {device.upper()}…",
                  flush=True)
            _gen_tok   = T5Tokenizer.from_pretrained(_GEN_MODEL_NAME)
            _gen_model = T5ForConditionalGeneration.from_pretrained(_GEN_MODEL_NAME)
            _gen_model = _gen_model.to(device)
            _gen_model.eval()
        except Exception as e:
            print(f"  Generative model unavailable ({e}).", flush=True)
            _gen_model = False
    return _gen_tok, _gen_model


def is_available() -> bool:
    _, model = _get_gen_model()
    return bool(model)


# ---------------------------------------------------------------------------
# Core generation helper
# ---------------------------------------------------------------------------

def _run(prompt: str, max_new_tokens: int = 80) -> str:
    tok, model = _get_gen_model()
    if not model:
        return ""
    try:
        import torch
        device  = next(model.parameters()).device
        inputs  = tok(prompt, return_tensors="pt", max_length=512,
                      truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 num_beams=4, early_stopping=True,
                                 no_repeat_ngram_size=3)
        return tok.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""


def _truncate(text: str, chars: int) -> str:
    return text if len(text) <= chars else text[:chars - 1] + "…"


# ---------------------------------------------------------------------------
# 1. Page description generation
# ---------------------------------------------------------------------------

def _build_page_context(G, page_node_id: str) -> str:
    page_data   = G.nodes.get(page_node_id, {})
    raw_prose   = page_data.get("prose", "").strip()

    section_labels: list[str] = []
    table_captions: list[str] = []
    col_headers:    list[str] = []
    sample_rows:    list[str] = []
    image_caps:     list[str] = []

    for pred, _, edata in G.in_edges(page_node_id, data=True):
        if edata.get("relation") != "on_page":
            continue
        node = G.nodes[pred]
        ntype = node.get("type", "")

        if ntype == "Section":
            lbl = node.get("label", "").strip()
            if lbl and lbl not in section_labels:
                section_labels.append(lbl)
        elif ntype == "Table":
            cap = node.get("label", "").strip()
            if cap and cap not in table_captions:
                table_captions.append(cap)
        elif ntype == "Column":
            hdr  = node.get("header", "").strip()
            unit = node.get("unit", "").strip()
            lbl  = f"{hdr} ({unit})" if unit else hdr
            if lbl and lbl not in col_headers:
                col_headers.append(lbl)
        elif ntype == "Row" and len(sample_rows) < 3:
            text = node.get("text", "")
            chunk = text.split("|")[1].strip() if "|" in text else ""
            if chunk:
                sample_rows.append(_truncate(chunk, 120))
        elif ntype == "Image":
            cap = node.get("caption", "").strip()
            if cap:
                image_caps.append(_truncate(cap, 80))

    parts: list[str] = []
    if section_labels:
        parts.append(f"Section: {'; '.join(section_labels[:3])}")
    if table_captions:
        parts.append(f"Tables: {'; '.join(table_captions[:4])}")
    if col_headers:
        parts.append(f"Columns: {', '.join(col_headers[:8])}")
    if sample_rows:
        parts.append(f"Sample data: {' | '.join(sample_rows)}")
    if raw_prose:
        parts.append(f"Context: {_truncate(raw_prose, 200)}")
    if image_caps:
        parts.append(f"Images: {'; '.join(image_caps[:2])}")

    return "\n".join(parts)


def generate_page_descriptions(G, gen_size: str = "small") -> int:
    """
    Generate and write back natural language descriptions for all Page nodes.
    Returns the number of pages successfully enhanced.
    """
    configure_gen_model(gen_size)
    if not is_available():
        return 0

    page_ids = [n for n, d in G.nodes(data=True) if d.get("type") == "Page"]
    enhanced = 0

    for page_id in page_ids:
        ctx = _build_page_context(G, page_id)
        if not ctx.strip():
            continue

        prompt = (
            "Summarize the following page content in 2-3 sentences. "
            "Capture the main topic, key facts, specific values, and "
            "any important terms or definitions present.\n"
            f"{ctx}\nSummary:"
        )
        description = _run(prompt, max_new_tokens=80)
        if description:
            G.nodes[page_id]["text"]  = description
            G.nodes[page_id]["prose"] = description
            enhanced += 1

    print(f"  Generative page descriptions: {enhanced}/{len(page_ids)} enhanced.",
          flush=True)
    return enhanced


# ---------------------------------------------------------------------------
# 2. Block classification assist
# ---------------------------------------------------------------------------

def classify_block_gen(text: str, spatial_class: str) -> str:
    """
    Ask Flan-T5 to classify an ambiguous text block.
    Only called for borderline cases (vote score == 2 in step1d).
    """
    if not is_available():
        return spatial_class

    prompt = (
        "Classify the following text as one of: table, summary, "
        "description, heading, or caption. Answer with only one word.\n"
        f"Text: {_truncate(text, 300)}\nClassification:"
    )

    result = _run(prompt, max_new_tokens=6).lower().strip()
    _MAP = {
        "table": "table", "summary": "summary", "description": "description",
        "heading": "heading", "caption": "caption",
        "list": "description", "paragraph": "description", "text": "description",
        "figure": "caption",
    }
    for k, v in _MAP.items():
        if k in result:
            return v
    return spatial_class


# ---------------------------------------------------------------------------
# 3. HyPE — hypothetical prompt embeddings at index time
# ---------------------------------------------------------------------------

def generate_hype_queries(G, gen_size: str = "small") -> int:
    """
    HyPE: for each Row node, generate 2 hypothetical search queries a user
    might ask if they were looking for that row's content.  Append those
    queries to the node's "text" field so they get embedded into the index.

    This runs at BUILD time, with zero query-time cost.

    Research: Vake et al. 2025 (HyPE) showed up to +42pp precision and
    +45pp recall vs standard retrieval by flipping from
    "what does this document say?" to "what questions does this answer?"
    """
    configure_gen_model(gen_size)
    if not is_available():
        return 0

    row_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "Row"]
    enhanced  = 0

    for node_id, data in row_nodes:
        original_text = data.get("text", "")
        if not original_text:
            continue

        # Build a compact representation of the row for the prompt
        section = data.get("section", "")
        caption = data.get("table_caption", "")
        entity  = data.get("entity", "")
        row_preview = _truncate(original_text.split("|")[-1].strip()
                                if "|" in original_text
                                else original_text, 200)

        prompt = (
            "Given this data row from a document, write exactly 2 short search "
            "queries a user might type to find this specific information. "
            "Use vocabulary appropriate to the content domain. "
            "Output only the queries, one per line.\n"
            f"Section: {section}\nTable: {caption}\nEntity: {entity}\n"
            f"Data: {row_preview}\nQueries:"
        )

        raw = _run(prompt, max_new_tokens=60)
        if not raw:
            continue

        # Parse the two queries from the output
        queries = [q.strip().lstrip("12.-) ") for q in raw.splitlines()
                   if q.strip() and len(q.strip()) > 8][:2]
        if not queries:
            continue

        # Append hypothetical queries to the node text
        G.nodes[node_id]["text"] = (
            original_text + " | HyPE: " + " | ".join(queries)
        )
        enhanced += 1

    print(f"  HyPE queries generated: {enhanced}/{len(row_nodes)} rows enriched.",
          flush=True)
    return enhanced
