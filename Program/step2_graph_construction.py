"""
Step 2 — Knowledge Graph Construction
======================================
Builds a multi-layered knowledge graph from the structured output of
step 1.  All node embeddings are computed here and stored directly on
nodes as numpy arrays (portable, pickle-safe).

Complete node and edge schema
-----------------------------

Physical backbone:
    PDF  --HAS_PAGE-->  Page  --NEXT_PAGE / PREV_PAGE-->  Page

Logical hierarchy:
    Section  --STARTS_ON-->      Page        (starting page of section)
    Section  --PARENT_SECTION--> Section     (upward: child -> parent)
    Section  --HAS_SUBSECTION--> Section     (downward: parent -> child)

Content (indexed nodes):
    SemanticChunk  --IN_SECTION-->   Section
    Table          --IN_SECTION-->   Section
    Table          --ON_PAGE-->      Page

Anchor / bridge nodes (not indexed, used for graph routing only):
    Sentence  --PART_OF_CHUNK-->  SemanticChunk
    Sentence  --ON_PAGE-->        Page
    Sentence  --NEXT_SENTENCE-->  Sentence
    Sentence  --PREV_SENTENCE-->  Sentence

Node attributes
---------------
    PDF           : title, path, page_count
    Page          : page  (int)
    Section       : heading, level (1/2/3), start_page, text, embedding
    SemanticChunk : text, page (start page), section, n_sentences, embedding
    Sentence      : text, index (global int), page
    Table         : caption, raw, matrix, page, text, embedding

Node embeddings (all-mpnet-base-v2, stored as float32 numpy arrays):
    SemanticChunk  -- full chunk text
    Table          -- caption + column headers
    Section        -- section heading text

Semantic chunking algorithm
---------------------------
1. Embed all sentences in a section with the lightweight split model
   (all-MiniLM-L6-v2).
2. Compute cosine distance between each consecutive sentence pair.
3. Place a boundary where distance > mean + SIGMA * std_dev.
   This relative threshold adapts per-section: sections with uniform
   topic flow get fewer splits; sections with abrupt topic shifts get more.
4. Safety cap: MAX_CHUNK_SENTENCES prevents degenerate single-chunk sections.
5. Encode each chunk with the heavy embed model (all-mpnet-base-v2).

Index (build_index)
-------------------
Collects SemanticChunk and Table nodes into a mean-centred dense
embedding matrix and a parallel BM25 sparse index.
Section embeddings stay on nodes only -- accessed during graph traversal
for score boosting, not included in the primary index matrix.
"""

from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25Okapi      = None
    _BM25_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Lightweight model -- sentence boundary detection only (ingestion time).
_SPLIT_MODEL_NAME = "all-MiniLM-L6-v2"

# Heavy model -- final node embeddings (ingestion) and query encoding (step 3).
_EMBED_MODEL_NAME = "all-mpnet-base-v2"

# Relative threshold sensitivity.  Higher = fewer, larger chunks.
SIGMA_THRESHOLD = 1.5

# Hard cap: any chunk > this is force-split.
MAX_CHUNK_SENTENCES = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_split_model: SentenceTransformer | None = None
_embed_model: SentenceTransformer | None = None


# ---------------------------------------------------------------------------
# Model loaders (lazy, cached)
# ---------------------------------------------------------------------------

def _get_split_model() -> SentenceTransformer:
    global _split_model
    if _split_model is None:
        print(f"  Loading split model ({_SPLIT_MODEL_NAME}) on {DEVICE.upper()}...",
              flush=True)
        _split_model = SentenceTransformer(_SPLIT_MODEL_NAME, device=DEVICE)
    return _split_model


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"  Loading embed model ({_EMBED_MODEL_NAME}) on {DEVICE.upper()}...",
              flush=True)
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME, device=DEVICE)
    return _embed_model


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

def _semantic_chunk(
    sentences   : list[str],
    split_model : SentenceTransformer,
    sigma       : float = SIGMA_THRESHOLD,
) -> list[list[str]]:
    """
    Partition a list of sentences into semantically coherent chunks using
    a cosine-distance spike detector with a relative threshold.

    Boundary rule
    -------------
    Place a boundary between sentence i and i+1 when:
        cosine_distance(i, i+1)  >  mean(distances) + sigma * std(distances)

    This adapts per section:
        - Uniform topic flow  -> high threshold -> fewer, larger chunks
        - Abrupt topic shifts -> lower threshold -> tighter, focused chunks

    Safety cap
    ----------
    Chunks exceeding MAX_CHUNK_SENTENCES are force-split into equal parts.

    Returns
    -------
    List of sentence groups; each becomes one SemanticChunk node.
    """
    if len(sentences) <= 2:
        return [sentences]

    embs = split_model.encode(
        sentences,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=DEVICE,
    )

    distances: list[float] = []
    for i in range(len(sentences) - 1):
        sim = util.cos_sim(embs[i], embs[i + 1]).item()
        distances.append(1.0 - sim)

    mean_d = float(np.mean(distances))
    std_d  = float(np.std(distances))
    thresh = mean_d + sigma * std_d

    raw_chunks: list[list[str]] = []
    current   : list[str]       = [sentences[0]]

    for i, dist in enumerate(distances):
        if dist > thresh:
            raw_chunks.append(current)
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    if current:
        raw_chunks.append(current)

    # Safety cap
    final_chunks: list[list[str]] = []
    for chunk in raw_chunks:
        if len(chunk) <= MAX_CHUNK_SENTENCES:
            final_chunks.append(chunk)
        else:
            for start in range(0, len(chunk), MAX_CHUNK_SENTENCES):
                final_chunks.append(chunk[start : start + MAX_CHUNK_SENTENCES])

    return final_chunks


def _batch_encode(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode texts and return a float32 numpy array (n x d)."""
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    embs = model.encode(
        texts,
        batch_size=64,
        convert_to_tensor=False,
        show_progress_bar=False,
        device=DEVICE,
    )
    return np.array(embs, dtype=np.float32)


# ---------------------------------------------------------------------------
# BM25 tokenizer (must match step3 and step4 exactly)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:-[\w]+)*", text.lower())


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_knowledge_graph(doc: dict, pdf_path: str) -> nx.MultiDiGraph:
    """
    Build the knowledge graph from extracted PDF content.

    Parameters
    ----------
    doc      : dict returned by step1.extract_pdf()
    pdf_path : path to the source PDF

    Returns
    -------
    nx.MultiDiGraph -- fully populated with all nodes and edges per schema.
    """
    G           = nx.MultiDiGraph()
    split_model = _get_split_model()
    embed_model = _get_embed_model()
    stem        = Path(pdf_path).stem

    # ------------------------------------------------------------------
    # PDF node
    # ------------------------------------------------------------------
    pdf_node = f"pdf::{stem}"
    G.add_node(
        pdf_node,
        type       = "PDF",
        label      = doc["title"] or stem,
        title      = doc["title"],
        path       = str(pdf_path),
        page_count = len(doc["pages"]),
    )

    # ------------------------------------------------------------------
    # Page nodes  +  HAS_PAGE  +  NEXT_PAGE / PREV_PAGE
    # ------------------------------------------------------------------
    sorted_pages = sorted(doc["pages"].keys())

    for p in sorted_pages:
        pid = f"page::{p}"
        G.add_node(pid, type="Page", label=f"Page {p}", page=p)
        G.add_edge(pdf_node, pid, relation="HAS_PAGE")

    for i in range(len(sorted_pages) - 1):
        p1, p2 = sorted_pages[i], sorted_pages[i + 1]
        G.add_edge(f"page::{p1}", f"page::{p2}", relation="NEXT_PAGE")
        G.add_edge(f"page::{p2}", f"page::{p1}", relation="PREV_PAGE")

    # ------------------------------------------------------------------
    # First pass: Section and Sentence nodes
    #
    # We accumulate all sentences per section before running semantic
    # chunking so that the full cosine-distance sequence is available.
    # ------------------------------------------------------------------
    section_counter : int = 0
    sentence_counter: int = 0   # global sentence index across the whole doc

    # Cross-page section state
    last_section_heading: str | None = None
    last_section_id     : str | None = None

    # level -> section_id currently at that level (for hierarchy)
    section_stack: dict[int, str] = {}

    # Accumulated data keyed by section_id
    # section_id -> list of sentence texts (in order, may span pages)
    section_sentence_texts: dict[str, list[str]] = {}
    # section_id -> list of sentence node IDs (parallel to texts)
    section_sent_ids      : dict[str, list[str]] = {}
    # section_id -> last sentence node ID (for NEXT/PREV links)
    section_last_sent     : dict[str, str | None] = {}

    # page_num -> last active section node ID (for table IN_SECTION edge)
    page_active_section: dict[int, str | None] = {}

    for page_num in sorted_pages:
        page_data = doc["pages"][page_num]
        page_id   = f"page::{page_num}"

        # Inherit previous page's active section for cross-page continuity
        last_active: str | None = last_section_id

        for section_data in page_data["sections"]:
            heading   = section_data["heading"]
            level     = section_data["level"]
            sentences = section_data["sentences"]
            if not sentences:
                continue

            # ── Section node (new vs. cross-page continuation) ─────────────
            is_continuation = (
                heading == last_section_heading
                and last_section_id is not None
            )

            if is_continuation:
                section_id = last_section_id
            else:
                section_counter += 1
                section_id = f"section::{section_counter}"

                G.add_node(
                    section_id,
                    type       = "Section",
                    label      = heading or "(untitled)",
                    heading    = heading,
                    level      = level,
                    start_page = page_num,
                    text       = heading,
                )

                # STARTS_ON: Section -> Page (starting page)
                G.add_edge(section_id, page_id, relation="STARTS_ON")

                # Section hierarchy: pop same-or-deeper levels from stack
                for lvl in [k for k in section_stack if k >= level]:
                    del section_stack[lvl]

                # PARENT_SECTION (child -> parent) and HAS_SUBSECTION (parent -> child)
                for parent_lvl in sorted(section_stack.keys(), reverse=True):
                    if parent_lvl < level:
                        parent_id = section_stack[parent_lvl]
                        G.add_edge(section_id, parent_id,
                                   relation="PARENT_SECTION")
                        G.add_edge(parent_id, section_id,
                                   relation="HAS_SUBSECTION")
                        break

                section_stack[level] = section_id

                last_section_heading              = heading
                last_section_id                   = section_id
                section_sentence_texts[section_id] = []
                section_sent_ids      [section_id] = []
                section_last_sent     [section_id] = None

            last_active = section_id

            # ── Sentence nodes ─────────────────────────────────────────────
            section_sentence_texts[section_id].extend(sentences)
            prev_sent_id = section_last_sent[section_id]

            for sent_text in sentences:
                sentence_counter += 1
                sent_id = f"sentence::{sentence_counter}"

                G.add_node(
                    sent_id,
                    type  = "Sentence",
                    text  = sent_text,
                    index = sentence_counter,   # global sentence position
                    page  = page_num,
                )

                # ON_PAGE: Sentence -> Page  (the bridge between logical and physical)
                G.add_edge(sent_id, page_id, relation="ON_PAGE")

                # Sequential links within the same section
                if prev_sent_id:
                    G.add_edge(prev_sent_id, sent_id, relation="NEXT_SENTENCE")
                    G.add_edge(sent_id, prev_sent_id, relation="PREV_SENTENCE")

                prev_sent_id = sent_id
                section_sent_ids[section_id].append(sent_id)

            section_last_sent[section_id] = prev_sent_id

        page_active_section[page_num] = last_active

        # ── Table nodes ────────────────────────────────────────────────────
        for t_idx, table_data in enumerate(page_data["tables"]):
            table_id = f"table::{page_num}::{t_idx}"
            caption  = table_data["caption"]
            raw      = table_data["raw"]

            # Parse through step 2 table parser for clean column headers
            matrix = (raw)
            try:
                if matrix and matrix.get("columns"):
                    header_text = " | ".join(
                        c["header"] for c in matrix["columns"] if c["header"]
                    )
                else:
                    header_text = " | ".join(
                        str(c) for c in (raw[0] if raw else []) if c
                    )
            except Exception as e:
                header_text = ""

            table_text = f"{caption} {header_text}".strip()

            G.add_node(
                table_id,
                type    = "Table",
                label   = caption or f"Table on page {page_num}",
                caption = caption,
                raw     = raw,
                matrix  = matrix,
                page    = page_num,
                text    = table_text,
            )

            # ON_PAGE: Table -> Page
            G.add_edge(table_id, page_id, relation="ON_PAGE")

            # IN_SECTION: Table -> Section (nearest active section on this page)
            active_section = page_active_section.get(page_num)
            if active_section:
                G.add_edge(table_id, active_section, relation="IN_SECTION")

    # ------------------------------------------------------------------
    # Second pass: Semantic chunking
    #
    # All sentences per section are now collected, so cosine distances
    # are computed over the full section sequence.
    # ------------------------------------------------------------------
    print("  Running semantic chunking...", flush=True)

    chunk_counter  : int       = 0
    all_chunk_texts: list[str] = []
    chunk_id_order : list[str] = []

    for section_id, sentences in section_sentence_texts.items():
        if not sentences:
            continue

        chunks   = _semantic_chunk(sentences, split_model, sigma=SIGMA_THRESHOLD)
        sent_ids = section_sent_ids[section_id]
        heading  = G.nodes[section_id]["heading"]
        start_pg = G.nodes[section_id]["start_page"]
        sent_pos = 0

        for chunk_sents in chunks:
            chunk_counter += 1
            chunk_id   = f"chunk::{chunk_counter}"
            chunk_text = " ".join(chunk_sents)

            G.add_node(
                chunk_id,
                type        = "SemanticChunk",
                text        = chunk_text,
                page        = start_pg,   # starting page; exact page distribution
                                          # is computed at query time via sentence traversal
                section     = heading,
                n_sentences = len(chunk_sents),
            )

            # IN_SECTION: SemanticChunk -> Section
            G.add_edge(chunk_id, section_id, relation="IN_SECTION")

            # PART_OF_CHUNK: Sentence -> SemanticChunk
            # Uses positional mapping (order-preserving, O(n))
            for _ in chunk_sents:
                if sent_pos < len(sent_ids):
                    G.add_edge(sent_ids[sent_pos], chunk_id,
                               relation="PART_OF_CHUNK")
                    sent_pos += 1

            all_chunk_texts.append(chunk_text)
            chunk_id_order.append(chunk_id)

    # ------------------------------------------------------------------
    # Batch embedding
    # ------------------------------------------------------------------

    # SemanticChunk nodes
    if all_chunk_texts:
        print(f"  Embedding {len(all_chunk_texts)} semantic chunks...", flush=True)
        chunk_embs = _batch_encode(all_chunk_texts, embed_model)
        for i, cid in enumerate(chunk_id_order):
            G.nodes[cid]["embedding"] = chunk_embs[i]

    # Section title nodes
    section_ids    = [n for n, d in G.nodes(data=True) if d.get("type") == "Section"]
    section_titles = [G.nodes[s]["heading"] for s in section_ids]
    if section_titles:
        print(f"  Embedding {len(section_titles)} section titles...", flush=True)
        sect_embs = _batch_encode(section_titles, embed_model)
        for i, sid in enumerate(section_ids):
            G.nodes[sid]["embedding"] = sect_embs[i]

    # Table nodes
    table_ids   = [n for n, d in G.nodes(data=True) if d.get("type") == "Table"]
    table_texts = [G.nodes[t]["text"] for t in table_ids]
    if table_texts:
        print(f"  Embedding {len(table_texts)} tables...", flush=True)
        table_embs = _batch_encode(table_texts, embed_model)
        for i, tid in enumerate(table_ids):
            G.nodes[tid]["embedding"] = table_embs[i]

    print(
        f"  Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"({len(chunk_id_order)} chunks, {len(table_ids)} tables, "
        f"{sentence_counter} sentences).",
        flush=True,
    )
    return G


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(G: nx.MultiDiGraph) -> dict:
    """
    Build a retrieval index from pre-computed node embeddings.

    Primary index (dense + BM25): SemanticChunk, Table
    Boost-only    (not in index):  Section embeddings on nodes, read during
                                   graph traversal in step 3.

    Returns
    -------
    {
        "ids"         : list[str]         -- node IDs in index order
        "types"       : list[str]         -- "SemanticChunk" or "Table"
        "texts"       : list[str]         -- raw text (for BM25)
        "embeddings"  : torch.Tensor      -- (n x d) float32, mean-centred
        "corpus_mean" : torch.Tensor      -- (1 x d) float32
        "bm25"        : BM25Okapi | None
    }
    """
    ids     : list[str] = []
    types   : list[str] = []
    texts   : list[str] = []
    raw_embs: list      = []

    for node_id, data in G.nodes(data=True):
        if data.get("type") not in ("SemanticChunk", "Table"):
            continue
        emb = data.get("embedding")
        if emb is None:
            continue
        ids.append(node_id)
        types.append(data["type"])
        texts.append(data.get("text", ""))
        raw_embs.append(emb)

    if not ids:
        return {
            "ids": [], "types": [], "texts": [],
            "embeddings": None, "corpus_mean": None, "bm25": None,
        }

    print(
        f"  Building index: {len(ids)} nodes "
        f"({types.count('SemanticChunk')} chunks, {types.count('Table')} tables)...",
        flush=True,
    )

    embeddings  = torch.tensor(np.stack(raw_embs), dtype=torch.float32)
    corpus_mean = embeddings.mean(dim=0, keepdim=True)
    embeddings  = embeddings - corpus_mean

    bm25 = None
    if _BM25_AVAILABLE:
        tokenized = [_tokenize(t) for t in texts]
        try:
            bm25 = _BM25Okapi(tokenized)
            print(f"  BM25 ready ({len(ids)} docs).", flush=True)
        except Exception as e:
            print(f"  Warning: BM25 construction failed ({e}).", flush=True)
    else:
        print("  Note: rank-bm25 not installed -- dense-only retrieval.", flush=True)

    return {
        "ids"        : ids,
        "types"      : types,
        "texts"      : texts,
        "embeddings" : embeddings,
        "corpus_mean": corpus_mean,
        "bm25"       : bm25,
    }
