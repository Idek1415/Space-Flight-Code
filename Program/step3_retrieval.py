"""
Step 3 — Retrieval
==================
Hardware-aware query pipeline.  Accepts a query string, searches the
pre-built index, traverses the knowledge graph for context, and returns
a ranked list of the most relevant pages.

Y-Fork architecture
-------------------
Both paths share Stages 1 and 2.  They diverge at Stage 3.

    Stage 1 — Hybrid candidate retrieval  (shared)
        Dense cosine search + BM25, fused with Reciprocal Rank Fusion.
        Produces the top N candidate nodes (SemanticChunk or Table).

    Stage 2 — Graph traversal  (shared)
        For each candidate, traverse the graph upward and downward:
            Upward   : chunk --IN_SECTION--> Section  (grab header)
            Downward : chunk <--PART_OF_CHUNK-- Sentence --ON_PAGE--> Page
        This resolves which physical pages the chunk spans and assembles
        the context block for scoring.

    Stage 3A — CPU Path: Heuristic voting
        Distribute the candidate's RRF score across its pages in proportion
        to the number of sentences on each page.  Apply a section-title
        similarity boost and a table-presence boost.  Accumulate votes
        per page and return the top-k pages.

    Stage 3B — GPU Path: Cross-Encoder reranking
        Pass (query, context_block) pairs through a CUDA Cross-Encoder.
        Distribute the sigmoid-normalised score across the candidate's
        pages proportionally.  Return the top-k pages by score.

Page resolution via sentence traversal
---------------------------------------
A SemanticChunk may straddle a page boundary.  The correct primary page
is determined by counting sentences per page:

    chunk <--PART_OF_CHUNK-- Sentence --ON_PAGE--> Page

If a chunk has 4 sentences on page 4 and 2 on page 5, the primary page
is 4 (66%), but the result reports "pages 4-5" so the caller knows to
show both pages.  Votes/scores are distributed proportionally (4 gets
66% of the score, 5 gets 33%).

For Table nodes, the ON_PAGE edge is direct:
    Table --ON_PAGE--> Page

Return format (list of dicts, sorted descending by score)
----------------------------------------------------------
{
    "rank"              : int,
    "page"              : int,      primary page (most sentences)
    "page_range"        : str,      "4" or "4-5" if cross-page
    "score"             : float,
    "node_id"           : str,
    "node_type"         : str,      "SemanticChunk" | "Table"
    "chunk_text"        : str,
    "section"           : str,
    "section_score"     : float,    cosine sim between query and section title
    "table_caption"     : str,      non-empty when best hit is a Table node
    "context_block"     : str,      assembled text sent to Cross-Encoder
    "page_distribution" : dict,     {page_num: fraction_of_sentences}
}

Dependencies
------------
    pip install sentence-transformers rank-bm25 torch networkx
"""

from __future__ import annotations

import re
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Must match the heavy model used at index time in step2.
_EMBED_MODEL_NAME    = "all-mpnet-base-v2"
_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

_RRF_K        = 60
_DENSE_WEIGHT = 0.60
_BM25_WEIGHT  = 0.40

_CPU_CANDIDATES = 10
_GPU_CANDIDATES = 20

# Heuristic vote weights (CPU path)
_VOTE_CHUNK_HIT     = 1.0   # base score from RRF
_VOTE_SECTION_TITLE = 1.5   # multiplier applied when section sim >= threshold
_VOTE_TABLE_PRESENT = 0.5   # additive bonus when section contains a Table node

_SECTION_BOOST_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Negation query detection and boosting
# ---------------------------------------------------------------------------
# When a query contains negation language ("not permitted", "SHALL NOT",
# "what cannot", etc.), the source chunks that answer it use restrictive
# vocabulary ("SHALL NOT", "restricted", "prohibited") that standard dense
# and BM25 retrieval under-weights because the query rarely contains those
# exact marker tokens.
#
# Fix: detect negation queries, then add a third BM25 ranking built from
# a curated set of restriction-marker tokens.  This is fused into RRF at
# _NEGATION_WEIGHT (lower than the primary BM25 to avoid over-biasing).
#
# _NEGATION_TRIGGERS: patterns whose presence in the query activates boosting.
# _NEGATION_MARKERS:  tokens injected into the supplementary BM25 query.
# _NEGATION_WEIGHT:   RRF weight for the supplementary ranking.

_NEGATION_TRIGGERS: list[str] = [
    r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bwithout\b",
    r"\bexcept\b", r"\bexclud", r"\bshall not\b", r"\bcannot\b",
    r"\bcan't\b", r"\bdon't\b", r"\bdoesn't\b", r"\bisn't\b",
    r"\baren't\b", r"\bwon't\b", r"\bwouldn't\b", r"\bshouldn't\b",
    r"\bprohibit", r"\bforbid", r"\bdisallow", r"\bimpermissible\b",
    r"\brestrict", r"\bunauthori[sz]ed\b", r"\bineligible\b",
]

# Tokens that commonly appear in source chunks that answer restriction queries.
# These are passed to BM25 as the supplementary "negation query".
_NEGATION_MARKERS: list[str] = [
    "shall", "not", "restricted", "prohibited", "forbidden",
    "disallowed", "excluded", "unauthorized", "must not", "cannot",
    "impermissible", "ineligible", "restriction", "limitation",
    "not permitted", "not allowed", "not required", "not applicable",
]

_NEGATION_WEIGHT: float = 0.25   # RRF weight for supplementary negation ranking

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_embed_model   : SentenceTransformer | None = None
_reranker_model: CrossEncoder        | None = None


# ---------------------------------------------------------------------------
# Model loaders (lazy, cached)
# ---------------------------------------------------------------------------

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"  Loading embed model ({_EMBED_MODEL_NAME}) on {DEVICE.upper()}...",
              flush=True)
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME, device=DEVICE)
    return _embed_model


def _get_reranker() -> CrossEncoder | None:
    global _reranker_model
    if _reranker_model is None:
        try:
            print(
                f"  Loading reranker ({_RERANKER_MODEL_NAME}) on {DEVICE.upper()}...",
                flush=True,
            )
            _reranker_model = CrossEncoder(
                _RERANKER_MODEL_NAME,
                device=DEVICE,
                max_length=512,
            )
        except Exception as e:
            print(f"  Reranker unavailable ({e}) -- falling back to CPU path.",
                  flush=True)
            _reranker_model = False   # type: ignore[assignment]
    return _reranker_model if _reranker_model else None


# ---------------------------------------------------------------------------
# BM25 helpers (tokenizer must match step2 and step4 exactly)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:-[\w]+)*", text.lower())


def _bm25_scores(bm25_index, query_tokens: list[str]) -> list[float]:
    if bm25_index is None:
        return []
    try:
        return bm25_index.get_scores(query_tokens).tolist()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

_NEGATION_PATTERN = re.compile(
    "|".join(_NEGATION_TRIGGERS),
    re.IGNORECASE,
)


def _is_negation_query(query_text: str) -> bool:
    """
    Return True if the query contains negation or restriction language.

    A single-pass regex check over the compiled trigger list.  Fast enough
    to run on every query without measurable overhead.
    """
    return bool(_NEGATION_PATTERN.search(query_text))


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    rankings: list[list[int]],
    weights : list[float],
    k       : int = _RRF_K,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion (Cormack et al. 2009)."""
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Stage 1 — Hybrid candidate retrieval (shared)
# ---------------------------------------------------------------------------

def _hybrid_candidates(
    query_text: str,
    index     : dict,
    top_k     : int,
) -> list[tuple[int, float]]:
    """
    Dense cosine + BM25 retrieval, fused with Reciprocal Rank Fusion.

    When the query contains negation language, a supplementary BM25 ranking
    built from restriction-marker tokens (_NEGATION_MARKERS) is added as a
    third RRF component at _NEGATION_WEIGHT.  This surfaces chunks that use
    "SHALL NOT", "restricted", "prohibited" etc., which standard retrieval
    under-weights because those tokens rarely appear in the query itself.

    Returns
    -------
    list of (index_position, fused_score) sorted descending, len <= top_k.
    """
    model      = _get_embed_model()
    embeddings = index["embeddings"]
    n          = len(index["ids"])

    if embeddings is None or n == 0:
        return []

    if str(embeddings.device) != DEVICE:
        embeddings = embeddings.to(DEVICE)
        index["embeddings"] = embeddings

    # Dense
    query_emb   = model.encode(query_text, convert_to_tensor=True, device=DEVICE)
    corpus_mean = index.get("corpus_mean")
    if corpus_mean is not None:
        cm = corpus_mean.to(device=query_emb.device, dtype=query_emb.dtype)
        query_emb = query_emb - cm

    dense_sims    = util.cos_sim(query_emb, embeddings)[0].tolist()
    dense_ranking = sorted(range(n), key=lambda i: dense_sims[i], reverse=True)

    # Primary BM25
    bm25_raw = _bm25_scores(index.get("bm25"), _tokenize(query_text))

    if bm25_raw:
        bm25_top       = max(bm25_raw)
        bm25_threshold = bm25_top * 0.05
        bm25_ranking   = sorted(
            [i for i in range(n) if bm25_raw[i] >= bm25_threshold],
            key=lambda i: bm25_raw[i],
            reverse=True,
        )
        rankings = [dense_ranking, bm25_ranking]
        weights  = [_DENSE_WEIGHT, _BM25_WEIGHT]
    else:
        rankings = [dense_ranking]
        weights  = [_DENSE_WEIGHT + _BM25_WEIGHT]  # consolidate weight on dense

    # Supplementary negation BM25 ranking
    # Activated only when the query contains negation/restriction language.
    # Uses a curated token set that matches restrictive source vocabulary
    # ("SHALL NOT", "restricted", "prohibited") rather than query tokens.
    if bm25_raw and _is_negation_query(query_text):
        negation_tokens = _tokenize(" ".join(_NEGATION_MARKERS))
        neg_raw = _bm25_scores(index.get("bm25"), negation_tokens)
        if neg_raw:
            neg_top       = max(neg_raw)
            neg_threshold = neg_top * 0.05
            neg_ranking   = sorted(
                [i for i in range(n) if neg_raw[i] >= neg_threshold],
                key=lambda i: neg_raw[i],
                reverse=True,
            )
            rankings.append(neg_ranking)
            weights.append(_NEGATION_WEIGHT)
            print("  [step3] Negation query detected — applying restriction-marker boost.",
                  flush=True)

    fused = _rrf_fuse(rankings, weights)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Stage 2 — Graph traversal (shared)
# ---------------------------------------------------------------------------

def _resolve_pages(G, node_id: str, node_type: str) -> dict[int, int]:
    """
    Traverse the graph to discover which physical pages a node occupies
    and how many sentences belong to each page.

    For SemanticChunk:
        chunk <--PART_OF_CHUNK-- Sentence --ON_PAGE--> Page

    For Table:
        Table --ON_PAGE--> Page  (direct, counts as 1)

    Returns
    -------
    {page_num: sentence_count}  -- empty dict if traversal finds nothing.
    """
    page_counts: dict[int, int] = {}

    if node_type == "Table":
        for _, tgt, edata in G.out_edges(node_id, data=True):
            if edata.get("relation") == "ON_PAGE":
                pg = G.nodes[tgt].get("page")
                if pg is not None:
                    page_counts[pg] = 1
        return page_counts

    # SemanticChunk: walk sentences via PART_OF_CHUNK (incoming to chunk)
    for src, _, edata in G.in_edges(node_id, data=True):
        if edata.get("relation") != "PART_OF_CHUNK":
            continue
        # src is a Sentence node; follow its ON_PAGE edge
        for _, tgt, sedge in G.out_edges(src, data=True):
            if sedge.get("relation") == "ON_PAGE":
                pg = G.nodes[tgt].get("page")
                if pg is not None:
                    page_counts[pg] = page_counts.get(pg, 0) + 1

    return page_counts


def _page_distribution(page_counts: dict[int, int]) -> dict[int, float]:
    """
    Convert sentence counts per page into fractional distribution.
    E.g. {4: 4, 5: 2} -> {4: 0.667, 5: 0.333}
    """
    total = sum(page_counts.values())
    if total == 0:
        return {}
    return {pg: cnt / total for pg, cnt in page_counts.items()}


def _primary_page(page_counts: dict[int, int]) -> int | None:
    """Return the page with the most sentences, or None if empty."""
    if not page_counts:
        return None
    return max(page_counts, key=lambda pg: page_counts[pg])


def _page_range_str(page_counts: dict[int, int]) -> str:
    """Return "4" for single-page chunks, "4-5" for cross-page chunks."""
    pages = sorted(page_counts.keys())
    if not pages:
        return "?"
    if len(pages) == 1:
        return str(pages[0])
    return f"{pages[0]}-{pages[-1]}"


def _get_section_for_node(G, node_id: str) -> tuple[str, str]:
    """
    Walk IN_SECTION to find the parent Section node.
    Returns (section_node_id, section_heading).
    """
    for _, tgt, edata in G.out_edges(node_id, data=True):
        if edata.get("relation") == "IN_SECTION":
            return tgt, G.nodes[tgt].get("heading", "")
    return "", ""


def _get_table_in_section(G, section_id: str) -> str:
    """
    Return caption of the first Table IN_SECTION -> section_id, else "".
    """
    if not section_id:
        return ""
    for src, _, edata in G.in_edges(section_id, data=True):
        if edata.get("relation") == "IN_SECTION":
            if G.nodes[src].get("type") == "Table":
                return G.nodes[src].get("caption", "")
    return ""


def _section_embedding(G, section_id: str) -> np.ndarray | None:
    if not section_id:
        return None
    return G.nodes[section_id].get("embedding")


def _assemble_context_block(G, node_id: str, node_data: dict) -> str:
    """
    Assemble the context string for the Cross-Encoder:
        [Section: <title>] <chunk or table text> [Table: <caption>]
    """
    section_id, section_heading = _get_section_for_node(G, node_id)
    table_caption               = _get_table_in_section(G, section_id)

    if node_data.get("type") == "SemanticChunk":
        core = node_data.get("text", "")
    else:
        raw        = node_data.get("raw") or []
        header_row = " | ".join(str(c) for c in raw[0] if c) if raw else ""
        core       = f"{node_data.get('caption', '')} {header_row}".strip()

    parts = []
    if section_heading:
        parts.append(f"[Section: {section_heading}]")
    if core:
        parts.append(core)
    if table_caption and table_caption != core:
        parts.append(f"[Table: {table_caption}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Stage 3A — CPU path: heuristic graph voting
# ---------------------------------------------------------------------------

def _cpu_score(
    query_text: str,
    candidates: list[tuple[int, float]],
    index     : dict,
    G,
) -> dict[int, float]:
    """
    Distribute each candidate's RRF score to its pages proportionally,
    applying section-title and table-presence boosts.

    Scoring per candidate
    ---------------------
        base_vote = rrf_score * VOTE_CHUNK_HIT
        if section_sim >= SECTION_BOOST_THRESHOLD:
            base_vote += VOTE_SECTION_TITLE
        if section contains a Table node:
            base_vote += VOTE_TABLE_PRESENT
        page_vote[page] += base_vote * fraction_of_sentences_on_page

    Returns
    -------
    {page_num: cumulative_vote_score}
    """
    model       = _get_embed_model()
    q_emb       = model.encode(query_text, convert_to_tensor=True, device=DEVICE)
    page_votes  : dict[int, float] = {}

    for idx, rrf_score in candidates:
        node_id   = index["ids"][idx]
        node_type = index["types"][idx]
        node_data = G.nodes[node_id]

        page_counts = _resolve_pages(G, node_id, node_type)
        if not page_counts:
            # Fallback: use the stored page attribute
            pg = node_data.get("page")
            if pg is not None:
                page_counts = {pg: 1}
            else:
                continue

        dist = _page_distribution(page_counts)

        # Compute vote before distributing across pages
        vote = rrf_score * _VOTE_CHUNK_HIT

        section_id, _ = _get_section_for_node(G, node_id)
        if section_id:
            se = _section_embedding(G, section_id)
            if se is not None:
                st  = torch.tensor(se, dtype=torch.float32, device=DEVICE)
                sim = util.cos_sim(q_emb, st).item()
                if sim >= _SECTION_BOOST_THRESHOLD:
                    vote += _VOTE_SECTION_TITLE
            if _get_table_in_section(G, section_id):
                vote += _VOTE_TABLE_PRESENT

        # Distribute proportionally across pages
        for pg, fraction in dist.items():
            page_votes[pg] = page_votes.get(pg, 0.0) + vote * fraction

    return page_votes


# ---------------------------------------------------------------------------
# Stage 3B — GPU path: cross-encoder reranking
# ---------------------------------------------------------------------------

def _gpu_score(
    query_text: str,
    candidates: list[tuple[int, float]],
    index     : dict,
    G,
) -> dict[int, float]:
    """
    Score (query, context_block) pairs with a Cross-Encoder, then
    distribute each score proportionally across the candidate's pages.

    Falls back to _cpu_score if the reranker is unavailable.

    Returns
    -------
    {page_num: best_cross_encoder_score}
    """
    reranker = _get_reranker()
    if reranker is None:
        return _cpu_score(query_text, candidates, index, G)

    # Build (context_block, page_distribution) per candidate
    context_blocks: list[str]            = []
    distributions : list[dict[int, float]] = []

    for idx, _ in candidates:
        node_id   = index["ids"][idx]
        node_type = index["types"][idx]
        node_data = G.nodes[node_id]

        page_counts = _resolve_pages(G, node_id, node_type)
        if not page_counts:
            pg = node_data.get("page")
            if pg is not None:
                page_counts = {pg: 1}
            else:
                continue

        context_blocks.append(_assemble_context_block(G, node_id, node_data))
        distributions.append(_page_distribution(page_counts))

    if not context_blocks:
        return {}

    pairs = [(query_text, ctx) for ctx in context_blocks]
    try:
        raw_scores  = reranker.predict(pairs)
        norm_scores = (1.0 / (1.0 + np.exp(-raw_scores))).tolist()
    except Exception as e:
        print(f"  Cross-Encoder failed ({e}) -- falling back to CPU path.",
              flush=True)
        return _cpu_score(query_text, candidates, index, G)

    # Distribute each score across its pages; keep the max per page
    page_scores: dict[int, float] = {}
    for dist, score in zip(distributions, norm_scores):
        for pg, fraction in dist.items():
            weighted = score * fraction
            if weighted > page_scores.get(pg, -1.0):
                page_scores[pg] = weighted

    return page_scores


# ---------------------------------------------------------------------------
# Stage 4 — Result assembly (shared)
# ---------------------------------------------------------------------------

def _build_results(
    query_text  : str,
    page_scores : dict[int, float],
    candidates  : list[tuple[int, float]],
    index       : dict,
    G,
    top_k       : int,
) -> list[dict]:
    """
    Map scored pages back to their best candidate node, compute page
    range strings and distributions, and assemble the output dicts.
    """
    model       = _get_embed_model()
    q_emb       = model.encode(query_text, convert_to_tensor=True, device=DEVICE)
    corpus_mean = index.get("corpus_mean")
    if corpus_mean is not None:
        q_emb = q_emb - corpus_mean.to(device=q_emb.device, dtype=q_emb.dtype)

    # Map primary_page -> best candidate index (first occurrence wins)
    page_best_idx: dict[int, tuple[int, dict[int, int]]] = {}
    for idx, _ in candidates:
        node_id   = index["ids"][idx]
        node_type = index["types"][idx]
        node_data = G.nodes[node_id]

        page_counts = _resolve_pages(G, node_id, node_type)
        if not page_counts:
            pg = node_data.get("page")
            page_counts = {pg: 1} if pg is not None else {}

        primary = _primary_page(page_counts)
        if primary is not None and primary not in page_best_idx:
            page_best_idx[primary] = (idx, page_counts)

    sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
    results      = []

    for rank, (primary_pg, score) in enumerate(sorted_pages[:top_k], start=1):
        entry = page_best_idx.get(primary_pg)
        if entry is None:
            continue

        idx, page_counts = entry
        node_id          = index["ids"][idx]
        node_type        = index["types"][idx]
        node_data        = G.nodes[node_id]

        section_id, section_heading = _get_section_for_node(G, node_id)
        table_caption               = _get_table_in_section(G, section_id)
        context_block               = _assemble_context_block(G, node_id, node_data)

        sect_sim = 0.0
        if section_id:
            se = _section_embedding(G, section_id)
            if se is not None:
                st       = torch.tensor(se, dtype=torch.float32, device=DEVICE)
                sect_sim = float(util.cos_sim(q_emb, st).item())

        # Collect sentences in document order for step 5 extractive compression.
        # Sentence nodes carry a global `index` attribute; sort on it to recover
        # document order regardless of graph traversal order.
        chunk_sentences: list[dict] = []
        if node_type == "SemanticChunk":
            for src, _, edata in G.in_edges(node_id, data=True):
                if edata.get("relation") == "PART_OF_CHUNK":
                    sdata = G.nodes[src]
                    chunk_sentences.append({
                        "text" : sdata.get("text", ""),
                        "index": sdata.get("index", 0),
                        "page" : sdata.get("page"),
                    })
            chunk_sentences.sort(key=lambda s: s["index"])

        results.append({
            "rank"             : rank,
            "page"             : primary_pg,
            "page_range"       : _page_range_str(page_counts),
            "score"            : round(score, 4),
            "normalized_score" : 0.0,   # filled in below after all scores are known
            "node_id"          : node_id,
            "node_type"        : node_type,
            "chunk_text"       : node_data.get("text", ""),
            "sentences"        : chunk_sentences,
            "section"          : section_heading,
            "section_score"    : round(sect_sim, 4),
            "table_caption"    : table_caption,
            "context_block"    : context_block,
            "page_distribution": _page_distribution(page_counts),
            # Query embedding stored as numpy array so step 5 can reuse it
            # without re-encoding.  All results in one query share the same
            # array reference; it is safe to read but must not be mutated.
            "query_embedding"  : q_emb.cpu().numpy(),
        })

    # ------------------------------------------------------------------
    # Min-max normalise scores across the returned set so step 5 has a
    # consistent [0, 1] confidence range for relevance marker prepending.
    # Guarded against degenerate single-result or zero-range cases.
    # ------------------------------------------------------------------
    if results:
        raw_scores = [r["score"] for r in results]
        s_min, s_max = min(raw_scores), max(raw_scores)
        score_range = s_max - s_min if s_max > s_min else 1.0
        for r in results:
            r["normalized_score"] = round((r["score"] - s_min) / score_range, 4)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query(
    G,
    index     : dict,
    query_text: str,
    top_k     : int = 5,
    timing_out: dict | None = None,
) -> list[dict]:
    """
    Run a hybrid retrieval query against the knowledge graph index.

    Automatically selects CPU or GPU scoring path based on
    torch.cuda.is_available().

    Parameters
    ----------
    G          : nx.MultiDiGraph from step2.build_knowledge_graph()
    index      : dict from step2.build_index()
    query_text : natural language query string
    top_k      : number of pages to return
    timing_out : if provided, cleared and filled with per-stage seconds for
                 this call only (``candidate_retrieval_s``, ``graph_traversal_scoring_s``,
                 ``result_assembly_s``, ``total_s``).

    Returns
    -------
    list of result dicts (see module docstring), sorted by score descending.
    """
    t_wall0 = time.perf_counter()

    def _emit_timing(
        cand_s: float,
        score_s: float,
        assembly_s: float,
    ) -> None:
        if timing_out is None:
            return
        total = time.perf_counter() - t_wall0
        timing_out.clear()
        timing_out.update({
            "candidate_retrieval_s"      : cand_s,
            "graph_traversal_scoring_s": score_s,
            "result_assembly_s"        : assembly_s,
            "total_s"                  : total,
        })

    if index.get("embeddings") is None:
        print("  Index is empty -- no results.")
        _emit_timing(0.0, 0.0, 0.0)
        return []

    use_gpu      = torch.cuda.is_available()
    n_candidates = _GPU_CANDIDATES if use_gpu else _CPU_CANDIDATES
    path_label   = "GPU (Cross-Encoder)" if use_gpu else "CPU (Heuristic)"
    print(f"  Query path: {path_label}", flush=True)

    # Stage 1: hybrid candidate retrieval
    t0 = time.perf_counter()
    candidates = _hybrid_candidates(query_text, index, n_candidates)
    t1 = time.perf_counter()
    cand_s = t1 - t0
    if not candidates:
        _emit_timing(cand_s, 0.0, 0.0)
        return []

    # Stage 2 + 3: path-specific scoring (traversal happens inside each scorer)
    t0 = time.perf_counter()
    if use_gpu:
        page_scores = _gpu_score(query_text, candidates, index, G)
    else:
        page_scores = _cpu_score(query_text, candidates, index, G)
    t1 = time.perf_counter()
    score_s = t1 - t0

    # Stage 4: result assembly
    t0 = time.perf_counter()
    out = _build_results(query_text, page_scores, candidates, index, G, top_k)
    t1 = time.perf_counter()
    assembly_s = t1 - t0
    _emit_timing(cand_s, score_s, assembly_s)
    return out


def format_results(results: list[dict], query_text: str) -> str:
    """Format query results as a human-readable string for CLI display."""
    lines = [f'\nQuery: "{query_text}"', "-" * 60]

    if not results:
        lines.append("  No matching results found.")
        return "\n".join(lines)

    for r in results:
        page_label = f"Pages {r['page_range']}" if "-" in r["page_range"] \
                     else f"Page {r['page']}"
        lines.append(
            f"\n  #{r['rank']}  Score: {r['score']:.4f}  |  "
            f"{page_label}  [{r['node_type']}]"
        )
        if r["section"]:
            lines.append(
                f"       Section   : {r['section']}  (sim: {r['section_score']:.3f})"
            )
        if r["table_caption"]:
            lines.append(f"       Table     : {r['table_caption']}")
        dist = r.get("page_distribution", {})
        if len(dist) > 1:
            dist_str = ", ".join(
                f"p{pg}: {pct:.0%}" for pg, pct in sorted(dist.items())
            )
            lines.append(f"       Pages     : {dist_str}")
        preview = r["chunk_text"]
        if len(preview) > 200:
            preview = preview[:197] + "..."
        lines.append(f"       Content   : {preview}")

    lines.append("")
    return "\n".join(lines)