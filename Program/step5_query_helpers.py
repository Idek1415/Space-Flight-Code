"""
Step 5 — NLP Query Engine  (Hybrid BM25 + Dense + RRF)
=======================================================
Pipeline
--------
1.  Negation parsing   → positive_query + penalty terms
2.  Synonym expansion  → expanded_query (WordNet + domain table)
3.  Dense scoring      → cosine sim with negation penalty
4.  BM25 scoring       → sparse lexical scoring on expanded query
5.  RRF fusion         → Reciprocal Rank Fusion of both rankings
6.  (Optional) HyDE    → average hypothetical-doc embedding into query
7.  Page aggregation   → max+mean blend per page
8.  (Optional) CE      → cross-encoder reranking of top-20 pages

Research basis
--------------
- BM25+dense RRF improves BEIR nDCG@10 from 43.4 → 52.6+ (Hybrid IR
  Survey 2024; emergentmind/bm25 2025).
- Synonym expansion + CE reranking filters synonym noise while keeping
  recall gains (AGH IR at LongEval 2025).
- HyDE bridges query-document distributional gap; HyPE shows +42pp
  precision / +45pp recall on some datasets (Vake et al. 2025).
- Negation penalty: explicit scoring outperforms relying on embedding
  polarity (DEO 2025; Enhancing Negation Awareness 2025).
"""

from __future__ import annotations

import re
from functools import lru_cache

from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

from App.console_progress import status_line, status_line_done
from App.device_config import get_torch_device_str

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25Okapi = None       # type: ignore[assignment,misc]
    _BM25_AVAILABLE = False

try:
    import hnswlib as _hnswlib
    _HNSW_AVAILABLE = True
except ImportError:
    _hnswlib = None         # type: ignore[assignment,misc]
    _HNSW_AVAILABLE = False

MODEL_NAME = "all-mpnet-base-v2"
DEVICE     = get_torch_device_str()
_model     = None

EMBEDDER_MODELS = {
    "small": "all-MiniLM-L6-v2",
    "large": "all-mpnet-base-v2",
}

_INDEX_ENCODE_BATCH_SIZE = 64
_PRIOR_SIM_THRESHOLD     = 0.32
_PAGE_SCORE_ALPHA        = 0.70
_PAGE_SCORE_BETA         = 0.30
_NEGATION_PENALTY_WEIGHT = 0.55
_RRF_K                   = 60
_DENSE_WEIGHT            = 0.60
_BM25_WEIGHT             = 0.40
_BM25_CONFIDENCE_RATIO   = 2.5     # skip HyDE+CE when top BM25 ≥ 2.5× runner-up
_HNSW_M                  = 32
_HNSW_EF_CONSTRUCTION    = 200
_HNSW_EF_SEARCH          = 80
_HNSW_QUERY_CANDIDATES   = 256

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
_reranker: object = None

# Domain-specific engineering synonym map
_DOMAIN_SYNONYMS: dict[str, list[str]] = {
    "hydraulic":    ["fluid power", "hydro", "fluid pressure"],
    "pneumatic":    ["air powered", "compressed air"],
    "pressure":     ["psi", "force", "load"],
    "temperature":  ["temp", "thermal", "heat", "degrees"],
    "seal":         ["gasket", "o-ring", "sealing", "packing"],
    "compatible":   ["suitable", "resistant", "rated for"],
    "chemical":     ["solvent", "acid", "alkaline", "reagent"],
    "elastomer":    ["rubber", "polymer", "compound", "material"],
    "dynamic":      ["moving", "reciprocating", "rotary", "oscillating"],
    "static":       ["stationary", "fixed", "non-moving"],
    "maximum":      ["max", "upper limit", "highest", "peak"],
    "minimum":      ["min", "lower limit", "lowest"],
    "diameter":     ["ID", "OD", "bore", "size"],
    "hardness":     ["shore", "durometer"],
    "specification": ["spec", "standard", "rating"],
    "application":  ["use", "service", "environment"],
    "high":         ["elevated", "extreme", "increased"],
    "low":          ["reduced", "minimal", "decreased"],
    "resistant":    ["compatible", "suitable", "stable"],
}

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def configure_embedder_model(size: str) -> str:
    global MODEL_NAME, _model
    key = size.strip().lower()
    if key not in EMBEDDER_MODELS:
        raise ValueError(f"Unknown embedder model size: {size!r}")
    MODEL_NAME = EMBEDDER_MODELS[key]
    _model = None
    return MODEL_NAME


def configure_embedder_from_saved(model_name: str | None) -> str:
    global MODEL_NAME, _model
    if model_name and str(model_name).strip():
        MODEL_NAME = str(model_name).strip()
        _model = None
        print(f"  Query embedder fixed to saved KG model: {MODEL_NAME}")
    return MODEL_NAME


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  Loading NLP model ({MODEL_NAME}) on {DEVICE.upper()}...")
        if DEVICE == "cpu":
            print("  (CUDA not available — install CUDA PyTorch for GPU embedding.)",
                  flush=True)
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return _model


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            device = "cuda" if DEVICE == "cuda" else "cpu"
            print(f"  Loading reranker ({_RERANKER_MODEL}) on {device.upper()}…",
                  flush=True)
            _reranker = CrossEncoder(_RERANKER_MODEL, device=device, max_length=512)
        except Exception as e:
            print(f"  Reranker unavailable ({e}). Skipping CE reranking.", flush=True)
            _reranker = False
    return _reranker


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Tokenize keeping hyphenated terms and alphanumeric codes intact."""
    return re.findall(r"[\w]+(?:-[\w]+)*", text.lower())


def _build_bm25(texts: list[str]):
    if not _BM25_AVAILABLE or not texts:
        return None
    tokenized = [_tokenize(t) for t in texts]
    try:
        return _BM25Okapi(tokenized)
    except Exception:
        return None


def _bm25_scores(bm25_index, query_tokens: list[str]) -> list[float]:
    if bm25_index is None:
        return []
    try:
        return bm25_index.get_scores(query_tokens).tolist()
    except Exception:
        return []


def _bm25_is_confident(
    scores: list[float],
    ratio: float = _BM25_CONFIDENCE_RATIO,
) -> bool:
    """True when the top BM25 hit is well-separated from the runner-up.

    A high ratio means a strong lexical match exists and CE reranking
    would likely hurt by overriding the correct BM25 answer.
    """
    if not scores:
        return False
    sorted_desc = sorted(scores, reverse=True)
    top = sorted_desc[0]
    if top <= 0:
        return False
    runner_up = sorted_desc[1] if len(sorted_desc) > 1 else 0.0
    if runner_up <= 0:
        return True
    return top / runner_up >= ratio


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

def _rrf_fuse(
    rankings: list[list[int]],
    weights: list[float] | None = None,
    k: int = _RRF_K,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion (Cormack et al. 2009)."""
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Synonym expansion
# ---------------------------------------------------------------------------

def _ensure_wordnet() -> None:
    try:
        from nltk.corpus import wordnet
        wordnet.synsets("test")
    except Exception:
        try:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            pass


@lru_cache(maxsize=1024)
def _wordnet_synonyms(word: str, max_syn: int = 2) -> tuple[str, ...]:
    try:
        from nltk.corpus import wordnet
        seen: set[str] = set()
        results: list[str] = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name != word and name not in seen and len(name) > 2:
                    seen.add(name)
                    results.append(name)
                    if len(results) >= max_syn:
                        return tuple(results)
        return tuple(results)
    except Exception:
        return ()


SKIP_WORDS = {
    "a", "an", "the", "is", "are", "for", "of", "in", "on", "at",
    "to", "and", "or", "with", "by", "this", "that", "not", "no",
    "what", "which", "how", "when", "where", "who",
}


def expand_synonyms(query_text: str, max_per_word: int = 3) -> str:
    """
    Expand the query with domain synonyms + WordNet for improved recall.
    Synonym noise is tolerated here because the cross-encoder reranker
    filters irrelevant candidates at scoring time.
    """
    _ensure_wordnet()
    words = query_text.split()
    expanded: list[str] = []

    for word in words:
        expanded.append(word)
        wl = word.lower().strip(".,?!")
        if wl in SKIP_WORDS or len(wl) < 3:
            continue

        domain = _DOMAIN_SYNONYMS.get(wl, [])
        expanded.extend(domain[:max_per_word])

        remaining = max_per_word - len(domain[:max_per_word])
        if remaining > 0:
            for syn in _wordnet_synonyms(wl, max_syn=remaining):
                if syn not in domain:
                    expanded.append(syn)

    return " ".join(expanded)


# ---------------------------------------------------------------------------
# Negation parsing
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS = [
    re.compile(r'\bnot\s+(?:for\s+)?(\w+(?:\s+\w+){0,2})',    re.IGNORECASE),
    re.compile(r'\bno\s+(\w+(?:\s+\w+){0,2})',                 re.IGNORECASE),
    re.compile(r'\bwithout\s+(\w+(?:\s+\w+){0,2})',            re.IGNORECASE),
    re.compile(r'\bavoid(?:ing)?\s+(\w+(?:\s+\w+){0,2})',      re.IGNORECASE),
    re.compile(r'\bexcept\s+(\w+(?:\s+\w+){0,2})',             re.IGNORECASE),
    re.compile(r'\bexclud(?:e|ing)\s+(\w+(?:\s+\w+){0,2})',    re.IGNORECASE),
    re.compile(r'\bnon[-\s](\w+)',                              re.IGNORECASE),
    re.compile(r'\bunsuitable\s+for\s+(\w+(?:\s+\w+){0,2})',   re.IGNORECASE),
    re.compile(r'\bnot\s+recommended\s+for\s+(\w+(?:\s+\w+){0,2})', re.IGNORECASE),
]

_ANTONYM_STOPWORDS = {
    "a", "an", "the", "for", "of", "in", "on", "at",
    "use", "used", "using", "applications", "application",
}


@lru_cache(maxsize=512)
def _get_antonyms(word: str) -> list[str]:
    try:
        from nltk.corpus import wordnet
        antonyms: set[str] = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    antonyms.add(ant.name().replace("_", " "))
        return list(antonyms)
    except Exception:
        return []


def parse_negation(query: str) -> tuple[str, list[str]]:
    _ensure_wordnet()
    negated_terms: list[str] = []
    positive_query = query

    for pattern in _NEGATION_PATTERNS:
        for m in pattern.finditer(query):
            phrase = m.group(1).strip()
            negated_terms.append(phrase)
            for word in phrase.split():
                if word.lower() not in _ANTONYM_STOPWORDS:
                    for ant in _get_antonyms(word.lower()):
                        if ant not in negated_terms:
                            negated_terms.append(ant)
        positive_query = pattern.sub("", positive_query)

    positive_query = " ".join(positive_query.split())
    return positive_query, negated_terms


# ---------------------------------------------------------------------------
# HyDE — query-time expansion
# ---------------------------------------------------------------------------

_hyde_tok   = None
_hyde_model = None


def _get_hyde_model():
    global _hyde_tok, _hyde_model
    if _hyde_model is None:
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            name = "google/flan-t5-small"
            device = "cuda" if DEVICE == "cuda" else "cpu"
            _hyde_tok   = T5Tokenizer.from_pretrained(name)
            _hyde_model = T5ForConditionalGeneration.from_pretrained(name).to(device)
            _hyde_model.eval()
            print(f"  HyDE model loaded ({name} on {device.upper()}).", flush=True)
        except Exception as e:
            print(f"  HyDE model unavailable ({e}).", flush=True)
            _hyde_model = False
    return _hyde_tok, _hyde_model


def _hyde_embedding(query_text: str) -> torch.Tensor | None:
    """Generate hypothetical answer, embed it. Average with query emb downstream."""
    tok, model = _get_hyde_model()
    if not model:
        return None
    try:
        prompt = (
            "Answer the following question in 2-3 sentences as if you are "
            "an engineering reference document. Be specific and factual.\n"
            f"Question: {query_text}\nAnswer:"
        )
        device = next(model.parameters()).device
        inputs = tok(prompt, return_tensors="pt", max_length=256,
                     truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80,
                                 num_beams=4, early_stopping=True)
        hyp_text = tok.decode(out[0], skip_special_tokens=True).strip()
        if not hyp_text:
            return None
        embedder = _get_model()
        return embedder.encode(hyp_text, convert_to_tensor=True, device=DEVICE)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------

def _merge_queries_with_same_embedder(priors: list[str], current: str) -> str:
    cur = current.strip()
    ps  = [p.strip() for p in priors if p.strip()]
    if not ps:
        return cur
    if not cur:
        return " ".join(ps)
    model = _get_model()
    texts = ps + [cur]
    embs  = model.encode(texts, convert_to_tensor=True,
                          show_progress_bar=False, device=DEVICE)
    prior_embs, cur_emb = embs[:-1], embs[-1:]
    sims = util.cos_sim(prior_embs, cur_emb).squeeze(-1)
    idx_kept = [i for i, s in enumerate(sims.tolist()) if s >= _PRIOR_SIM_THRESHOLD]
    if not idx_kept:
        idx_kept = [len(ps) - 1]
    return " ".join(ps[i] for i in sorted(set(idx_kept))) + " " + cur


def _weighted_effective_string(merged_prefix: str, current: str) -> str:
    p = merged_prefix.strip()
    c = current.strip()
    if not p:
        return f"{c} | {c}"
    return f"{p} | {c} | {c}"


def build_effective_query(query_history: list[str], window: int = 2) -> str:
    if not query_history:
        return ""
    recent  = query_history[-window:] if len(query_history) >= window else query_history
    current = recent[-1]
    priors  = recent[:-1]
    if not priors:
        return _weighted_effective_string("", current)
    merged = _merge_queries_with_same_embedder(priors, current)
    return _weighted_effective_string(merged, current)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(G) -> dict:
    """
    Build dense + sparse index over all Row and Image nodes.

    Returns
    -------
    {
        "ids":        list[str],
        "types":      list[str],
        "texts":      list[str],
        "embeddings": torch.Tensor (n × d),
        "bm25":       BM25Okapi | None,
        "hnsw":       hnswlib.Index | None,
    }
    """
    model = _get_model()
    ids, types, texts = [], [], []
    for node, data in G.nodes(data=True):
        if data.get("type") in ("Row", "Image"):
            text = data.get("text", "")
            if text:
                ids.append(node)
                types.append(data["type"])
                texts.append(text)

    if not ids:
        return {"ids": [], "types": [], "texts": [],
                "embeddings": None, "bm25": None}

    print(f"  Indexing {len(ids)} nodes "
          f"({types.count('Row')} rows, {types.count('Image')} images)...")

    n  = len(texts)
    bs = max(1, min(_INDEX_ENCODE_BATCH_SIZE, n))
    chunks: list[torch.Tensor] = []

    for start in range(0, n, bs):
        end   = min(start + bs, n)
        batch = texts[start:end]
        pct   = round(end / n * 100)
        status_line(f"  Encoding (dense)... {pct}% ({end}/{n})")
        chunks.append(
            model.encode(batch, batch_size=len(batch), convert_to_tensor=True,
                          show_progress_bar=False, device=DEVICE)
        )

    embeddings = torch.cat(chunks, dim=0)
    status_line_done(f"  Encoding (dense)... 100% ({n}/{n}) — done.")

    # BM25 sparse index
    bm25 = None
    if _BM25_AVAILABLE:
        print("  Building BM25 index…", flush=True)
        bm25 = _build_bm25(texts)
        if bm25:
            print(f"  BM25 ready ({n} docs).", flush=True)
    else:
        print("  Note: rank-bm25 not installed — dense-only retrieval.", flush=True)

    # Optional ANN index (HNSW) for fast dense candidate retrieval.
    hnsw_index = _build_hnsw_index(embeddings)
    if _HNSW_AVAILABLE and hnsw_index is None:
        print("  Note: HNSW unavailable — falling back to full dense scan.", flush=True)

    return {"ids": ids, "types": types, "texts": texts,
            "embeddings": embeddings, "bm25": bm25, "hnsw": hnsw_index}


# ---------------------------------------------------------------------------
# Adaptive page scoring
# ---------------------------------------------------------------------------

def _adaptive_page_weights(
    page_to_items: dict[int, list],
) -> tuple[float, float]:
    """Return (alpha, beta) tuned to document density.

    Table-heavy docs benefit from max-score dominance (high alpha);
    prose-heavy docs benefit from a balanced mean blend (higher beta).
    """
    if not page_to_items:
        return _PAGE_SCORE_ALPHA, _PAGE_SCORE_BETA
    avg_items = sum(len(v) for v in page_to_items.values()) / len(page_to_items)
    if avg_items >= 5:
        return 0.80, 0.20
    if avg_items <= 2:
        return 0.55, 0.45
    return _PAGE_SCORE_ALPHA, _PAGE_SCORE_BETA


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query(
    G,
    index: dict,
    query_text: str,
    top_k: int = 5,
    use_generative: bool = False,
    use_hyde: bool = False,
) -> list[dict]:
    """
    Hybrid retrieval with RRF, synonym expansion, negation, and
    optional cross-encoder reranking / HyDE.
    """
    if index["embeddings"] is None:
        print("  No nodes in index.")
        return []

    model      = _get_model()
    embeddings = index["embeddings"]
    if hasattr(embeddings, "device") and str(embeddings.device) != DEVICE:
        embeddings = embeddings.to(DEVICE)
        index["embeddings"] = embeddings

    n = len(index["ids"])

    # 1. Synonym expansion first (catches synonyms of negated terms)
    expanded_full = expand_synonyms(query_text)

    # 2. Negation on expanded text (strips negated clusters + their synonyms)
    positive_query, negated_terms = parse_negation(expanded_full)
    if not positive_query.strip():
        positive_query = expanded_full
    expanded_query = positive_query

    # 3. BM25 scoring (computed early for confidence gate)
    bm25_raw = _bm25_scores(index.get("bm25"), _tokenize(expanded_query))

    # 4. Confidence gate — skip HyDE+CE when BM25 has a clear winner
    bm25_confident = _bm25_is_confident(bm25_raw) if bm25_raw else False
    if bm25_confident:
        use_hyde = False
        use_generative = False

    # 5. Dense scoring
    pos_emb = model.encode(expanded_query, convert_to_tensor=True, device=DEVICE)

    if use_hyde:
        hyde_emb = _hyde_embedding(query_text)
        if hyde_emb is not None:
            pos_emb = (pos_emb + hyde_emb) / 2.0

    candidate_idxs: list[int] | None = _hnsw_candidates(index, pos_emb, len(index["ids"]))
    if candidate_idxs:
        subset = embeddings[candidate_idxs]
        subset_scores = util.cos_sim(pos_emb, subset)[0].tolist()
        pos_scores = [-1.0] * n
        for idx, score in zip(candidate_idxs, subset_scores):
            pos_scores[idx] = score
    else:
        pos_scores = util.cos_sim(pos_emb, embeddings)[0].tolist()

    # 6. Negation penalty
    neg_scores = [0.0] * n
    for term in negated_terms:
        if not term.strip():
            continue
        try:
            neg_emb = model.encode(term, convert_to_tensor=True, device=DEVICE)
            if candidate_idxs:
                n_sims = util.cos_sim(neg_emb, embeddings[candidate_idxs])[0].tolist()
                for i, ns in zip(candidate_idxs, n_sims):
                    if ns > neg_scores[i]:
                        neg_scores[i] = ns
            else:
                n_sims  = util.cos_sim(neg_emb, embeddings)[0].tolist()
                for i, ns in enumerate(n_sims):
                    if ns > neg_scores[i]:
                        neg_scores[i] = ns
        except Exception:
            pass

    dense_net = [ps - _NEGATION_PENALTY_WEIGHT * ns
                 for ps, ns in zip(pos_scores, neg_scores)]

    # 7. RRF fusion (bm25_raw already computed above)
    dense_pool = candidate_idxs if candidate_idxs else list(range(n))
    dense_ranking = sorted(dense_pool, key=lambda i: dense_net[i], reverse=True)
    if bm25_raw:
        bm25_top = max(bm25_raw) if bm25_raw else 0.0
        bm25_threshold = bm25_top * 0.08
        bm25_ranking = sorted(
            [i for i in range(n) if bm25_raw[i] >= bm25_threshold],
            key=lambda i: bm25_raw[i],
            reverse=True,
        )
        fused = _rrf_fuse([dense_ranking, bm25_ranking],
                          weights=[_DENSE_WEIGHT, _BM25_WEIGHT])
        idx_to_score = {idx: sc for idx, sc in fused}
        final_scores = [idx_to_score.get(i, 0.0) for i in range(n)]
    else:
        if dense_pool:
            max_d = max(abs(dense_net[i]) for i in dense_pool) or 1.0
            final_scores = [0.0] * n
            for i in dense_pool:
                final_scores[i] = dense_net[i] / max_d
        else:
            final_scores = [0.0] * n

    # 8. Page aggregation (adaptive alpha/beta based on doc density)
    page_to_items: dict[int, list[tuple[int, float]]] = {}
    for idx, score in enumerate(final_scores):
        page = G.nodes[index["ids"][idx]].get("page")
        if page is None:
            continue
        page_to_items.setdefault(page, []).append((idx, score))

    if not page_to_items:
        return []

    alpha, beta = _adaptive_page_weights(page_to_items)
    page_scores: list[tuple[int, float]] = []
    for page, items in page_to_items.items():
        values = [s for _, s in items]
        pg_sc  = alpha * max(values) + beta * (sum(values) / len(values))
        page_scores.append((page, pg_sc))

    # Sort by fused page score so CE sees the true top-N candidates (dict order is arbitrary).
    page_scores.sort(key=lambda x: x[1], reverse=True)

    # 9. Optional cross-encoder reranking (skipped by confidence gate)
    if use_generative:
        page_scores = _ce_rerank(query_text, page_scores, G, top_n=20)

    # 10. Build result dicts
    results = []
    for rank, (page, page_score) in enumerate(page_scores[:top_k], start=1):
        best_idx, best_score = max(page_to_items[page], key=lambda x: x[1])
        node_id   = index["ids"][best_idx]
        node_type = index["types"][best_idx]
        node_data = G.nodes[node_id]

        page_node  = f"page::{page}"
        page_prose = (G.nodes[page_node].get("prose", "")
                      if G.has_node(page_node) else "")

        # Sentence-level relevance for highlighting
        top_sents = _top_sentences(node_data.get("text", ""), pos_emb, model, top_n=3)

        result: dict = {
            "rank":          rank,
            "score":         round(page_score, 4),
            "node_score":    round(best_score, 4),
            "type":          node_type,
            "page":          page,
            "page_prose":    page_prose,
            "negated":       negated_terms,
            "top_sentences": top_sents,
        }

        if node_type == "Row":
            result.update({
                "section":    node_data.get("section", ""),
                "caption":    node_data.get("table_caption", ""),
                "entity":     node_data.get("entity", ""),
                "conditions": _get_row_conditions(G, node_id),
                "node_id":    node_id,
            })
        elif node_type == "Image":
            result.update({
                "caption":      node_data.get("caption", ""),
                "path":         node_data.get("path", ""),
                "node_id":      node_id,
                "linked_table": _get_image_table(G, node_id),
            })

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CE reranking helper
# ---------------------------------------------------------------------------

def _ce_rerank(
    query_text: str,
    page_scores: list[tuple[int, float]],
    G,
    top_n: int = 20,
) -> list[tuple[int, float]]:
    reranker = _get_reranker()
    if not reranker:
        page_scores.sort(key=lambda x: x[1], reverse=True)
        return page_scores

    candidates = page_scores[:top_n]
    pairs = []
    for page, _ in candidates:
        pn = f"page::{page}"
        text = (G.nodes[pn].get("text", "") if G.has_node(pn) else "")[:512]
        pairs.append((query_text, text))

    if not pairs:
        page_scores.sort(key=lambda x: x[1], reverse=True)
        return page_scores

    try:
        raw  = reranker.predict(pairs)
        sig  = 1.0 / (1.0 + np.exp(-raw))
        reranked = sorted(
            zip([p for p, _ in candidates], sig.tolist()),
            key=lambda x: x[1], reverse=True,
        )
        return reranked + page_scores[top_n:]
    except Exception:
        page_scores.sort(key=lambda x: x[1], reverse=True)
        return page_scores


# ---------------------------------------------------------------------------
# Sentence-level relevance for PDF highlighting
# ---------------------------------------------------------------------------

def _top_sentences(
    node_text: str,
    query_emb: torch.Tensor,
    model: SentenceTransformer,
    top_n: int = 3,
) -> list[str]:
    """
    Return the top_n sentences from node_text most similar to query_emb.
    Used by the PDF viewer to highlight semantically relevant spans
    rather than exact keywords.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', node_text)
                 if len(s.strip()) > 15]
    if not sentences:
        return []
    try:
        sent_embs = model.encode(sentences, convert_to_tensor=True,
                                 show_progress_bar=False, device=DEVICE)
        sims   = util.cos_sim(query_emb.unsqueeze(0), sent_embs)[0].tolist()
        ranked = sorted(zip(sentences, sims), key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:top_n]]
    except Exception:
        return sentences[:top_n]


# ---------------------------------------------------------------------------
# Output formatting (CLI / main.py)
# ---------------------------------------------------------------------------

def format_results(results: list[dict], query_text: str) -> str:
    lines = [f'\nQuery: "{query_text}"']
    if results and results[0].get("negated"):
        lines.append(f'  Excluding: {", ".join(results[0]["negated"][:6])}')
    lines.append("-" * 60)

    if not results:
        lines.append("  No matching results found.")
        return "\n".join(lines)

    for r in results:
        lines.append(
            f"\n  #{r['rank']}  Score: {r['score']:.2f}"
            f"  |  Page {r['page']}  [{r['type']}]"
        )
        if r["type"] == "Row":
            lines.append(f"       Section : {r.get('section', '')}")
            lines.append(f"       Table   : {r.get('caption', '')}")
            prose = r.get("page_prose", "").strip()
            if prose:
                display = prose if len(prose) <= 300 else prose[:297] + "…"
                lines.append(f"       Page desc: {display}")
            for cond in r.get("conditions", []):
                lines.append(f"       Note    : {cond}")
        elif r["type"] == "Image":
            lines.append(f"       Caption : {r.get('caption', '')}")
            lines.append(f"       File    : {r.get('path', '')}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_row_conditions(G, row_id: str) -> list[str]:
    return [G.nodes[succ].get("label", "")
            for _, succ, ed in G.out_edges(row_id, data=True)
            if ed.get("relation") == "subject_to"]


def _get_image_table(G, image_id: str) -> str | None:
    for _, succ, ed in G.out_edges(image_id, data=True):
        if ed.get("relation") == "illustrates":
            return G.nodes[succ].get("label", "")
    return None


# ---------------------------------------------------------------------------
# HNSW helpers
# ---------------------------------------------------------------------------

def _to_unit_numpy(embeddings: torch.Tensor) -> np.ndarray:
    vecs = embeddings.detach().to("cpu", dtype=torch.float32).numpy()
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def _build_hnsw_index(embeddings: torch.Tensor):
    if not _HNSW_AVAILABLE or embeddings is None or embeddings.numel() == 0:
        return None
    try:
        vecs = _to_unit_numpy(embeddings)
        n, dim = vecs.shape
        idx = _hnswlib.Index(space="cosine", dim=dim)
        idx.init_index(max_elements=n, ef_construction=_HNSW_EF_CONSTRUCTION, M=_HNSW_M)
        idx.add_items(vecs, np.arange(n))
        idx.set_ef(max(_HNSW_EF_SEARCH, _HNSW_QUERY_CANDIDATES))
        print(f"  HNSW ready ({n} vectors).", flush=True)
        return idx
    except Exception:
        return None


def _ensure_hnsw(index: dict):
    if not _HNSW_AVAILABLE:
        return None
    existing = index.get("hnsw")
    if existing is not None:
        return existing
    emb = index.get("embeddings")
    if emb is None:
        return None
    built = _build_hnsw_index(emb)
    if built is not None:
        index["hnsw"] = built
    return built


def _hnsw_candidates(index: dict, query_emb: torch.Tensor, n: int) -> list[int] | None:
    hnsw = _ensure_hnsw(index)
    if hnsw is None or n == 0:
        return None
    try:
        q = _to_unit_numpy(query_emb.unsqueeze(0))
        k = min(_HNSW_QUERY_CANDIDATES, n)
        labels, _ = hnsw.knn_query(q, k=k)
        out = [int(i) for i in labels[0] if isinstance(i, (int, np.integer)) and i >= 0]
        return out or None
    except Exception:
        return None
