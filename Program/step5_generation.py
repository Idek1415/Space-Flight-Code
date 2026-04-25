"""
Step 5 — Generation with Post-Hoc Citation
===========================================
Decouples fluent generation from citation.  The LLM is given a context
block and asked to write a normal prose answer (restoring generation
quality).  A second, fully deterministic pass then splits that answer into
individual claims, embeds each one, and finds the best-matching source node
in the retrieval index.  If the similarity exceeds a threshold, the program
appends a hard-coded citation to that sentence.  Claims that fall below the
threshold are left uncited rather than fabricated.

Architecture
------------

  Component 1 — Graph Neighbourhood Expansion           [UNCHANGED]
  Component 2 — MMR Selection                           [UNCHANGED]

  Component 3 — Context Assembly + Single LLM Call      [RESTORED from v1]
      Assembles a readable context block from MMR-selected chunks, grouped
      by section.  Sends one prompt to the LLM: context + query.  Returns
      fluent prose with no citation markers.

  Component 4 — Claim Decomposition                     [NEW]
      Regex sentence-splits the generated answer into individual claim
      strings.  No NLP dependency required.

  Component 5 — Per-Claim Retrieval & Citation Injection [UPDATED]
      For each claim, a four-stage scoring pipeline finds the best source:
        a. Cosine pre-selection: embed the claim, dot-product against the
           full index, keep the top CE_CANDIDATES (default 20) nodes.
           This bounds cross-encoder calls regardless of corpus size.
        b. Cross-encoder reranking: score each (claim, candidate_text) pair
           with a cross-encoder model.  Cross-encoders see both strings
           jointly so they handle paraphrase and domain vocabulary far
           better than cosine similarity alone.  Raw logit is sigmoid-ed
           to [0, 1] for a stable threshold.
        c. Keyword overlap boost: compute token-level Jaccard similarity
           between the claim and the candidate, scaled by KEYWORD_BOOST_WEIGHT
           (default 0.15), and add it to the cross-encoder score.  This
           recovers cases where the cross-encoder under-scores a strong match
           due to domain vocabulary it was not fine-tuned on.
        d. Citation diversity penalty: after a node is selected, its raw
           score is divided by (1 + DIVERSITY_ALPHA * times_already_cited)
           for all subsequent claims.  This prevents a single high-scoring
           hub chunk (e.g., an appendix summary or abstract page) from
           absorbing all citations in a long answer.
        e. Hub exclusion: if a node has been cited >= HUB_CITE_LIMIT times
           in the current answer, it is hard-excluded from further citation.
           This is a safety net for extreme hub collapse cases where the
           diversity penalty alone is insufficient.
        f. If final_score >= CITE_THRESHOLD: attach a hard-coded citation.
           Otherwise: leave the sentence uncited.
      Reassemble sentences (cited and uncited) into the final answer string.

Why this architecture
---------------------
- Fluency: the LLM writes prose normally; no oracle loop, no robotic lists.
- Citation accuracy: citations are assigned by embedding similarity to the
  full corpus index, not inferred from LLM output.
- KR: generation is unconstrained so the model can synthesise across chunks;
  KR is not penalised by sentence-selection decisions.
- Latency: one LLM call instead of N oracle calls.
- Uncited sentences are transparent: a claim with no citation above the
  threshold is flagged implicitly rather than silently cited incorrectly.

Tuning CITE_THRESHOLD
---------------------
- 0.50 is the default.  The final score is sigmoid(cross_encoder_logit) +
  keyword_jaccard * KEYWORD_BOOST_WEIGHT, so it is nominally in [0, ~1.15].
  In practice most strong matches land between 0.65 and 0.95.
- Raise CITE_THRESHOLD to tighten citation (fewer but more accurate).
- Lower it to increase citation rate at the cost of weaker matches.
- Raise KEYWORD_BOOST_WEIGHT if your corpus has dense domain vocabulary
  that the cross-encoder was not fine-tuned on (e.g. clinical, legal, code).
- DIVERSITY_ALPHA (default 0.5) controls how steeply repeated citations of
  the same node are penalised.  Raise for stricter hub suppression.
- HUB_CITE_LIMIT (default 3) is the hard exclusion ceiling per node per
  answer.  Lower it (e.g. 2) for short corpora where hub collapse is severe.
- Inspect result["citation_stats"] for score distribution and hub counts.

Citation format (written by program, never by LLM)
--------------------------------------------------
    [p.{page} | section-{section}]

Return structure
----------------
{
    "query"          : str,
    "answer"         : str,         final prose with inline citations
    "raw_answer"     : str,         LLM output before citation pass
    "source_pages"   : list[int],
    "cited_claims"   : list[dict],  one dict per sentence in the answer
    "citation_stats" : dict,        summary counts and similarity distribution
    "model_name"     : str,
    "backend"        : str,
}

Each cited_claim dict:
    {
        "text"          : str,   original claim sentence
        "cited_text"    : str,   sentence + citation (or sentence if uncited)
        "cited"         : bool,
        "score"         : float | None,   final combined score (ce + keyword boost)
        "ce_score"      : float | None,   sigmoid cross-encoder score alone
        "keyword_sim"   : float | None,   token Jaccard similarity
        "matched_node"  : str | None,     node_id of best match
        "matched_text"  : str | None,     text of best matching chunk
        "page"          : int | None,
        "section"       : str,
        "citation"      : str,            "" if uncited
    }
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_EMBED_MODEL_NAME = "all-mpnet-base-v2"   # must match step2 / step3

# Neighbourhood expansion
_SENT_WINDOW_SIM_THRESHOLD = 0.40

# MMR
_MMR_LAMBDA     = 0.47
_MMR_MAX_CHUNKS = 15

# Post-hoc citation
# CITE_THRESHOLD applies to the final combined score:
#   sigmoid(cross_encoder_logit) + keyword_jaccard * KEYWORD_BOOST_WEIGHT
# Strong matches typically score 0.65–0.95; weak ones < 0.45.
CITE_THRESHOLD = 0.50

# Cross-encoder model for claim-to-source reranking.
# ms-marco-MiniLM-L-6-v2 is fast (~6M params) and accurate for passage
# relevance.  Swap for a larger variant if latency allows:
#   cross-encoder/ms-marco-MiniLM-L-12-v2  (more accurate, ~2x slower)
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Number of candidates passed from cosine pre-selection to the cross-encoder.
# Higher = better recall at the cost of more cross-encoder calls per claim.
# 20 is a good default; raise to 30-40 for large corpora.
CE_CANDIDATES = 20

# Weight applied to token-level Jaccard keyword overlap before adding to the
# cross-encoder score.  Raise if your corpus has dense domain vocabulary
# the cross-encoder was not fine-tuned on (clinical, legal, code, etc.).
KEYWORD_BOOST_WEIGHT = 0.15

# Citation diversity penalty.
# After a node is cited, subsequent citations of the same node are penalised:
#   penalised_score = raw_score / (1 + DIVERSITY_ALPHA * times_already_cited)
# At alpha=0.5 a second citation scores at raw/1.5, a third at raw/2.0, etc.
# Raise alpha for stricter diversity; set to 0.0 to disable.
DIVERSITY_ALPHA = 0.5

# Hub chunk exclusion.
# If a single node accumulates >= HUB_CITE_LIMIT citations within one answer,
# it is permanently excluded from further citation in that answer.  This
# prevents appendix/boilerplate chunks from absorbing all citations in long
# answers (the "hub collapse" failure mode observed in document types F and I).
HUB_CITE_LIMIT = 3

# Context assembly
# Maximum characters of chunk text included per chunk in the context block.
# Prevents very long chunks from dominating the context window.
_MAX_CHUNK_CHARS = 800

# Ollama HTTP
_OLLAMA_DEFAULT_HOST   = "http://127.0.0.1:11434"
_OLLAMA_HTTP_TIMEOUT_S = 120

# llama_cpp
_DEFAULT_MODEL_WINDOW = 32768

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_embed_model      : SentenceTransformer | None = None
_cross_encoder                                 = None   # CrossEncoder instance
_llama_cpp_model                               = None


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(
            f"  [step5] Loading embed model ({_EMBED_MODEL_NAME}) on "
            f"{DEVICE.upper()}...",
            flush=True,
        )
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME, device=DEVICE)
    return _embed_model


def _get_cross_encoder(model_name: str = _CROSS_ENCODER_MODEL):
    """
    Lazy-load and cache a sentence_transformers CrossEncoder.

    The cross-encoder sees (claim, candidate_text) jointly, making it
    significantly more accurate than cosine similarity for paraphrase and
    domain-vocabulary mismatches.  It is only run on the CE_CANDIDATES
    pre-selected by cosine similarity, so per-query latency stays bounded.

    Requires: pip install sentence-transformers
    The CrossEncoder class is included in the same package as SentenceTransformer.
    """
    global _cross_encoder
    cached_name = getattr(_cross_encoder, "_model_name", None)
    if _cross_encoder is None or cached_name != model_name:
        from sentence_transformers import CrossEncoder  # type: ignore[attr-defined]
        print(
            f"  [step5] Loading cross-encoder ({model_name})...",
            flush=True,
        )
        _cross_encoder = CrossEncoder(model_name, device=DEVICE)
        _cross_encoder._model_name = model_name
    return _cross_encoder



    """
    Lazy-load and cache a llama_cpp.Llama instance.
    Re-initialises automatically if model_path changes.
    """
    global _llama_cpp_model
    cached_path = getattr(_llama_cpp_model, "_model_path", None)
    if _llama_cpp_model is None or cached_path != model_path:
        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed.  "
                "Run: pip install llama-cpp-python"
            ) from exc
        print(f"  [step5] Loading llama_cpp model: {model_path}", flush=True)
        _llama_cpp_model = Llama(
            model_path   = model_path,
            n_ctx        = n_ctx,
            n_gpu_layers = -1 if torch.cuda.is_available() else 0,
            verbose      = False,
        )
        _llama_cpp_model._model_path = model_path
    return _llama_cpp_model


def _to_tensor(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.float().cpu()
    return torch.tensor(arr, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Component 1 — Graph Neighbourhood Expansion  (UNCHANGED)
# ---------------------------------------------------------------------------

def _chunk_embedding_tensor(G, node_id: str) -> torch.Tensor | None:
    emb = G.nodes[node_id].get("embedding")
    return _to_tensor(emb) if emb is not None else None


def _get_chunk_for_sentence(G, sent_id: str) -> str | None:
    for _, tgt, edata in G.out_edges(sent_id, data=True):
        if edata.get("relation") == "PART_OF_CHUNK":
            return tgt
    return None


def _boundary_sentences(G, chunk_id: str) -> tuple[str | None, str | None]:
    sents = []
    for src, _, edata in G.in_edges(chunk_id, data=True):
        if edata.get("relation") == "PART_OF_CHUNK":
            sents.append((G.nodes[src].get("index", 0), src))
    if not sents:
        return None, None
    sents.sort()
    return sents[0][1], sents[-1][1]


def _expand_sentence_window(
    G,
    origin_chunk_id: str,
    origin_emb     : torch.Tensor,
    sim_threshold  : float = _SENT_WINDOW_SIM_THRESHOLD,
) -> set[str]:
    discovered: set[str] = set()
    first_sent, last_sent = _boundary_sentences(G, origin_chunk_id)
    for start_sent, direction in [
        (first_sent, "PREV_SENTENCE"),
        (last_sent,  "NEXT_SENTENCE"),
    ]:
        if start_sent is None:
            continue
        current_sent = start_sent
        while True:
            next_sent = None
            for _, tgt, edata in G.out_edges(current_sent, data=True):
                if edata.get("relation") == direction:
                    next_sent = tgt
                    break
            if next_sent is None:
                break
            neighbour_chunk = _get_chunk_for_sentence(G, next_sent)
            if neighbour_chunk is None or neighbour_chunk == origin_chunk_id:
                current_sent = next_sent
                continue
            n_emb = _chunk_embedding_tensor(G, neighbour_chunk)
            if n_emb is None:
                break
            if float(util.cos_sim(origin_emb, n_emb).item()) < sim_threshold:
                break
            discovered.add(neighbour_chunk)
            current_sent = next_sent
    return discovered


def _expand_sibling_chunks(G, chunk_id: str) -> set[str]:
    siblings: set[str] = set()
    section_id = None
    for _, tgt, edata in G.out_edges(chunk_id, data=True):
        if edata.get("relation") == "IN_SECTION":
            section_id = tgt
            break
    if section_id is None:
        return siblings
    for src, _, edata in G.in_edges(section_id, data=True):
        if (edata.get("relation") == "IN_SECTION"
                and G.nodes[src].get("type") == "SemanticChunk"
                and src != chunk_id):
            siblings.add(src)
    return siblings


def _expand_parent_section(G, chunk_id: str) -> set[str]:
    neighbours: set[str] = set()
    section_id = None
    for _, tgt, edata in G.out_edges(chunk_id, data=True):
        if edata.get("relation") == "IN_SECTION":
            section_id = tgt
            break
    if section_id is None:
        return neighbours
    parent_id = None
    for _, tgt, edata in G.out_edges(section_id, data=True):
        if edata.get("relation") == "PARENT_SECTION":
            parent_id = tgt
            break
    if parent_id is None:
        return neighbours
    for _, tgt, edata in G.out_edges(parent_id, data=True):
        if edata.get("relation") == "HAS_SUBSECTION" and tgt != section_id:
            for src, _, edata2 in G.in_edges(tgt, data=True):
                if (edata2.get("relation") == "IN_SECTION"
                        and G.nodes[src].get("type") == "SemanticChunk"):
                    neighbours.add(src)
    return neighbours


def expand_neighbourhood(
    G,
    top_k_results: list[dict],
    sim_threshold: float = _SENT_WINDOW_SIM_THRESHOLD,
) -> dict[str, set[str]]:
    neighbourhood: dict[str, set[str]] = {}
    for result in top_k_results:
        node_id   = result["node_id"]
        node_type = result["node_type"]
        if node_type != "SemanticChunk":
            neighbourhood[node_id] = set()
            continue
        origin_emb = _chunk_embedding_tensor(G, node_id)
        if origin_emb is None:
            neighbourhood[node_id] = set()
            continue
        neighbourhood[node_id] = (
            _expand_sentence_window(G, node_id, origin_emb, sim_threshold)
            | _expand_sibling_chunks(G, node_id)
            | _expand_parent_section(G, node_id)
        )
    return neighbourhood


# ---------------------------------------------------------------------------
# Component 2 — MMR Selection  (UNCHANGED)
# ---------------------------------------------------------------------------

def _get_embedding(G, node_id: str) -> torch.Tensor | None:
    emb = G.nodes[node_id].get("embedding")
    return _to_tensor(emb) if emb is not None else None


def mmr_select(
    G,
    top_k_results  : list[dict],
    neighbourhood  : dict[str, set[str]],
    query_embedding: np.ndarray,
    lambda_mmr     : float = _MMR_LAMBDA,
    budget_n       : int   = _MMR_MAX_CHUNKS,
) -> list[dict]:
    """
    Maximal Marginal Relevance over (top-k chunks union neighbourhood chunks).
    Returns chunk dicts in MMR selection order.
    """
    q_emb = _to_tensor(query_embedding).unsqueeze(0)

    top_k_sims: dict[str, float] = {}
    for r in top_k_results:
        emb = _get_embedding(G, r["node_id"])
        if emb is not None:
            top_k_sims[r["node_id"]] = float(
                util.cos_sim(q_emb, emb.unsqueeze(0)).item()
            )

    sim_min   = min(top_k_sims.values()) if top_k_sims else 0.0
    sim_max   = max(top_k_sims.values()) if top_k_sims else 1.0
    sim_range = sim_max - sim_min if sim_max > sim_min else 1.0

    candidates: dict[str, dict] = {}

    for r in top_k_results:
        nid = r["node_id"]
        emb = _get_embedding(G, nid)
        if emb is None:
            continue
        candidates[nid] = {
            "node_id"          : nid,
            "node_type"        : r["node_type"],
            "query_sim"        : top_k_sims.get(nid, 0.0),
            "normalized_score" : r["normalized_score"],
            "is_top_k"         : True,
            "text"             : G.nodes[nid].get("text", ""),
            "page"             : G.nodes[nid].get("page"),
            "section"          : r["section"],
            "embedding"        : emb,
        }

    for origin_id, nbrs in neighbourhood.items():
        for nid in nbrs:
            if nid in candidates:
                continue
            emb = _get_embedding(G, nid)
            if emb is None:
                continue
            sim      = float(util.cos_sim(q_emb, emb.unsqueeze(0)).item())
            nbr_norm = max(0.0, min((sim - sim_min) / sim_range * 0.9, 1.0))
            section_heading = ""
            for _, tgt, edata in G.out_edges(nid, data=True):
                if edata.get("relation") == "IN_SECTION":
                    section_heading = G.nodes[tgt].get("heading", "")
                    break
            nd = G.nodes[nid]
            candidates[nid] = {
                "node_id"          : nid,
                "node_type"        : nd.get("type", "SemanticChunk"),
                "query_sim"        : sim,
                "normalized_score" : round(nbr_norm, 4),
                "is_top_k"         : False,
                "text"             : nd.get("text", ""),
                "page"             : nd.get("page"),
                "section"          : section_heading,
                "embedding"        : emb,
            }

    if not candidates:
        return []

    cand_ids : list[str]          = list(candidates.keys())
    selected : list[str]          = []
    sel_embs : list[torch.Tensor] = []

    for _ in range(min(budget_n, len(cand_ids))):
        if not cand_ids:
            break
        best_id    = None
        best_score = -float("inf")
        for nid in cand_ids:
            c   = candidates[nid]
            rel = lambda_mmr * c["query_sim"]
            if sel_embs:
                c_emb   = c["embedding"].unsqueeze(0)
                sel_mat = torch.stack(sel_embs)
                max_sim = float(util.cos_sim(c_emb, sel_mat).max().item())
                div     = (1.0 - lambda_mmr) * max_sim
            else:
                div = 0.0
            score = rel - div
            if score > best_score:
                best_score = score
                best_id    = nid
        if best_id is None:
            break
        selected.append(best_id)
        sel_embs.append(candidates[best_id]["embedding"])
        cand_ids.remove(best_id)

    result = []
    for nid in selected:
        c = dict(candidates[nid])
        c.pop("embedding", None)
        result.append(c)
    return result


# ---------------------------------------------------------------------------
# Component 3 — Context Assembly + Single LLM Call
# ---------------------------------------------------------------------------

def _assemble_context(mmr_selected: list[dict]) -> str:
    """
    Build a readable context block from MMR-selected chunks, grouped by section.

    Chunks from the same section are placed together under a section header.
    Each chunk is capped at _MAX_CHUNK_CHARS to prevent any single chunk from
    dominating the context window.
    """
    # Group by section, preserving MMR relevance order within each group
    sections: dict[str, list[dict]] = {}
    section_order: list[str] = []
    for chunk in mmr_selected:
        sec = chunk.get("section", "").strip() or "General"
        if sec not in sections:
            sections[sec] = []
            section_order.append(sec)
        sections[sec].append(chunk)

    blocks: list[str] = []
    for sec in section_order:
        blocks.append(f"[{sec}]")
        for chunk in sections[sec]:
            text = chunk.get("text", "").strip()
            if len(text) > _MAX_CHUNK_CHARS:
                text = text[:_MAX_CHUNK_CHARS] + "..."
            page = chunk.get("page")
            label = f"(p.{page})" if page is not None else ""
            blocks.append(f"{text} {label}".strip())
        blocks.append("")   # blank line between sections

    return "\n".join(blocks).strip()


def _build_generation_prompt(query: str, context: str) -> str:
    return (
        f"Answer the following query using only the provided context. "
        f"Write in clear, fluent prose. Never add information not present. Never make up information. Never use any of your own knowledge. Never use any general knowledge. Never use any information not present in the context. "
        f"Never include citation markers.\n\n"
        f"Phrases like based upon general knowledge are not allowed. Phrases like based upon the context are allowed. Phrases like based upon the information in the context are allowed. Phrases like based upon the information outside the context are not allowed. "
        f"Context:\n{context}\n\n"
        f"Query: {query}\n\n"
        f"Answer:"
    )


def _call_llm_ollama(
    model_name: str,
    prompt    : str,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    url  = f"{os.environ.get('OLLAMA_HOST', _OLLAMA_DEFAULT_HOST).rstrip('/')}/api/generate"
    body = {
        "model"  : model_name,
        "prompt" : prompt,
        "stream" : False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        url,
        data    = json.dumps(body).encode("utf-8"),
        method  = "POST",
        headers = {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_OLLAMA_HTTP_TIMEOUT_S) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise RuntimeError(f"Ollama generation error: {msg}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {url}. "
            f"Ensure `ollama serve` is running. ({e!s})"
        ) from e
    return raw.get("response", "").strip()


def _call_llm_llama_cpp(
    model_path : str,
    prompt     : str,
    max_tokens : int   = 512,
    temperature: float = 0.1,
    n_ctx      : int   = _DEFAULT_MODEL_WINDOW,
) -> str:
    llm = _get_llama_cpp_model(model_path, n_ctx)
    out = llm(prompt, max_tokens=max_tokens, temperature=temperature, echo=False)
    return out["choices"][0]["text"].strip()


def generate_response(
    query       : str,
    mmr_selected: list[dict],
    backend     : str,
    model_name  : str,
    n_ctx       : int   = _DEFAULT_MODEL_WINDOW,
    max_tokens  : int   = 512,
    temperature : float = 0.1,
) -> str:
    """
    Assemble context from MMR-selected chunks and make a single LLM call.

    Returns the raw generated answer string with no citation markers.
    Fluency is entirely the LLM's responsibility here; citation is handled
    separately in Component 5.
    """
    context = _assemble_context(mmr_selected)
    prompt  = _build_generation_prompt(query, context)

    if backend == "ollama":
        return _call_llm_ollama(model_name, prompt, max_tokens, temperature)
    elif backend == "llama_cpp":
        return _call_llm_llama_cpp(
            model_name, prompt, max_tokens, temperature, n_ctx,
        )
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Use 'ollama' or 'llama_cpp'."
        )


# ---------------------------------------------------------------------------
# Component 4 — Claim Decomposition
# ---------------------------------------------------------------------------

# Sentence boundary: stdlib ``re`` only supports fixed-width lookbehinds.
# Split on sentence punctuation + whitespace + likely sentence starter, then
# merge splits that occurred at common abbreviations (Mr., Dr., Fig., etc.).
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\(\[])')
_ABBREV_TAIL = re.compile(
    r'\b(?:Mr|Mrs|Ms|Mx|Dr|Prof|St|vs|Fig|etc|approx|Eq|Vol|No|pp|al|et)\.?\s*$',
    re.IGNORECASE,
)


def split_into_claims(text: str) -> list[str]:
    """
    Split a generated answer into individual claim sentences.

    Returns a list of non-empty stripped strings.  Preserves the original
    text — no words are added or removed, only boundaries are identified.
    """
    text = text.strip()
    if not text:
        return []

    parts = _SENT_BOUNDARY.split(text)
    if len(parts) <= 1:
        return [s.strip() for s in parts if s.strip()]

    merged: list[str] = [parts[0]]
    for p in parts[1:]:
        prev = merged[-1]
        if _ABBREV_TAIL.search(prev) and (p[:1].isupper() or p[:1].isdigit()):
            merged[-1] = prev.rstrip() + " " + p
        else:
            merged.append(p)

    return [s.strip() for s in merged if s.strip()]


# ---------------------------------------------------------------------------
# Component 5 — Per-Claim Retrieval & Citation Injection
# ---------------------------------------------------------------------------

# Stopwords excluded from keyword overlap to avoid boosting on function words.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "that", "this", "these", "those",
    "it", "its", "as", "not", "no", "so", "if", "than", "then", "also",
    "which", "who", "what", "when", "where", "how", "there", "their",
    "they", "we", "i", "you", "he", "she", "his", "her", "our", "your",
})

_TOKEN_RE = re.compile(r"[\w]+(?:-[\w]+)*")


def _keyword_jaccard(claim: str, candidate: str) -> float:
    """
    Token-level Jaccard similarity between claim and candidate, after
    lowercasing and stopword removal.

    Returns a float in [0, 1].  A score of 0.0 means no shared content
    words; 1.0 means identical token sets (rare in practice).

    This complements the cross-encoder: CE handles semantic similarity,
    Jaccard handles exact term overlap for domain vocabulary (abbreviations,
    model names, chemical compounds, etc.) that the CE may not have seen
    during fine-tuning.
    """
    def _tokens(text: str) -> frozenset[str]:
        return frozenset(
            t for t in _TOKEN_RE.findall(text.lower())
            if t not in _STOPWORDS and len(t) > 1
        )

    claim_toks = _tokens(claim)
    cand_toks  = _tokens(candidate)
    if not claim_toks and not cand_toks:
        return 0.0
    intersection = len(claim_toks & cand_toks)
    union        = len(claim_toks | cand_toks)
    return intersection / union if union else 0.0


def _format_citation(page: int | None, section: str, node_id: str) -> str:
    """
    Build a provenance citation string from KG metadata.
    Written by the program; never generated by the LLM.

    Format: [p. {page}]
    """
    if page is not None:
        return f"[p. {page}]"
    return f"[node:{node_id}]"


def _get_node_provenance(G, node_id: str) -> tuple[int | None, str]:
    """Return (page, section_heading) for a graph node."""
    nd      = G.nodes[node_id]
    page    = nd.get("page")
    section = ""
    for _, tgt, edata in G.out_edges(node_id, data=True):
        if edata.get("relation") == "IN_SECTION":
            section = G.nodes[tgt].get("heading", "")
            break
    return page, section


def cite_claims(
    claims          : list[str],
    G,
    index           : dict,
    threshold       : float = CITE_THRESHOLD,
    ce_candidates   : int   = CE_CANDIDATES,
    keyword_weight  : float = KEYWORD_BOOST_WEIGHT,
    ce_model_name   : str   = _CROSS_ENCODER_MODEL,
    diversity_alpha : float = DIVERSITY_ALPHA,
    hub_cite_limit  : int   = HUB_CITE_LIMIT,
) -> list[dict]:
    """
    For each claim sentence, find the best-matching node in the full retrieval
    index using a four-stage pipeline and attach a hard-coded citation if
    the final score >= threshold.

    Pipeline per claim
    ------------------
    1. Cosine pre-selection: dot-product the claim embedding against the full
       index to get the top `ce_candidates` candidates.  Cheap single matrix
       multiply for all claims at once; bounds cross-encoder calls.

    2. Cross-encoder reranking: score each (claim, candidate_text) pair with
       a CrossEncoder.  The CE sees both strings jointly so it handles
       paraphrase and vocabulary mismatch far better than cosine.  Raw logit
       is sigmoid-ed to [0, 1] for a stable threshold.

    3. Keyword boost: token-level Jaccard similarity (stopwords removed),
       multiplied by `keyword_weight`, added to the CE score.

    4. Citation diversity penalty + hub exclusion (stateful across claims):
       - Any node already cited N times has its raw score divided by
         (1 + diversity_alpha * N) before the threshold check.
       - Any node cited >= hub_cite_limit times is hard-excluded (score
         forced to -1) to prevent appendix/abstract hub collapse.
       - citation_counts is updated after each successful citation.

    Parameters
    ----------
    claims          : output of split_into_claims()
    G               : knowledge graph
    index           : dict from step3/step4 with keys: ids, embeddings, texts
    threshold       : minimum final score to attach a citation
    ce_candidates   : how many cosine-pre-selected candidates to cross-encode
    keyword_weight  : Jaccard overlap multiplier added to CE score
    ce_model_name   : HuggingFace model id for the CrossEncoder
    diversity_alpha : repeated-citation penalty weight (0.0 = disabled)
    hub_cite_limit  : hard exclusion ceiling; node excluded after this many
                      citations in the current answer

    Returns
    -------
    List of cited_claim dicts, one per claim, in original order.
    """
    if not claims:
        return []

    index_embeddings = index.get("embeddings")
    index_ids        = index.get("ids", [])
    index_texts      = index.get("texts", [])

    _uncited = lambda c: {   # noqa: E731
        "text"        : c,
        "cited_text"  : c,
        "cited"       : False,
        "score"       : None,
        "ce_score"    : None,
        "keyword_sim" : None,
        "matched_node": None,
        "matched_text": None,
        "page"        : None,
        "section"     : "",
        "citation"    : "",
    }

    if index_embeddings is None or len(index_ids) == 0:
        return [_uncited(c) for c in claims]

    # ------------------------------------------------------------------
    # Stage 1: cosine pre-selection (all claims in one matrix multiply)
    # ------------------------------------------------------------------
    embed_model = _get_embed_model()
    claim_embs = _to_tensor(
        embed_model.encode(claims, show_progress_bar=False, device=DEVICE)
    ).to(DEVICE)
    if claim_embs.dim() == 1:
        # Single-claim edge case: enforce (1 x d) for matrix math.
        claim_embs = claim_embs.unsqueeze(0)

    idx_embs = _to_tensor(index_embeddings).to(DEVICE)
    if idx_embs.dim() == 1:
        # Single-index-node edge case: enforce (1 x d) for matrix math.
        idx_embs = idx_embs.unsqueeze(0)
    idx_norm = torch.nn.functional.normalize(idx_embs, dim=1)
    clm_norm = torch.nn.functional.normalize(claim_embs, dim=1)

    # sim_matrix: (n_claims x n_index_nodes)
    sim_matrix = clm_norm @ idx_norm.T

    n_candidates = min(ce_candidates, len(index_ids))

    # top-k indices per claim: shape (n_claims x n_candidates)
    topk_indices = sim_matrix.topk(n_candidates, dim=1).indices.cpu().tolist()

    # ------------------------------------------------------------------
    # Stage 2+3: cross-encoder reranking + keyword boost
    # ------------------------------------------------------------------
    ce = _get_cross_encoder(ce_model_name)

    # ------------------------------------------------------------------
    # Stage 4: diversity state — tracks citation counts per node_id
    # across all claims in this answer.  Updated after each citation.
    # ------------------------------------------------------------------
    citation_counts: dict[str, int] = {}

    results: list[dict] = []

    for i, claim in enumerate(claims):
        cand_indices = topk_indices[i]
        cand_texts   = [
            index_texts[j] if j < len(index_texts) else ""
            for j in cand_indices
        ]

        # Stage 2: Cross-encoder on (claim, candidate) pairs
        pairs     = [(claim, ct) for ct in cand_texts]
        ce_logits = ce.predict(pairs, show_progress_bar=False)
        ce_scores = torch.sigmoid(
            torch.tensor(ce_logits, dtype=torch.float32)
        ).tolist()

        # Stage 3: Keyword Jaccard boost per candidate
        kw_sims = [_keyword_jaccard(claim, ct) for ct in cand_texts]

        # Raw combined score before diversity adjustment
        raw_combined = [
            ce_scores[k] + kw_sims[k] * keyword_weight
            for k in range(len(cand_indices))
        ]

        # Stage 4: apply diversity penalty and hub exclusion
        adjusted = []
        for k, raw_score in enumerate(raw_combined):
            node_id  = index_ids[cand_indices[k]]
            n_cited  = citation_counts.get(node_id, 0)

            if n_cited >= hub_cite_limit:
                # Hard exclusion — this node has been cited too many times
                adjusted.append(-1.0)
            elif n_cited > 0 and diversity_alpha > 0.0:
                # Soft penalty — score degrades with repeated use
                adjusted.append(raw_score / (1.0 + diversity_alpha * n_cited))
            else:
                adjusted.append(raw_score)

        best_local = int(np.argmax(adjusted))
        best_score = adjusted[best_local]
        best_ce    = ce_scores[best_local]
        best_kw    = kw_sims[best_local]
        best_idx   = cand_indices[best_local]

        if best_score >= threshold:
            node_id      = index_ids[best_idx]
            matched_text = cand_texts[best_local]
            page, section = (
                _get_node_provenance(G, node_id)
                if G.has_node(node_id)
                else (None, "")
            )
            citation   = _format_citation(page, section, node_id)
            cited_text = f"{claim} {citation}"

            # Update diversity state
            citation_counts[node_id] = citation_counts.get(node_id, 0) + 1

            results.append({
                "text"        : claim,
                "cited_text"  : cited_text,
                "cited"       : True,
                "score"       : round(best_score, 4),
                "ce_score"    : round(best_ce, 4),
                "keyword_sim" : round(best_kw, 4),
                "matched_node": node_id,
                "matched_text": matched_text,
                "page"        : page,
                "section"     : section,
                "citation"    : citation,
            })
        else:
            r = _uncited(claim)
            r["score"]       = round(best_score, 4)
            r["ce_score"]    = round(best_ce, 4)
            r["keyword_sim"] = round(best_kw, 4)
            results.append(r)

    return results


def _citation_stats(cited_claims: list[dict]) -> dict:
    """Compute summary statistics over the citation pass results."""
    total   = len(cited_claims)
    n_cited = sum(1 for c in cited_claims if c["cited"])
    scores  = [c["score"] for c in cited_claims if c.get("score") is not None]
    ce_scores = [c["ce_score"] for c in cited_claims if c.get("ce_score") is not None]
    kw_sims   = [c["keyword_sim"] for c in cited_claims if c.get("keyword_sim") is not None]
    return {
        "total_claims"   : total,
        "cited"          : n_cited,
        "uncited"        : total - n_cited,
        "cite_rate"      : round(n_cited / total, 4) if total else 0.0,
        "mean_score"     : round(float(np.mean(scores)), 4) if scores else None,
        "min_score"      : round(float(np.min(scores)), 4) if scores else None,
        "max_score"      : round(float(np.max(scores)), 4) if scores else None,
        "mean_ce_score"  : round(float(np.mean(ce_scores)), 4) if ce_scores else None,
        "mean_keyword_sim": round(float(np.mean(kw_sims)), 4) if kw_sims else None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    G,
    index        : dict,
    top_k_results: list[dict],
    *,
    backend       : str   = "ollama",
    model_name    : str   = "qwen2.5:3b",
    n_ctx         : int   = _DEFAULT_MODEL_WINDOW,
    lambda_mmr    : float = _MMR_LAMBDA,
    budget_n      : int   = _MMR_MAX_CHUNKS,
    sim_threshold : float = _SENT_WINDOW_SIM_THRESHOLD,
    cite_threshold: float = CITE_THRESHOLD,
    ce_candidates : int   = CE_CANDIDATES,
    keyword_weight: float = KEYWORD_BOOST_WEIGHT,
    ce_model_name : str   = _CROSS_ENCODER_MODEL,
    max_tokens    : int   = 512,
    temperature   : float = 0.1,
    timing_out    : dict | None = None,
) -> dict:
    """
    Generate a fluent answer then attach citations via post-hoc retrieval.

    Parameters
    ----------
    G             : nx.MultiDiGraph from step2
    index         : retrieval index dict from step3/step4
    top_k_results : list of result dicts from step3.query()
    backend       : "ollama" | "llama_cpp"
    model_name    : ollama model tag or absolute .gguf path
    n_ctx         : llama_cpp context window (ignored for ollama)
    lambda_mmr    : MMR relevance/diversity trade-off (0-1)
    budget_n      : max chunks selected by MMR
    sim_threshold : cosine sim floor for neighbourhood expansion
    cite_threshold: minimum combined score (CE + keyword boost) to cite
    ce_candidates : top-N cosine candidates passed to the cross-encoder
    keyword_weight: Jaccard overlap multiplier added to CE score
    ce_model_name : HuggingFace model id for the CrossEncoder
    max_tokens    : maximum tokens for the LLM generation call
    temperature   : generation temperature (low = more deterministic)
    timing_out    : if provided, filled with per-phase wall-clock seconds:
                    {neighbourhood_s, mmr_s, generation_s, citation_s, total_s}

    Returns
    -------
    {
        "query"          : str,
        "answer"         : str,
        "raw_answer"     : str,
        "source_pages"   : list[int],
        "cited_claims"   : list[dict],
        "citation_stats" : dict,
        "model_name"     : str,
        "backend"        : str,
    }
    """
    if not top_k_results:
        raise ValueError("top_k_results is empty -- run step3.query() first.")

    query_embedding = top_k_results[0].get("query_embedding")
    if query_embedding is None:
        raise ValueError(
            "query_embedding missing from top_k_results. "
            "Ensure step3.query() attaches query_embedding to each result dict."
        )

    query_text = top_k_results[0].get("_query_text", "")
    if not query_text:
        query_text = top_k_results[0].get("chunk_text", "")[:120]

    t_wall0 = time.perf_counter()

    # -- Phase 1: Neighbourhood expansion -----------------------------------
    print("  [step5] Expanding graph neighbourhoods...", flush=True)
    t0 = time.perf_counter()
    neighbourhood = expand_neighbourhood(G, top_k_results, sim_threshold)
    nh_s = time.perf_counter() - t0

    # -- Phase 2: MMR -------------------------------------------------------
    print("  [step5] Running MMR selection...", flush=True)
    t0 = time.perf_counter()
    mmr_selected = mmr_select(
        G, top_k_results, neighbourhood, query_embedding,
        lambda_mmr=lambda_mmr, budget_n=budget_n,
    )
    mmr_s = time.perf_counter() - t0
    print(f"  [step5] {len(mmr_selected)} chunks selected by MMR.", flush=True)

    if not mmr_selected:
        return {
            "query"         : query_text,
            "answer"        : "No relevant content found in the document.",
            "raw_answer"    : "",
            "source_pages"  : [],
            "cited_claims"  : [],
            "citation_stats": _citation_stats([]),
            "model_name"    : model_name,
            "backend"       : backend,
        }

    # -- Phase 3: Single LLM generation call --------------------------------
    print(
        f"  [step5] Generating response "
        f"(backend={backend}, model={model_name})...",
        flush=True,
    )
    t0 = time.perf_counter()
    raw_answer = generate_response(
        query_text, mmr_selected, backend, model_name, n_ctx,
        max_tokens, temperature,
    )
    gen_s = time.perf_counter() - t0
    print(f"  [step5] Generated {len(raw_answer.split())} words in {gen_s:.2f}s.", flush=True)

    # -- Phase 4: Post-hoc citation -----------------------------------------
    print(
        f"  [step5] Running citation pass "
        f"(threshold={cite_threshold})...",
        flush=True,
    )
    t0          = time.perf_counter()
    claims      = split_into_claims(raw_answer)
    cited       = cite_claims(
        claims,
        G,
        index,
        threshold      = cite_threshold,
        ce_candidates  = ce_candidates,
        keyword_weight = keyword_weight,
        ce_model_name  = ce_model_name,
    )
    cite_s      = time.perf_counter() - t0

    stats = _citation_stats(cited)
    print(
        f"  [step5] {stats['cited']}/{stats['total_claims']} claims cited "
        f"(rate={stats['cite_rate']:.0%}, mean_score={stats['mean_score']}) "
        f"in {cite_s:.2f}s.",
        flush=True,
    )

    if timing_out is not None:
        timing_out.clear()
        timing_out.update({
            "neighbourhood_s": round(nh_s, 3),
            "mmr_s"          : round(mmr_s, 3),
            "generation_s"   : round(gen_s, 3),
            "citation_s"     : round(cite_s, 3),
            "total_s"        : round(time.perf_counter() - t_wall0, 3),
        })

    answer       = " ".join(c["cited_text"] for c in cited)
    source_pages = sorted({
        c["page"] for c in cited if c.get("page") is not None
    })

    return {
        "query"         : query_text,
        "answer"        : answer,
        "raw_answer"    : raw_answer,
        "source_pages"  : source_pages,
        "cited_claims"  : cited,
        "citation_stats": stats,
        "model_name"    : model_name,
        "backend"       : backend,
    }


def query_and_generate(
    G,
    index        : dict,
    query_text   : str,
    *,
    top_k        : int   = 8,
    backend      : str   = "ollama",
    model_name   : str   = "qwen2.5:3b",
    n_ctx        : int   = _DEFAULT_MODEL_WINDOW,
    lambda_mmr   : float = _MMR_LAMBDA,
    budget_n     : int   = _MMR_MAX_CHUNKS,
    sim_threshold: float = _SENT_WINDOW_SIM_THRESHOLD,
    cite_threshold: float = CITE_THRESHOLD,
    ce_candidates : int   = CE_CANDIDATES,
    keyword_weight: float = KEYWORD_BOOST_WEIGHT,
    ce_model_name : str   = _CROSS_ENCODER_MODEL,
    max_tokens    : int   = 512,
    temperature  : float = 0.1,
    **_ignored_kwargs: Any,
) -> dict:
    """
    Convenience wrapper: runs step3 retrieval then generate() in one call.

    Parameters match generate() with the addition of top_k for retrieval.
    Legacy kwargs (context_budget, max_iters, etc.) are silently ignored.
    """
    from step3_retrieval import query as _retrieval_query  # type: ignore[import]

    results = _retrieval_query(G, index, query_text, top_k=top_k)
    if not results:
        return {
            "query"         : query_text,
            "answer"        : "No relevant content found in the document.",
            "raw_answer"    : "",
            "source_pages"  : [],
            "cited_claims"  : [],
            "citation_stats": _citation_stats([]),
            "model_name"    : model_name,
            "backend"       : backend,
        }

    for r in results:
        r["_query_text"] = query_text

    return generate(
        G,
        index,
        results,
        backend       = backend,
        model_name    = model_name,
        n_ctx         = n_ctx,
        lambda_mmr    = lambda_mmr,
        budget_n      = budget_n,
        sim_threshold = sim_threshold,
        cite_threshold= cite_threshold,
        ce_candidates = ce_candidates,
        keyword_weight= keyword_weight,
        ce_model_name = ce_model_name,
        max_tokens    = max_tokens,
        temperature   = temperature,
    )


def format_generation(result: dict) -> str:
    """Format a generate() result dict as a human-readable string."""
    stats = result.get("citation_stats", {})
    lines = [
        f'\nQuery     : "{result["query"]}"',
        f'Model     : {result["model_name"]} [{result["backend"]}]',
        f'Pages     : {result["source_pages"]}',
        f'Citations : {stats.get("cited", "?")}/'
        f'{stats.get("total_claims", "?")} claims '
        f'({stats.get("cite_rate", 0):.0%})',
        "-" * 60,
        "",
        result["answer"],
        "",
    ]
    return "\n".join(lines)