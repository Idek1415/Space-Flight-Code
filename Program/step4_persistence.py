"""
Step 4 — Persistence
====================
Saves, loads, and manages the knowledge graph and retrieval index so a
PDF does not need to be re-processed on every run.

Saved bundle layout
-------------------
    <save_root>/<pdf_stem>_kg/
        graph.pkl        -- NetworkX MultiDiGraph (pickle)
                           Sentence node embeddings are stripped before
                           saving (they are never indexed); all other node
                           attributes, including 'embedding' numpy arrays
                           on SemanticChunk / Table / Section nodes, are kept.
        embeddings.pt    -- float16 tensor (n x d), mean-centred
        corpus_mean.pt   -- float32 tensor (1 x d)
        bm25.pkl         -- serialised BM25Okapi index (optional)
        index_meta.json  -- node IDs and types (texts reconstructed from graph)
        metadata.json    -- pdf hash, timestamps, model names, node counts

Cache invalidation
------------------
The PDF is identified by a SHA-256 hash (first 16 hex chars).  If the
file changes, load_kg() detects the mismatch and returns None, triggering
a full rebuild in the calling code.

Storage optimisations
---------------------
- Embeddings stored as float16 (~50% size vs float32); upcast to float32
  on load for compute compatibility.
- Sentence nodes carry no embeddings, so nothing is stripped there.
- BM25 index is pickled alongside the graph; if the pickle is absent
  on load (e.g. legacy save), the index is rebuilt from stored texts.
- index_meta.json stores only IDs and types; texts are reconstructed
  from the graph on load to avoid duplication.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import torch

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25Okapi      = None   # type: ignore[assignment,misc]
    _BM25_AVAILABLE = False

import re


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bundle_dir(pdf_path: str | Path, save_root: Path | None = None) -> Path:
    """Return the KG bundle directory for a given PDF."""
    p = Path(pdf_path).resolve()
    root = Path(save_root).resolve() if save_root else p.parent
    return root / f"{p.stem}_kg"


def _pdf_hash(pdf_path: str | Path) -> str:
    """SHA-256 hash of the PDF file, truncated to 16 hex chars."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def _tokenize(text: str) -> list[str]:
    """Tokenizer matching step2 and step3 exactly."""
    return re.findall(r"[\w]+(?:-[\w]+)*", text.lower())


def _strip_sentence_embeddings(G):
    """
    Return a deep copy of G with the 'embedding' attribute removed from
    Sentence nodes.  Sentence nodes are never indexed, so their embeddings
    (if any were accidentally stored) waste disk space without benefit.
    """
    pruned = copy.deepcopy(G)
    for node_id, data in pruned.nodes(data=True):
        if data.get("type") == "Sentence":
            data.pop("embedding", None)
    return pruned


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_kg(
    pdf_path  : str | Path,
    G,
    index     : dict,
    *,
    model_name: str = "",
    save_root : Path | None = None,
) -> Path:
    """
    Persist the knowledge graph and retrieval index to disk.

    Parameters
    ----------
    pdf_path   : path to the source PDF (used for naming and hashing)
    G          : nx.MultiDiGraph from step2.build_knowledge_graph()
    index      : dict from step2.build_index()
    model_name : name of the embedding model (stored in metadata)
    save_root  : override directory; defaults to the folder containing pdf_path

    Returns
    -------
    Path to the bundle directory.
    """
    save_dir = _bundle_dir(pdf_path, save_root)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -- Graph -----------------------------------------------------------
    pruned_G = _strip_sentence_embeddings(G)
    with open(save_dir / "graph.pkl", "wb") as f:
        pickle.dump(pruned_G, f, protocol=pickle.HIGHEST_PROTOCOL)

    # -- Embeddings (float16 for ~50% size saving) -----------------------
    emb = index.get("embeddings")
    if emb is not None:
        torch.save(emb.detach().half().cpu(), save_dir / "embeddings.pt")

    cm = index.get("corpus_mean")
    if cm is not None:
        torch.save(cm.detach().float().cpu(), save_dir / "corpus_mean.pt")
    else:
        (save_dir / "corpus_mean.pt").unlink(missing_ok=True)

    # -- BM25 ------------------------------------------------------------
    bm25     = index.get("bm25")
    bm25_path = save_dir / "bm25.pkl"
    if bm25 is not None:
        try:
            with open(bm25_path, "wb") as f:
                pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"  Warning: could not save BM25 index: {e}", flush=True)
    else:
        bm25_path.unlink(missing_ok=True)

    # -- Index metadata (texts reconstructed from graph on load) ---------
    index_meta = {
        "ids"  : index.get("ids",   []),
        "types": index.get("types", []),
    }
    with open(save_dir / "index_meta.json", "w") as f:
        json.dump(index_meta, f)

    # -- Bundle metadata -------------------------------------------------
    p    = Path(pdf_path).resolve()
    meta = {
        "pdf_path"   : str(p),
        "pdf_stem"   : p.stem,
        "pdf_hash"   : _pdf_hash(pdf_path),
        "saved_at"   : datetime.now().isoformat(),
        "model_name" : model_name,
        "node_count" : G.number_of_nodes(),
        "edge_count" : G.number_of_edges(),
        "index_count": len(index.get("ids", [])),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    n_chunk = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "SemanticChunk")
    n_table = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "Table")
    print(
        f"  KG saved -> {save_dir}\n"
        f"  ({G.number_of_nodes()} nodes, {n_chunk} chunks, {n_table} tables, "
        f"{len(index.get('ids', []))} indexed)",
        flush=True,
    )
    return save_dir


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_kg(
    pdf_path : str | Path,
    *,
    save_root: Path | None = None,
) -> tuple | None:
    """
    Load a previously saved graph and index.

    Parameters
    ----------
    pdf_path  : path to the source PDF
    save_root : override directory (must match the save_root used in save_kg)

    Returns
    -------
    (G, index, metadata)  if a valid, hash-matched save is found
    None                  otherwise (caller should rebuild)
    """
    save_dir  = _bundle_dir(pdf_path, save_root)
    meta_path = save_dir / "metadata.json"

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # Hash check — invalidate if the source PDF has changed
    try:
        current_hash = _pdf_hash(pdf_path)
    except OSError:
        return None

    if meta.get("pdf_hash") != current_hash:
        print("  Saved KG found but PDF has changed -- will rebuild.", flush=True)
        return None

    graph_path = save_dir / "graph.pkl"
    index_path = save_dir / "index_meta.json"

    if not graph_path.exists() or not index_path.exists():
        print("  Saved KG metadata found but files are missing -- will rebuild.",
              flush=True)
        return None

    print(
        f"  Loading saved KG  (saved {meta.get('saved_at', '?')}, "
        f"model: {meta.get('model_name', '?')})...",
        flush=True,
    )

    # Graph
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # Index metadata
    with open(index_path) as f:
        index_data = json.load(f)

    ids   = index_data.get("ids",   [])
    types = index_data.get("types", [])

    # Reconstruct texts from graph nodes (avoids duplication on disk)
    texts = [
        G.nodes[nid].get("text", "") if G.has_node(nid) else ""
        for nid in ids
    ]

    # Embeddings: float16 on disk -> float32 for compute
    emb_path = save_dir / "embeddings.pt"
    embeddings = None
    if emb_path.exists():
        embeddings = torch.load(emb_path, map_location="cpu")
        if embeddings is not None and embeddings.dtype == torch.float16:
            embeddings = embeddings.float()

    # Corpus mean
    cm_path = save_dir / "corpus_mean.pt"
    corpus_mean = None
    if cm_path.exists():
        corpus_mean = torch.load(cm_path, map_location="cpu")
        if corpus_mean is not None and corpus_mean.dtype == torch.float16:
            corpus_mean = corpus_mean.float()

    # BM25: load from pickle, or rebuild from texts if pickle is absent
    bm25      = None
    bm25_path = save_dir / "bm25.pkl"
    if bm25_path.exists():
        try:
            with open(bm25_path, "rb") as f:
                bm25 = pickle.load(f)
        except Exception:
            bm25 = None

    if bm25 is None and texts and _BM25_AVAILABLE:
        print("  BM25 pickle absent -- rebuilding from texts...", flush=True)
        tokenized = [_tokenize(t) for t in texts]
        try:
            bm25 = _BM25Okapi(tokenized)
        except Exception:
            bm25 = None

    index = {
        "ids"        : ids,
        "types"      : types,
        "texts"      : texts,
        "embeddings" : embeddings,
        "corpus_mean": corpus_mean,
        "bm25"       : bm25,
    }

    print(
        f"  Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
        f"{len(ids)} indexed.",
        flush=True,
    )
    return G, index, meta


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete_kg(
    pdf_path : str | Path,
    *,
    save_root: Path | None = None,
) -> bool:
    """
    Delete the saved KG bundle for a PDF.

    Returns True if the bundle existed and was deleted, False otherwise.
    """
    save_dir = _bundle_dir(pdf_path, save_root)
    if save_dir.exists():
        shutil.rmtree(save_dir)
        print(f"  Deleted saved KG: {save_dir}", flush=True)
        return True
    print(f"  No saved KG found for: {Path(pdf_path).name}", flush=True)
    return False


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

def list_saved_kgs(search_dir: str | Path | None = None) -> list[dict]:
    """
    Discover all KG bundles under search_dir (defaults to current directory).

    Returns a list of metadata dicts, each with an added 'save_dir' key.
    """
    root  = Path(search_dir) if search_dir else Path.cwd()
    saved = []
    for meta_file in root.rglob("metadata.json"):
        if (meta_file.parent / "graph.pkl").exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                meta["save_dir"] = str(meta_file.parent)
                saved.append(meta)
            except Exception:
                pass
    return saved
