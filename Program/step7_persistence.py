"""
Step 7 — Persistence
====================
Save, load, and delete the knowledge graph and NLP index so a PDF
does not need to be re-processed on every run.

Saved structure (in a folder beside the PDF):
    <pdf_stem>_kg/
        graph.pkl         — NetworkX graph (pickle, Cell/Triple nodes pruned)
        index_meta.json   — node IDs and types (texts reconstructed from graph)
        embeddings.pt     — torch tensor (float16 for ~50% size savings), mean-centered
        corpus_mean.pt    — 1×d mean vector used to center queries (float32)
        hnsw.bin          — HNSW ANN index (avoids rebuild on load)
        bm25.pkl          — sparse BM25 index (rank_bm25), optional
        metadata.json     — PDF hash, timestamps, model names

The PDF is identified by a SHA-256 hash, so if the file changes the
saved graph is automatically invalidated.
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
    import hnswlib as _hnswlib
    _HNSW_AVAILABLE = True
except ImportError:
    _hnswlib = None   # type: ignore[assignment,misc]
    _HNSW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _bundle_dir(pdf_path: str | Path, save_root: Path | None = None) -> Path:
    """Directory for a PDF's KG bundle: beside the PDF, or under ``save_root``."""
    p = Path(pdf_path).resolve()
    if save_root is not None:
        return Path(save_root).resolve() / f"{p.stem}_kg"
    return p.parent / f"{p.stem}_kg"


def _save_dir(pdf_path: str | Path) -> Path:
    return _bundle_dir(pdf_path, None)


def _pdf_hash(pdf_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _prune_graph_for_storage(G):
    """Return a copy of G with Cell and Triple nodes removed.

    These nodes' information is fully redundant with Row node text
    and is not used by the query pipeline.
    """
    pruned = copy.deepcopy(G)
    to_remove = [
        n for n, d in pruned.nodes(data=True)
        if d.get("type") in ("Cell", "Triple")
    ]
    pruned.remove_nodes_from(to_remove)
    return pruned, len(to_remove)


def save_kg(
    pdf_path: str | Path,
    G,
    index: dict,
    *,
    model_name:    str = "",
    caption_model: str = "",
    save_root: Path | None = None,
) -> Path:
    """
    Save graph and index to disk beside the PDF, or under ``save_root`` as
    ``{pdf_stem}_kg/`` when ``save_root`` is set.

    Storage optimisations applied:
    - Cell/Triple nodes pruned from graph (redundant with Row text)
    - Embeddings stored in float16 (halves size, negligible quality loss)
    - HNSW index persisted (avoids rebuild on load)
    - BM25 index persisted when present (rebuilt from texts on load if missing)
    - Index texts omitted from JSON (reconstructed from graph on load)

    Returns the save directory path.
    """
    save_dir = _bundle_dir(pdf_path, save_root)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Graph — prune redundant Cell / Triple nodes
    pruned_G, pruned_count = _prune_graph_for_storage(G)
    with open(save_dir / "graph.pkl", "wb") as f:
        pickle.dump(pruned_G, f, protocol=pickle.HIGHEST_PROTOCOL)
    if pruned_count:
        print(f"  Pruned {pruned_count} Cell/Triple nodes from saved graph.")

    # Index metadata (texts reconstructed from graph on load)
    index_data = {
        "ids":   index.get("ids", []),
        "types": index.get("types", []),
    }
    with open(save_dir / "index_meta.json", "w") as f:
        json.dump(index_data, f)

    # Embeddings — stored as float16 (~50% size savings), always CPU
    emb = index.get("embeddings")
    if emb is not None:
        torch.save(emb.detach().half().cpu(), save_dir / "embeddings.pt")

    cm_path = save_dir / "corpus_mean.pt"
    cm = index.get("corpus_mean")
    if cm is not None:
        torch.save(cm.detach().float().cpu(), cm_path)
    elif cm_path.exists():
        try:
            cm_path.unlink()
        except OSError:
            pass

    # HNSW index
    hnsw = index.get("hnsw")
    if hnsw is not None and _HNSW_AVAILABLE:
        try:
            hnsw.save_index(str(save_dir / "hnsw.bin"))
        except Exception:
            pass

    # BM25 sparse index (hybrid retrieval)
    bm25 = index.get("bm25")
    bm25_path = save_dir / "bm25.pkl"
    if bm25 is not None:
        try:
            with open(bm25_path, "wb") as f:
                pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"  Warning: could not save BM25 index: {e}")
    elif bm25_path.exists():
        try:
            bm25_path.unlink()
        except OSError:
            pass

    # Metadata
    p = Path(pdf_path).resolve()
    meta = {
        "pdf_path":      str(p),
        "pdf_stem":      p.stem,
        "pdf_hash":      _pdf_hash(pdf_path),
        "saved_at":      datetime.now().isoformat(),
        "model_name":    model_name,
        "caption_model": caption_model,
        "node_count":    G.number_of_nodes(),
        "edge_count":    G.number_of_edges(),
        "index_count":   len(index.get("ids", [])),
        "pruned_nodes":  pruned_count,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  KG saved → {save_dir}")
    return save_dir


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_kg(pdf_path: str | Path, *, save_root: Path | None = None) -> tuple | None:
    """
    Load a previously saved graph and index.

    Looks beside the PDF by default, or under ``save_root`` / ``{stem}_kg`` when
    ``save_root`` is set (same layout as :func:`save_kg`).

    Returns (G, index, metadata) if a valid save exists and the PDF hash matches,
    or None if nothing is saved or the PDF has changed.

    Handles both legacy saves (texts in JSON, float32 embeddings) and
    optimised saves (texts reconstructed from graph, float16 → float32).
    """
    save_dir  = _bundle_dir(pdf_path, save_root)
    meta_path = save_dir / "metadata.json"

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    current_hash = _pdf_hash(pdf_path)
    if meta.get("pdf_hash") != current_hash:
        print("  Saved KG found but PDF has changed — will rebuild.")
        return None

    graph_path = save_dir / "graph.pkl"
    index_path = save_dir / "index_meta.json"
    emb_path   = save_dir / "embeddings.pt"

    if not graph_path.exists() or not index_path.exists():
        print("  Saved KG metadata found but files are missing — will rebuild.")
        return None

    print(f"  Loading saved KG (saved {meta.get('saved_at', '?')}, "
          f"model: {meta.get('model_name', '?')})...")

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    with open(index_path) as f:
        index_data = json.load(f)

    # Embeddings — upcast float16 → float32 for compute
    embeddings = None
    if emb_path.exists():
        embeddings = torch.load(emb_path, map_location="cpu")
        if embeddings is not None and embeddings.dtype == torch.float16:
            embeddings = embeddings.float()

    corpus_mean = None
    cm_path = save_dir / "corpus_mean.pt"
    if cm_path.exists():
        corpus_mean = torch.load(cm_path, map_location="cpu")
        if corpus_mean is not None and corpus_mean.dtype == torch.float16:
            corpus_mean = corpus_mean.float()

    ids   = index_data.get("ids", [])
    types = index_data.get("types", [])

    # Reconstruct texts from graph (or fall back to legacy JSON texts)
    texts = index_data.get("texts", [])
    if not texts and ids:
        texts = [G.nodes[nid].get("text", "") if G.has_node(nid) else ""
                 for nid in ids]

    # Load persisted HNSW index if available
    hnsw = None
    hnsw_path = save_dir / "hnsw.bin"
    if hnsw_path.exists() and _HNSW_AVAILABLE and embeddings is not None:
        try:
            dim = embeddings.shape[1]
            hnsw = _hnswlib.Index(space="cosine", dim=dim)
            hnsw.load_index(str(hnsw_path), max_elements=embeddings.shape[0])
        except Exception:
            hnsw = None

    bm25 = None
    bm25_path = save_dir / "bm25.pkl"
    if bm25_path.exists():
        try:
            with open(bm25_path, "rb") as f:
                bm25 = pickle.load(f)
        except Exception:
            bm25 = None
    if bm25 is None and texts:
        try:
            from Program.step5_query_helpers import _build_bm25

            bm25 = _build_bm25(texts)
        except Exception:
            bm25 = None

    index = {
        "ids":          ids,
        "types":        types,
        "texts":        texts,
        "embeddings":   embeddings,
        "corpus_mean":  corpus_mean,
        "bm25":         bm25,
        "hnsw":         hnsw,
    }

    print(f"  Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(index['ids'])} indexed.")
    return G, index, meta


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete_kg(pdf_path: str | Path) -> bool:
    """
    Delete the saved KG for a PDF.
    Returns True if files were deleted, False if nothing was found.
    """
    save_dir = _save_dir(pdf_path)
    if save_dir.exists():
        shutil.rmtree(save_dir)
        print(f"  Deleted saved KG: {save_dir}")
        return True
    print(f"  No saved KG found for: {Path(pdf_path).name}")
    return False


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

def list_saved_kgs(search_dir: str | Path | None = None) -> list[dict]:
    """
    Find all saved KG directories in search_dir (default: user's home).
    Returns a list of metadata dicts with an added 'save_dir' key.
    """
    root  = Path(search_dir) if search_dir else Path.home()
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
