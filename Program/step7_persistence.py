"""
Step 7 — Persistence
====================
Save, load, and delete the knowledge graph and NLP index so a PDF
does not need to be re-processed on every run.

Saved structure (in a folder beside the PDF):
    <pdf_stem>_kg/
        graph.pkl         — NetworkX graph (pickle)
        index_meta.json   — node IDs, types, texts
        embeddings.pt     — torch tensor
        metadata.json     — PDF hash, timestamps, model names

The PDF is identified by a SHA-256 hash, so if the file changes the
saved graph is automatically invalidated.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _save_dir(pdf_path: str | Path) -> Path:
    p = Path(pdf_path).resolve()
    return p.parent / f"{p.stem}_kg"


def _pdf_hash(pdf_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_kg(
    pdf_path: str | Path,
    G,
    index: dict,
    *,
    model_name:    str = "",
    caption_model: str = "",
) -> Path:
    """
    Save graph and index to disk beside the PDF.

    Returns the save directory path.
    """
    save_dir = _save_dir(pdf_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Graph
    with open(save_dir / "graph.pkl", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Index metadata (JSON-serialisable parts)
    index_data = {
        "ids":   index.get("ids", []),
        "types": index.get("types", []),
        "texts": index.get("texts", []),
    }
    with open(save_dir / "index_meta.json", "w") as f:
        json.dump(index_data, f)

    # Embeddings tensor
    if index.get("embeddings") is not None:
        torch.save(index["embeddings"].cpu(), save_dir / "embeddings.pt")

    # Metadata
    meta = {
        "pdf_path":      str(Path(pdf_path).resolve()),
        "pdf_hash":      _pdf_hash(pdf_path),
        "saved_at":      datetime.now().isoformat(),
        "model_name":    model_name,
        "caption_model": caption_model,
        "node_count":    G.number_of_nodes(),
        "edge_count":    G.number_of_edges(),
        "index_count":   len(index.get("ids", [])),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  KG saved → {save_dir}")
    return save_dir


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_kg(pdf_path: str | Path) -> tuple | None:
    """
    Load a previously saved graph and index.

    Returns (G, index, metadata) if a valid save exists and the PDF hash matches,
    or None if nothing is saved or the PDF has changed.

    ``metadata`` is the ``metadata.json`` dict (includes ``model_name`` for the embedder).
    """
    save_dir  = _save_dir(pdf_path)
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

    embeddings = None
    if emb_path.exists():
        embeddings = torch.load(emb_path, map_location="cpu")

    index = {
        "ids":        index_data.get("ids", []),
        "types":      index_data.get("types", []),
        "texts":      index_data.get("texts", []),
        "embeddings": embeddings,
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
