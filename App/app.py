"""
app.py — NLP for PDFs Desktop Application (PyQt6)
============================================================
All KG data is stored inside the application folder under saved_kgs/.
No data is written outside the application folder without user permission.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Import paths (project reorganized into App/ and Program/) ──────────────
APP_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = APP_DIR.parent
PROGRAM_DIR = PROJECT_DIR / "Program"
for _p in (str(PROJECT_DIR), str(APP_DIR), str(PROGRAM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


_DEVNULL_STREAMS: list = []


def _ensure_standard_streams() -> None:
    """
    Some GUI launch modes can set sys.std* to None (e.g., pythonw).
    Third-party libraries may call .isatty() and crash otherwise.
    """
    for name, mode in (("stdin", "r"), ("stdout", "w"), ("stderr", "w")):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "isatty"):
            replacement = open(os.devnull, mode, encoding="utf-8", errors="ignore")
            _DEVNULL_STREAMS.append(replacement)
            setattr(sys, name, replacement)


_ensure_standard_streams()

# ── PyQt6 ──────────────────────────────────────────────────────────────────
try:
    from PyQt6.QtCore import (
        Qt, QThread, QUrl, pyqtSignal, QSize, QTimer,
    )
    from PyQt6.QtGui import (
        QAction, QFont, QIcon, QImage, QPainter, QPixmap, QColor,
        QPalette, QTextCursor,
    )
    from PyQt6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDialog,
        QDialogButtonBox, QFileDialog, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QListWidget, QListWidgetItem,
        QMainWindow, QMessageBox, QPushButton, QRadioButton,
        QScrollArea, QSizePolicy, QSplitter, QStatusBar,
        QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
        QButtonGroup, QFrame, QProgressBar,
    )
except ImportError as _e:
    print(f"PyQt6 not found ({_e}). Run launcher.py first.", file=sys.stderr)
    sys.exit(1)

# ── PyMuPDF ────────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # PDF viewer will show a friendly error

# ── Application paths ──────────────────────────────────────────────────────
SAVED_KGS_DIR = APP_DIR / "saved_kgs"
ASSETS_DIR   = APP_DIR / "assets"
ICON_PATH    = ASSETS_DIR / "icon.ico"
SAVED_KGS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Persistence helpers (app-local; keeps KGs inside APP_DIR)
# ===========================================================================

def _app_save_dir(pdf_path: Path) -> Path:
    return SAVED_KGS_DIR / f"{pdf_path.stem}_kg"


def _pdf_hash(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def app_save_kg(pdf_path: Path, G, index: dict,
                model_name: str = "", caption_model: str = "") -> Path:
    import copy
    import torch
    save_dir = _app_save_dir(pdf_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prune Cell/Triple nodes (redundant with Row text)
    pruned_G = copy.deepcopy(G)
    to_remove = [n for n, d in pruned_G.nodes(data=True)
                 if d.get("type") in ("Cell", "Triple")]
    pruned_G.remove_nodes_from(to_remove)

    with open(save_dir / "graph.pkl", "wb") as f:
        pickle.dump(pruned_G, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Texts omitted — reconstructed from graph on load
    with open(save_dir / "index_meta.json", "w") as f:
        json.dump({
            "ids":   index.get("ids", []),
            "types": index.get("types", []),
        }, f)

    # Embeddings stored as float16
    if index.get("embeddings") is not None:
        torch.save(index["embeddings"].half().cpu(), save_dir / "embeddings.pt")

    cm_path = save_dir / "corpus_mean.pt"
    cm = index.get("corpus_mean")
    if cm is not None:
        torch.save(cm.detach().float().cpu(), cm_path)
    elif cm_path.exists():
        try:
            cm_path.unlink()
        except OSError:
            pass

    # Persist HNSW index
    hnsw = index.get("hnsw")
    if hnsw is not None:
        try:
            hnsw.save_index(str(save_dir / "hnsw.bin"))
        except Exception:
            pass

    meta = {
        "pdf_path":      str(pdf_path.resolve()),
        "pdf_stem":      pdf_path.stem,
        "pdf_hash":      _pdf_hash(pdf_path),
        "saved_at":      datetime.now().isoformat(),
        "model_name":    model_name,
        "caption_model": caption_model,
        "node_count":    G.number_of_nodes(),
        "edge_count":    G.number_of_edges(),
        "index_count":   len(index.get("ids", [])),
        "pruned_nodes":  len(to_remove),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return save_dir


def app_load_kg(pdf_path: Path):
    import torch
    save_dir  = _app_save_dir(pdf_path)
    meta_path = save_dir / "metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    try:
        if meta.get("pdf_hash") != _pdf_hash(pdf_path):
            return None   # PDF changed
    except FileNotFoundError:
        return None

    graph_path = save_dir / "graph.pkl"
    index_path = save_dir / "index_meta.json"
    if not graph_path.exists() or not index_path.exists():
        return None

    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    with open(index_path) as f:
        idx_data = json.load(f)

    emb_path   = save_dir / "embeddings.pt"
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

    ids   = idx_data.get("ids", [])
    types = idx_data.get("types", [])

    # Reconstruct texts from graph (or fall back to legacy JSON texts)
    texts = idx_data.get("texts", [])
    if not texts and ids:
        texts = [G.nodes[nid].get("text", "") if G.has_node(nid) else ""
                 for nid in ids]

    # Load persisted HNSW index
    hnsw = None
    hnsw_path = save_dir / "hnsw.bin"
    if hnsw_path.exists() and embeddings is not None:
        try:
            import hnswlib
            dim = embeddings.shape[1]
            hnsw = hnswlib.Index(space="cosine", dim=dim)
            hnsw.load_index(str(hnsw_path), max_elements=embeddings.shape[0])
        except Exception:
            hnsw = None

    index = {
        "ids":          ids,
        "types":        types,
        "texts":        texts,
        "embeddings":   embeddings,
        "corpus_mean":  corpus_mean,
        "hnsw":         hnsw,
    }
    return G, index, meta


def app_list_kgs() -> list[dict]:
    def _dir_size_bytes(root: Path) -> int:
        total = 0
        try:
            for p in root.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        except Exception:
            return 0
        return total

    def _format_size(num_bytes: int) -> str:
        size = float(max(0, num_bytes))
        units = ("B", "KB", "MB", "GB", "TB")
        idx = 0
        while size >= 1024.0 and idx < len(units) - 1:
            size /= 1024.0
            idx += 1
        if idx == 0:
            return f"{int(size)} {units[idx]}"
        return f"{size:.1f} {units[idx]}"

    saved = []
    for meta_file in SAVED_KGS_DIR.rglob("metadata.json"):
        if (meta_file.parent / "graph.pkl").exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                meta["save_dir"] = str(meta_file.parent)
                meta["saved_size"] = _format_size(_dir_size_bytes(meta_file.parent))
                saved.append(meta)
            except Exception:
                pass
    saved.sort(key=lambda m: m.get("saved_at", ""), reverse=True)
    return saved


def app_delete_kg(save_dir_str: str) -> bool:
    p = Path(save_dir_str)
    if p.exists() and p.is_relative_to(SAVED_KGS_DIR):
        shutil.rmtree(p)
        return True
    return False


# ===========================================================================
# stdout redirector (captures print() + status_line() from step modules)
# ===========================================================================

class _QtStream:
    """Redirect stdout to a Qt signal, handling \\r from status_line()."""

    def __init__(self, signal):
        self._signal = signal
        self._pending = ""

    def write(self, text: str) -> None:
        # status_line writes "\rMessage    " — take last segment after \r
        if "\r" in text:
            parts = text.split("\r")
            text  = parts[-1]
        text = text.strip()
        if text:
            self._signal.emit(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        # Some libraries probe tty capabilities on stdout/stderr.
        return False


# ===========================================================================
# Background pipeline worker
# ===========================================================================

class PipelineWorker(QThread):
    log      = pyqtSignal(str)
    finished = pyqtSignal(object, object)   # G, index
    error    = pyqtSignal(str)

    def __init__(self, pdf_path: str, include_images: bool,
                 caption_size: str, embedder_size: str,
                 no_captions: bool = False):
        super().__init__()
        self.pdf_path      = pdf_path
        self.include_images = include_images
        self.caption_size  = caption_size
        self.embedder_size = embedder_size
        self.no_captions   = no_captions

    def run(self) -> None:
        old_stdout = sys.stdout
        sys.stdout = _QtStream(self.log)
        try:
            self._run_pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")
        finally:
            sys.stdout = old_stdout

    def _run_pipeline(self) -> None:
        from Program.step5_query_helpers import configure_embedder_model
        configure_embedder_model(self.embedder_size)

        if self.include_images and not self.no_captions:
            from Program.step1c_image_captioning import configure_caption_model
            configure_caption_model(self.caption_size)

        # Step 1 — extract tables + page blocks
        self.log.emit("Extracting tables and page content …")
        from Program.step1_pdf_extraction import extract_tables_with_context
        extracted = extract_tables_with_context(self.pdf_path)

        # Steps 1b / 1c — images
        image_records: list = []
        if self.include_images:
            self.log.emit("Extracting images …")
            from Program.step1b_image_extraction import extract_images
            image_records = extract_images(self.pdf_path)

            if image_records and not self.no_captions:
                self.log.emit(f"Captioning {len(image_records)} image(s) …")
                from Program.step1c_image_captioning import caption_images
                image_records = caption_images(image_records)
            elif image_records:
                self.log.emit(
                    f"  {len(image_records)} image(s) extracted (captions skipped).")

        # Step 4 — build KG
        self.log.emit("Building knowledge graph …")
        from Program.step4_graph_construction import build_knowledge_graph
        G = build_knowledge_graph(extracted,
                                  image_records=image_records or None)
        self.log.emit(
            f"Graph ready: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges.")

        # Step 5 — NLP index
        self.log.emit("Building NLP index …")
        from Program.step5_query_helpers import build_index
        index = build_index(G)
        self.log.emit("Index ready — enter a query below.")

        self.finished.emit(G, index)


# ===========================================================================
# PDF viewer widget
# ===========================================================================

class PDFViewer(QWidget):
    """Renders a rolling window of pages for dynamic scrolling."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._doc:          Optional[fitz.Document] = None
        self._cur_page:     int   = 0
        self._total_pages:  int   = 0
        self._zoom:         float = 1.5
        self._window_radius: int = 2
        self._scroll_repositioning = False
        self._highlight_terms: list[str] = []
        self._sentence_highlights: dict[int, list[str]] = {}  # page_num → sentences
        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Navigation bar ────────────────────────────────────────────────
        nav = QHBoxLayout()
        nav.setContentsMargins(6, 4, 6, 4)

        self._prev_btn = QPushButton("◀ Prev")
        self._next_btn = QPushButton("Next ▶")
        self._page_lbl = QLabel("No document loaded")
        self._page_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._page_input = QLineEdit()
        self._page_input.setPlaceholderText("Page #")
        self._page_input.setFixedWidth(72)
        self._jump_btn = QPushButton("Go")
        self._jump_btn.setFixedWidth(46)

        self._zoom_out = QPushButton("−")
        self._zoom_in  = QPushButton("+")
        self._fit_btn  = QPushButton("Fit width")

        for btn in (self._prev_btn, self._next_btn,
                    self._zoom_out, self._zoom_in):
            btn.setFixedWidth(70)
        self._fit_btn.setFixedWidth(80)

        nav.addWidget(self._prev_btn)
        nav.addWidget(self._next_btn)
        nav.addWidget(self._page_lbl, 1)
        nav.addWidget(self._page_input)
        nav.addWidget(self._jump_btn)
        nav.addWidget(self._zoom_out)
        nav.addWidget(self._fit_btn)
        nav.addWidget(self._zoom_in)

        nav_bar = QFrame()
        nav_bar.setFrameShape(QFrame.Shape.StyledPanel)
        nav_bar.setLayout(nav)
        nav_bar.setFixedHeight(40)
        root.addWidget(nav_bar)

        # ── Scroll area ───────────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._pages_widget = QWidget()
        self._pages_layout = QVBoxLayout(self._pages_widget)
        self._pages_layout.setContentsMargins(0, 8, 0, 8)
        self._pages_layout.setSpacing(12)
        self._empty_label = QLabel("<br><br><br>Open a PDF to view it here.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pages_layout.addWidget(self._empty_label)
        self._scroll.setWidget(self._pages_widget)
        root.addWidget(self._scroll, 1)

        # ── Signals ───────────────────────────────────────────────────────
        self._prev_btn.clicked.connect(self.prev_page)
        self._next_btn.clicked.connect(self.next_page)
        self._zoom_in.clicked.connect(self.zoom_in)
        self._zoom_out.clicked.connect(self.zoom_out)
        self._fit_btn.clicked.connect(self.fit_to_width)
        self._jump_btn.clicked.connect(self._on_jump_page)
        self._page_input.returnPressed.connect(self._on_jump_page)
        self._scroll.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)

    # ── Public API ────────────────────────────────────────────────────────

    def load_pdf(self, path: str) -> bool:
        if fitz is None:
            self._empty_label.setText(
                "<b>PyMuPDF not installed.</b><br>"
                "Run <code>pip install PyMuPDF</code> and restart.")
            return False
        if self._doc:
            self._doc.close()
        try:
            self._doc = fitz.open(path)
        except Exception as e:
            self._empty_label.setText(f"<b>Could not open PDF:</b><br>{e}")
            return False
        self._total_pages = len(self._doc)
        self._cur_page    = 0
        self.fit_to_width()
        return True

    def close_pdf(self) -> None:
        if self._doc:
            self._doc.close()
            self._doc = None
        self._clear_rendered_pages()
        self._empty_label.setText("<br><br><br>Open a PDF to view it here.")
        self._pages_layout.addWidget(self._empty_label)
        self._page_lbl.setText("No document loaded")
        self._page_input.clear()
        self._update_nav_buttons()

    def goto_page(self, page_num: int) -> None:
        """Navigate to a 1-based page number."""
        if not self._doc:
            return
        idx = max(0, min(page_num - 1, self._total_pages - 1))
        self._cur_page = idx
        self._render()
        self._set_scroll_ratio(0.35)

    def prev_page(self) -> None:
        if self._cur_page > 0:
            self._cur_page -= 1
            self._render()

    def next_page(self) -> None:
        if self._doc and self._cur_page < self._total_pages - 1:
            self._cur_page += 1
            self._render()

    def _on_jump_page(self) -> None:
        if not self._doc:
            return
        raw = self._page_input.text().strip()
        try:
            page_num = int(raw)
        except ValueError:
            return
        self.goto_page(page_num)

    def zoom_in(self) -> None:
        self._zoom = min(self._zoom * 1.25, 6.0)
        self._render()

    def zoom_out(self) -> None:
        self._zoom = max(self._zoom / 1.25, 0.25)
        self._render()

    def fit_to_width(self) -> None:
        if not self._doc:
            return
        pw   = self._scroll.viewport().width() - 4
        page = self._doc[self._cur_page]
        if pw > 0 and page.rect.width > 0:
            self._zoom = pw / page.rect.width
        self._render()

    # ── Rendering ─────────────────────────────────────────────────────────

    def set_sentence_highlights(self, page_to_sentences: dict) -> None:
        """
        Supply sentence-level highlights per page (from query top_sentences).
        Called after each search.  Sentences are searched verbatim in the PDF
        via PyMuPDF and highlighted with a warm orange to distinguish them from
        keyword (yellow) highlights.
        """
        self._sentence_highlights = page_to_sentences or {}
        if self._doc:
            self._render()

    def set_highlight_terms(self, terms: list[str]) -> None:
        """Set keyword terms to highlight in the rendered PDF pages."""
        self._highlight_terms = [t.strip() for t in (terms or []) if t and t.strip()]
        if self._doc:
            self._render()

    def _render(self) -> None:
        if not self._doc or self._total_pages == 0:
            return
        self._clear_rendered_pages()
        start = max(0, self._cur_page - self._window_radius)
        end = min(self._total_pages - 1, self._cur_page + self._window_radius)

        for page_idx in range(start, end + 1):
            page = self._doc[page_idx]
            mat  = fitz.Matrix(self._zoom, self._zoom)
            pix  = page.get_pixmap(matrix=mat)
            img  = QImage(pix.samples, pix.width, pix.height,
                          pix.stride, QImage.Format.Format_RGB888)
            pm   = QPixmap.fromImage(img)

            # ── Highlighting ──────────────────────────────────────────────
            page_1based = page_idx + 1
            highlight_rects: list[tuple] = []   # (rect, is_sentence)

            # Sentence-level highlights (orange) — higher semantic relevance
            for sentence in self._sentence_highlights.get(page_1based, []):
                if not sentence or len(sentence) < 10:
                    continue
                # Extract a distinctive phrase (first 6+ chars of key noun)
                import re as _re
                phrases = _re.findall(r'\b[A-Za-z][A-Za-z0-9\-]{4,}\b', sentence)
                for phrase in phrases[:3]:
                    try:
                        rects = page.search_for(phrase, quads=False)
                        for r in rects:
                            highlight_rects.append((r, True))
                    except Exception:
                        pass

            # Keyword highlights (yellow) — exact term matches
            for term in self._highlight_terms:
                if not term:
                    continue
                try:
                    rects = page.search_for(term, quads=False)
                    for r in rects:
                        highlight_rects.append((r, False))
                except Exception:
                    pass

            if highlight_rects:
                painter = QPainter(pm)
                for rect, is_sentence in highlight_rects:
                    x0 = int(rect.x0 * self._zoom)
                    y0 = int(rect.y0 * self._zoom)
                    x1 = int(rect.x1 * self._zoom)
                    y1 = int(rect.y1 * self._zoom)
                    if is_sentence:
                        painter.setOpacity(0.35)
                        painter.setBrush(QColor(255, 140, 0))   # orange
                    else:
                        painter.setOpacity(0.45)
                        painter.setBrush(QColor(255, 255, 0))   # yellow
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(x0, y0, x1 - x0, y1 - y0)
                painter.end()

            holder = QWidget()
            vl = QVBoxLayout(holder)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(4)

            tag = QLabel(f"Page {page_idx + 1}")
            tag.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            tag.setStyleSheet(
                "font-size: 11px; color: #555; font-weight: bold;"
                if page_idx == self._cur_page else
                "font-size: 11px; color: #777;"
            )

            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setPixmap(pm)
            img_lbl.resize(pm.size())

            vl.addWidget(tag)
            vl.addWidget(img_lbl)
            self._pages_layout.addWidget(holder)

        self._pages_layout.addStretch(1)
        self._page_lbl.setText(
            f"Page {self._cur_page + 1} of {self._total_pages}")
        self._page_input.setText(str(self._cur_page + 1))
        self._update_nav_buttons()

    def _clear_rendered_pages(self) -> None:
        while self._pages_layout.count():
            item = self._pages_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _set_scroll_ratio(self, ratio: float) -> None:
        bar = self._scroll.verticalScrollBar()
        maxv = bar.maximum()
        if maxv <= 0:
            return
        self._scroll_repositioning = True
        try:
            bar.setValue(int(maxv * max(0.0, min(1.0, ratio))))
        finally:
            self._scroll_repositioning = False

    def _on_scroll_changed(self, value: int) -> None:
        if self._scroll_repositioning or not self._doc or self._total_pages <= 1:
            return
        bar = self._scroll.verticalScrollBar()
        maxv = bar.maximum()
        if maxv <= 0:
            return
        ratio = value / maxv
        if ratio > 0.92 and self._cur_page < self._total_pages - 1:
            self._cur_page += 1
            self._render()
            self._set_scroll_ratio(0.35)
        elif ratio < 0.08 and self._cur_page > 0:
            self._cur_page -= 1
            self._render()
            self._set_scroll_ratio(0.65)

    def _update_nav_buttons(self) -> None:
        self._prev_btn.setEnabled(self._cur_page > 0)
        self._next_btn.setEnabled(
            self._doc is not None
            and self._cur_page < self._total_pages - 1
        )
        loaded = self._doc is not None and self._total_pages > 0
        self._page_input.setEnabled(loaded)
        self._jump_btn.setEnabled(loaded)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Re-fit on resize if doc is open
        if self._doc:
            QTimer.singleShot(50, self.fit_to_width)


# ===========================================================================
# Saved KGs dialog
# ===========================================================================

class SavedKGsDialog(QDialog):
    load_requested = pyqtSignal(str, str)   # save_dir, pdf_path

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Saved Knowledge Graphs")
        self.setMinimumSize(640, 420)
        self._saved: list[dict] = []
        self._init_ui()
        self._refresh()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._load_btn   = QPushButton("Load selected")
        self._delete_btn = QPushButton("Delete selected")
        self._refresh_btn = QPushButton("Refresh")
        btn_row.addWidget(self._load_btn)
        btn_row.addWidget(self._delete_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._refresh_btn)
        layout.addLayout(btn_row)

        close_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close)
        layout.addWidget(close_box)
        close_box.rejected.connect(self.reject)

        self._load_btn.clicked.connect(self._on_load)
        self._delete_btn.clicked.connect(self._on_delete)
        self._refresh_btn.clicked.connect(self._refresh)

    def _refresh(self) -> None:
        self._saved = app_list_kgs()
        self._list.clear()
        if not self._saved:
            item = QListWidgetItem("  No saved knowledge graphs found.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(item)
            return
        for meta in self._saved:
            stem    = meta.get("pdf_stem", "unknown")
            saved   = meta.get("saved_at", "?")[:19].replace("T", "  ")
            nodes   = meta.get("node_count", "?")
            indexed = meta.get("index_count", "?")
            model   = meta.get("model_name", "?")
            size    = meta.get("saved_size", "?")
            text    = (f"{stem}   |   Saved: {saved}   |   "
                       f"Nodes: {nodes}   |   Indexed: {indexed}   |   "
                       f"Model: {model}   |   Size: {size}")
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, meta)
            self._list.addItem(item)

    def _selected_meta(self) -> Optional[dict]:
        items = self._list.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.ItemDataRole.UserRole)

    def _on_load(self) -> None:
        meta = self._selected_meta()
        if not meta:
            QMessageBox.information(self, "No selection",
                                    "Please select a saved KG first.")
            return
        self.load_requested.emit(
            meta.get("save_dir", ""),
            meta.get("pdf_path", ""),
        )
        self.accept()

    def _on_delete(self) -> None:
        meta = self._selected_meta()
        if not meta:
            QMessageBox.information(self, "No selection",
                                    "Please select a saved KG first.")
            return
        stem = meta.get("pdf_stem", "?")
        if QMessageBox.question(
            self, "Confirm delete",
            f"Delete saved KG for '{stem}'?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            app_delete_kg(meta.get("save_dir", ""))
            self._refresh()


# ===========================================================================
# Main window
# ===========================================================================

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NLP for PDFs")
        self.setMinimumSize(1100, 700)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        # State
        self._pdf_path:     Optional[Path] = None
        self._G                            = None
        self._index:        Optional[dict] = None
        self._worker:       Optional[PipelineWorker] = None
        self._query_history: list[str]     = []
        self._log_lines: list[str]         = []
        self._progress_line: Optional[str] = None

        self._init_ui()
        self._update_ui_state("no_pdf")

    # ── UI construction ───────────────────────────────────────────────────

    def _init_ui(self) -> None:
        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        # Menu bar
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        open_act = QAction("Open PDF …", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._on_open_pdf)
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        saved_act = QAction("View saved KGs …", self)
        saved_act.triggered.connect(self._on_view_saved)
        file_menu.addAction(saved_act)
        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # Central splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Left panel
        left = self._build_left_panel()
        splitter.addWidget(left)

        # Right panel — PDF viewer
        self._pdf_viewer = PDFViewer()
        splitter.addWidget(self._pdf_viewer)

        splitter.setStretchFactor(0, 38)
        splitter.setStretchFactor(1, 62)
        splitter.setHandleWidth(4)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(360)
        panel.setMaximumWidth(520)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── File section ──────────────────────────────────────────────────
        file_box = QGroupBox("Document")
        fl = QVBoxLayout(file_box)

        self._file_label = QLabel("No file loaded.")
        self._file_label.setWordWrap(True)
        fl.addWidget(self._file_label)

        btn_row = QHBoxLayout()
        self._open_btn  = QPushButton("Open PDF …")
        self._close_btn = QPushButton("Close")
        self._open_btn.clicked.connect(self._on_open_pdf)
        self._close_btn.clicked.connect(self._on_close_pdf)
        btn_row.addWidget(self._open_btn)
        btn_row.addWidget(self._close_btn)
        fl.addLayout(btn_row)
        layout.addWidget(file_box)

        # ── Initialization options ────────────────────────────────────────
        self._options_toggle = QPushButton("Initialization options ▼")
        self._options_toggle.clicked.connect(self._toggle_options_section)
        self._options_toggle.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; padding: 4px; }")
        layout.addWidget(self._options_toggle)
        self._options_box = QGroupBox()
        ol = QVBoxLayout(self._options_box)

        # Images
        self._img_chk = QCheckBox("Include images (slower)")
        self._img_chk.setChecked(True)
        ol.addWidget(self._img_chk)

        # Caption model
        cap_row = QHBoxLayout()
        cap_row.addWidget(QLabel("  Caption model:"))
        self._cap_small = QRadioButton("Small (900 MB)")
        self._cap_large = QRadioButton("Large (1.9 GB)")
        self._cap_small.setChecked(True)
        self._cap_grp = QButtonGroup(self)
        self._cap_grp.addButton(self._cap_small)
        self._cap_grp.addButton(self._cap_large)
        cap_row.addWidget(self._cap_small)
        cap_row.addWidget(self._cap_large)
        ol.addLayout(cap_row)

        self._no_cap_chk = QCheckBox(
            "  Skip captions (link images by position only)")
        ol.addWidget(self._no_cap_chk)

        # Embedder model
        emb_row = QHBoxLayout()
        emb_row.addWidget(QLabel("Query embedder:"))
        self._emb_small = QRadioButton("Small (80 MB)")
        self._emb_large = QRadioButton("Large (420 MB)")
        self._emb_large.setChecked(True)
        self._emb_grp = QButtonGroup(self)
        self._emb_grp.addButton(self._emb_small)
        self._emb_grp.addButton(self._emb_large)
        emb_row.addWidget(self._emb_small)
        emb_row.addWidget(self._emb_large)
        ol.addLayout(emb_row)
        self._apply_hardware_model_defaults()

        self._build_btn = QPushButton("Build Knowledge Graph")
        self._build_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #888; }")
        self._build_btn.clicked.connect(self._on_build_kg)
        ol.addWidget(self._build_btn)

        # Collapse when images unchecked
        self._img_chk.toggled.connect(self._on_img_toggle)
        layout.addWidget(self._options_box)

        # ── Progress log ──────────────────────────────────────────────────
        self._log_toggle = QPushButton("Build log ▼")
        self._log_toggle.clicked.connect(self._toggle_log_section)
        self._log_toggle.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; padding: 4px; }")
        layout.addWidget(self._log_toggle)
        self._log_box = QGroupBox()
        ll = QVBoxLayout(self._log_box)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._log_text.setFont(QFont("Consolas", 9))
        ll.addWidget(self._log_text)
        layout.addWidget(self._log_box)

        # ── Query section ─────────────────────────────────────────────────
        self._query_box = QGroupBox("Query")
        ql = QVBoxLayout(self._query_box)

        self._query_input = QLineEdit()
        self._query_input.setPlaceholderText("Type a query and press Enter …")
        self._query_input.returnPressed.connect(self._on_search)
        ql.addWidget(self._query_input)

        self._hyde_chk = QCheckBox("HyDE query expansion (Flan-T5, ~2s/query)")
        self._hyde_chk.setChecked(False)
        self._hyde_chk.setToolTip(
            "Generates a hypothetical answer, embeds it, and averages\n"
            "with the query embedding for richer semantic retrieval.")
        self._rerank_chk = QCheckBox("Cross-encoder reranking")
        self._rerank_chk.setChecked(True)
        self._rerank_chk.setToolTip(
            "Rerank top retrieved pages with a cross-encoder for higher precision.")
        query_opts_row = QHBoxLayout()
        query_opts_row.addWidget(self._hyde_chk)
        query_opts_row.addWidget(self._rerank_chk)
        query_opts_row.addStretch(1)
        ql.addLayout(query_opts_row)

        search_row = QHBoxLayout()
        self._search_btn = QPushButton("Search")
        self._search_btn.clicked.connect(self._on_search)
        self._clear_btn  = QPushButton("Clear history")
        self._clear_btn.clicked.connect(self._on_clear_history)
        search_row.addWidget(self._search_btn, 1)
        search_row.addWidget(self._clear_btn)
        ql.addLayout(search_row)
        layout.addWidget(self._query_box)

        # ── Results ───────────────────────────────────────────────────────
        results_lbl = QLabel("Results:")
        results_lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(results_lbl)

        self._results = QTextBrowser()
        self._results.setOpenLinks(False)
        self._results.anchorClicked.connect(self._on_result_link)
        self._results.setPlaceholderText(
            "Query results will appear here.\n"
            "Click a page link to jump to it in the PDF viewer.")
        layout.addWidget(self._results, 1)

        # ── Bottom buttons ────────────────────────────────────────────────
        bot = QHBoxLayout()
        self._save_btn  = QPushButton("Save KG")
        self._view_saved_btn = QPushButton("View saved KGs …")
        self._save_btn.clicked.connect(self._on_save_kg)
        self._view_saved_btn.clicked.connect(self._on_view_saved)
        bot.addWidget(self._save_btn)
        bot.addWidget(self._view_saved_btn)
        layout.addLayout(bot)

        return panel

    # ── UI state machine ──────────────────────────────────────────────────

    def _update_ui_state(self, state: str) -> None:
        """
        States: no_pdf | pdf_loaded | building | kg_ready | build_failed
        """
        pdf_loaded = state in ("pdf_loaded", "building", "kg_ready", "build_failed")
        building   = state == "building"
        kg_ready   = state == "kg_ready"
        build_failed = state == "build_failed"
        init_locked = kg_ready

        self._close_btn.setEnabled(pdf_loaded and not building)
        self._options_box.setVisible(pdf_loaded)
        self._options_box.setEnabled(pdf_loaded and not building and not init_locked)
        self._log_box.setVisible(building or kg_ready or build_failed)
        self._options_toggle.setVisible(pdf_loaded)
        self._options_toggle.setEnabled(pdf_loaded and not building)
        self._log_toggle.setVisible(building or kg_ready or build_failed)
        self._options_toggle.setText(
            "Initialization options ▼" if self._options_box.isVisible() else "Initialization options ▶")
        self._log_toggle.setText(
            "Build log ▼" if self._log_box.isVisible() else "Build log ▶")
        self._build_btn.setEnabled(pdf_loaded and not building and not init_locked)
        self._build_btn.setText(
            "Building …" if building else "Build Knowledge Graph")
        self._query_box.setVisible(kg_ready)
        self._results.setVisible(kg_ready)
        self._save_btn.setEnabled(kg_ready)
        self._search_btn.setEnabled(kg_ready)

    def _on_img_toggle(self, checked: bool) -> None:
        self._cap_small.setEnabled(checked)
        self._cap_large.setEnabled(checked)
        self._no_cap_chk.setEnabled(checked)

    def _toggle_options_section(self) -> None:
        visible = not self._options_box.isVisible()
        self._options_box.setVisible(visible)
        self._options_toggle.setText(
            "Initialization options ▼" if visible else "Initialization options ▶")

    def _toggle_log_section(self) -> None:
        visible = not self._log_box.isVisible()
        self._log_box.setVisible(visible)
        self._log_toggle.setText("Build log ▼" if visible else "Build log ▶")

    def _apply_hardware_model_defaults(self) -> None:
        """Choose sensible initial model sizes from local hardware."""
        has_cuda = False
        vram_gb = 0.0
        try:
            import torch  # type: ignore
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024 ** 3)
        except Exception:
            has_cuda = False

        if has_cuda and vram_gb >= 8.0:
            self._emb_large.setChecked(True)
            self._cap_large.setChecked(True)
        elif has_cuda:
            self._emb_large.setChecked(True)
            self._cap_small.setChecked(True)
        else:
            self._emb_small.setChecked(True)
            self._cap_small.setChecked(True)

    def _clear_log(self) -> None:
        self._log_lines.clear()
        self._progress_line = None
        self._log_text.clear()

    def _log_progress_key(self, msg: str) -> str | None:
        # Treat percent updates as "live progress" lines that should be replaced.
        if not re.search(r"\b\d{1,3}%\b", msg):
            return None
        return "progress"

    def _append_log(self, msg: str) -> None:
        clean = msg.strip()
        if not clean:
            return

        new_key = self._log_progress_key(clean)
        if new_key:
            # Keep exactly one live progress line at the bottom.
            self._progress_line = clean
        else:
            self._log_lines.append(clean)

        # Safety: ensure no legacy progress lines remain in static log lines.
        render_lines = [ln for ln in self._log_lines if self._log_progress_key(ln) is None]
        if self._progress_line:
            render_lines.append(self._progress_line)
        self._log_text.setPlainText("\n".join(render_lines))
        self._log_text.moveCursor(QTextCursor.MoveOperation.End)

    # ── File management ───────────────────────────────────────────────────

    def _on_open_pdf(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", str(Path.home()),
            "PDF files (*.pdf);;All files (*.*)",
        )
        if not path_str:
            return
        path = Path(path_str).resolve()

        # Permission check for files outside app folder
        if not self._check_permission(path, "open"):
            return

        self._pdf_path = path
        self._G      = None
        self._index  = None
        self._query_history.clear()
        self._clear_log()
        self._results.clear()
        self._file_label.setText(f"<b>{path.name}</b><br>"
                                 f"<small>{path.parent}</small>")

        # Load in viewer
        self._pdf_viewer.load_pdf(str(path))

        # Check for existing saved KG
        saved = app_load_kg(path)
        if saved:
            G, index, meta = saved
            reply = QMessageBox.question(
                self, "Saved KG found",
                f"A saved knowledge graph was found for this PDF\n"
                f"(saved {meta.get('saved_at','?')[:19].replace('T',' ')}, "
                f"model: {meta.get('model_name','?')}).\n\n"
                f"Load the saved version?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._G     = G
                self._index = index
                self._append_log("Loaded saved knowledge graph.")
                self._update_ui_state("kg_ready")
                self._status.showMessage(
                    f"Loaded saved KG — {G.number_of_nodes()} nodes")
                return

        self._update_ui_state("pdf_loaded")
        self._status.showMessage(f"PDF loaded: {path.name}")

    def _on_close_pdf(self) -> None:
        self._pdf_path = None
        self._G        = None
        self._index    = None
        self._query_history.clear()
        self._file_label.setText("No file loaded.")
        self._clear_log()
        self._results.clear()
        self._pdf_viewer.close_pdf()
        self._update_ui_state("no_pdf")
        self._status.showMessage("Ready")

    def _check_permission(self, path: Path, action: str = "access") -> bool:
        try:
            path.relative_to(APP_DIR)
            return True
        except ValueError:
            reply = QMessageBox.question(
                self, "File access permission",
                f"This application needs to {action}:\n\n"
                f"{path}\n\n"
                "This file is outside the application folder.\n"
                "Allow access?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            return reply == QMessageBox.StandardButton.Yes

    # ── Pipeline ──────────────────────────────────────────────────────────

    def _on_build_kg(self) -> None:
        if not self._pdf_path:
            return

        caption_size  = "large" if self._cap_large.isChecked() else "small"
        embedder_size = "large" if self._emb_large.isChecked() else "small"
        include_imgs  = self._img_chk.isChecked()
        no_captions   = self._no_cap_chk.isChecked()

        self._clear_log()
        self._update_ui_state("building")
        self._status.showMessage("Building knowledge graph …")

        self._worker = PipelineWorker(
            str(self._pdf_path),
            include_imgs, caption_size, embedder_size, no_captions,
        )
        self._worker.log.connect(self._on_pipeline_log)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()

    def _on_pipeline_log(self, msg: str) -> None:
        self._append_log(msg)

    def _on_pipeline_finished(self, G, index: dict) -> None:
        self._G     = G
        self._index = index
        self._update_ui_state("kg_ready")
        self._status.showMessage(
            f"KG ready: {G.number_of_nodes()} nodes. Enter a query below.")
        self._query_input.setFocus()

    def _on_pipeline_error(self, msg: str) -> None:
        # Keep build log visible so users can inspect the failure.
        self._update_ui_state("build_failed")
        self._status.showMessage("Build failed — see log.")
        self._append_log("--- ERROR ---")
        self._append_log(msg)
        QMessageBox.critical(self, "Build error",
                             "An error occurred. See the build log for details.")

    # ── Query ─────────────────────────────────────────────────────────────

    def _on_search(self) -> None:
        if not self._G or not self._index:
            return
        raw = self._query_input.text().strip()
        if not raw:
            return

        self._query_history.append(raw)
        self._status.showMessage("Searching …")

        try:
            from Program.step5_query_helpers import (
                build_effective_query, query as kg_query,
            )
            effective = build_effective_query(self._query_history, window=2)
            results   = kg_query(
                self._G, self._index, effective, top_k=5,
                use_generative=self._rerank_chk.isChecked(),
                use_hyde=self._hyde_chk.isChecked(),
            )
        except Exception as e:
            self._results.setHtml(
                f"<p style='color:red'><b>Query error:</b> {e}</p>")
            self._status.showMessage("Query error.")
            return

        html = self._format_results_html(results, raw)
        self._results.setHtml(html)
        self._results.moveCursor(QTextCursor.MoveOperation.Start)
        self._status.showMessage(
            f"Found {len(results)} result(s) for: {raw!r}")

        # ── Highlight PDF: keywords (yellow) + relevant sentences (orange) ──
        terms = self._extract_highlight_terms(raw)
        self._pdf_viewer.set_highlight_terms(terms)
        page_to_sents: dict[int, list[str]] = {}
        for r in results:
            pg    = r.get("page")
            sents = r.get("top_sentences", [])
            if pg and sents:
                page_to_sents[pg] = sents
        self._pdf_viewer.set_sentence_highlights(page_to_sents)

    def _extract_highlight_terms(self, query_text: str) -> list[str]:
        """
        Build a compact list of meaningful terms from the user query for
        exact-match PDF highlights.
        """
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
            "this", "to", "what", "which", "with",
        }
        tokens = re.findall(r"[A-Za-z0-9\-_/]+", query_text.lower())
        terms: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if len(tok) < 3 or tok in stop_words:
                continue
            if tok not in seen:
                seen.add(tok)
                terms.append(tok)
        return terms[:12]

    def _on_clear_history(self) -> None:
        self._query_history.clear()
        self._results.clear()
        self._status.showMessage("Query history cleared.")

    def _on_result_link(self, url: QUrl) -> None:
        # Be permissive because QTextBrowser can normalize custom URLs in
        # slightly different ways depending on platform/Qt version.
        raw_page = ""
        if url.scheme() == "page":
            raw_page = url.host() or url.path().lstrip("/")
        if not raw_page:
            s = url.toString()
            m = re.search(r"(?:^|://|:)(\d+)(?:/)?$", s)
            if m:
                raw_page = m.group(1)
        try:
            page_num = int(raw_page)
        except (TypeError, ValueError):
            return

        self._pdf_viewer.goto_page(page_num)
        self._status.showMessage(f"Jumped to page {page_num}")

    def _format_results_html(self, results: list, query_text: str) -> str:
        if not results:
            return (
                f"<p><i>No results found for: <b>{query_text}</b></i></p>")

        # Check for negated terms on first result
        neg_note = ""
        if results and results[0].get("negated"):
            neg_terms = results[0]["negated"][:6]
            neg_note  = (
                f"<p style='color:#e65100; font-size:11px;'>"
                f"⊘ Penalising: {', '.join(neg_terms)}</p>")

        html_parts = [
            f"<p><b>Query:</b> {query_text}</p>",
            neg_note,
        ]

        colour_map = {
            "Row":   "#1565C0",
            "Image": "#6A1B9A",
        }

        for r in results:
            page_num  = r.get("page")
            node_type = r.get("type", "Row")
            score     = r.get("score", 0.0)
            colour    = colour_map.get(node_type, "#333")

            page_link = (
                f'<a href="page:{page_num}" '
                f'style="color:#1976D2; font-weight:bold; '
                f'text-decoration:none;">📄 Page {page_num}</a>'
                if page_num is not None else "—"
            )

            block = (
                f'<div style="margin:0 0 10px 0; padding:10px; '
                f'border-left:4px solid {colour}; '
                f'background:#f5f7fa; border-radius:0 4px 4px 0;">'
                f'<b>#{r["rank"]}</b>&nbsp; '
                f'Score: <span style="color:{colour}">'
                f'{score:.2f}</span>'
                f'&nbsp;|&nbsp;{page_link}'
                f'&nbsp;|&nbsp;<span style="color:#888; font-size:11px;">'
                f'[{node_type}]</span><br>'
            )

            if node_type == "Row":
                section = r.get("section", "")
                caption = r.get("caption", "")
                if section:
                    block += (f'<span style="font-size:11px; color:#555;">'
                              f'Section: {section}</span><br>')
                if caption:
                    block += (f'<span style="font-size:11px; color:#555;">'
                              f'Table: {caption}</span><br>')
                prose = (r.get("page_prose") or "").strip()
                if prose:
                    display = prose[:400] + ("…" if len(prose) > 400 else "")
                    block += (f'<br><span style="font-size:12px;">'
                              f'{display}</span>')
                for cond in r.get("conditions", []):
                    block += (f'<br><span style="color:#e65100; '
                              f'font-size:11px;">⚠ {cond}</span>')

            elif node_type == "Image":
                cap   = r.get("caption", "")
                fpath = r.get("path", "")
                if cap:
                    block += f'<span style="font-size:12px;">{cap}</span><br>'
                if r.get("linked_table"):
                    block += (f'<span style="font-size:11px; color:#555;">'
                              f'Related table: {r["linked_table"]}</span><br>')
                if fpath:
                    block += (f'<span style="font-size:10px; color:#888;">'
                              f'{fpath}</span>')

            block += "</div>"
            html_parts.append(block)

        return "\n".join(html_parts)

    # ── Save / load KGs ──────────────────────────────────────────────────

    def _on_save_kg(self) -> None:
        if not self._G or not self._index or not self._pdf_path:
            QMessageBox.information(self, "Nothing to save",
                                    "Build a knowledge graph first.")
            return
        try:
            from Program.step5_query_helpers import MODEL_NAME as emb_model
        except Exception:
            emb_model = ""
        try:
            from Program.step1c_image_captioning import MODEL_NAME as cap_model
        except Exception:
            cap_model = ""

        save_dir = app_save_kg(
            self._pdf_path, self._G, self._index,
            model_name=emb_model, caption_model=cap_model,
        )
        self._status.showMessage(f"KG saved → {save_dir.name}")
        QMessageBox.information(
            self, "Saved",
            f"Knowledge graph saved to:\n{save_dir}")

    def _on_view_saved(self) -> None:
        dlg = SavedKGsDialog(self)
        dlg.load_requested.connect(self._on_load_saved_kg)
        dlg.exec()

    def _on_load_saved_kg(self, save_dir_str: str, pdf_path_str: str) -> None:
        # Locate the KG data directly from save_dir_str
        save_dir  = Path(save_dir_str)
        meta_path = save_dir / "metadata.json"
        if not meta_path.exists():
            QMessageBox.warning(self, "Not found",
                                "Saved KG metadata not found.")
            return

        # Load via the saved pdf_path
        pdf_path = Path(pdf_path_str) if pdf_path_str else None

        # We need the pdf_path to match the saved hash — try loading
        if pdf_path and pdf_path.exists():
            saved = app_load_kg(pdf_path)
        else:
            # PDF may have moved — load graph directly
            saved = self._load_kg_from_dir(save_dir)

        if saved is None:
            QMessageBox.warning(
                self, "Load failed",
                "Could not load the saved KG. "
                "The PDF may have changed or files may be missing.")
            return

        G, index, meta = saved

        self._G             = G
        self._index         = index
        self._query_history.clear()

        # Try to load the PDF in the viewer
        if pdf_path and pdf_path.exists():
            if self._check_permission(pdf_path, "open"):
                self._pdf_path = pdf_path
                self._file_label.setText(
                    f"<b>{pdf_path.name}</b><br>"
                    f"<small>{pdf_path.parent}</small>")
                self._pdf_viewer.load_pdf(str(pdf_path))
        else:
            # Ask user to locate the PDF
            path_str, _ = QFileDialog.getOpenFileName(
                self, f"Locate PDF for this KG",
                str(Path.home()), "PDF files (*.pdf)",
            )
            if path_str:
                p = Path(path_str).resolve()
                if self._check_permission(p, "open"):
                    self._pdf_path = p
                    self._file_label.setText(
                        f"<b>{p.name}</b><br><small>{p.parent}</small>")
                    self._pdf_viewer.load_pdf(str(p))

        self._clear_log()
        self._append_log("Loaded saved knowledge graph.")
        self._results.clear()
        self._update_ui_state("kg_ready")
        self._status.showMessage(
            f"Loaded KG: {G.number_of_nodes()} nodes")

    @staticmethod
    def _load_kg_from_dir(save_dir: Path):
        """Load G and index directly from a save_dir without hash check."""
        import torch
        graph_path = save_dir / "graph.pkl"
        index_path = save_dir / "index_meta.json"
        meta_path  = save_dir / "metadata.json"

        if not graph_path.exists() or not index_path.exists():
            return None

        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        with open(index_path) as f:
            idx_data = json.load(f)

        emb_path   = save_dir / "embeddings.pt"
        embeddings = (torch.load(emb_path, map_location="cpu")
                      if emb_path.exists() else None)
        if embeddings is not None and embeddings.dtype == torch.float16:
            embeddings = embeddings.float()

        corpus_mean = None
        cm_path = save_dir / "corpus_mean.pt"
        if cm_path.exists():
            corpus_mean = torch.load(cm_path, map_location="cpu")
            if corpus_mean is not None and corpus_mean.dtype == torch.float16:
                corpus_mean = corpus_mean.float()

        index = {
            "ids":          idx_data.get("ids", []),
            "types":        idx_data.get("types", []),
            "texts":        idx_data.get("texts", []),
            "embeddings":   embeddings,
            "corpus_mean":  corpus_mean,
        }
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        return G, index, meta


# ===========================================================================
# Application entry point
# ===========================================================================

def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("NLPforPDFs")
    app.setApplicationDisplayName("NLP for PDFs")

    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    # Increase base font size slightly for readability
    font = app.font()
    font.setPointSize(max(font.pointSize(), 10))
    app.setFont(font)

    win = MainWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
