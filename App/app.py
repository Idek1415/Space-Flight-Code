"""
app.py — NLP for PDFs Desktop Application (PyQt6)
==================================================
Redesigned for the step1-5 pipeline and updated UI layout.

Left panel layout:
  BUILD MODE  (full height during construction):
    - File section
    - Build options (embed model display)
    - Build log (scrollable, expands to fill)
    - Build button

  QUERY MODE  (after KG ready):
    - Query response box (large, ~70% height) — shows top-k then generative answer
    - Query input area
    - Collapsed settings strip (build log toggle, gen toggle, model, save)

Right panel: PDF viewer (navigation + highlighting unchanged)
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
from urllib.parse import quote, unquote
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Application paths
# ---------------------------------------------------------------------------
APP_DIR       = Path(__file__).parent.resolve()
ROOT_DIR      = APP_DIR.parent
PROGRAM_DIR   = ROOT_DIR / "Program"
SAVED_KGS_DIR = APP_DIR / "saved_kgs"
ASSETS_DIR    = APP_DIR / "assets"
ICON_PATH     = ASSETS_DIR / "icon.ico"
SAVED_KGS_DIR.mkdir(parents=True, exist_ok=True)

# Add project/module dirs so step1-5 modules are importable
for _p in (PROGRAM_DIR, ROOT_DIR, APP_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

_DEVNULL_STREAMS: list = []

def _ensure_standard_streams() -> None:
    for name, mode in (("stdin", "r"), ("stdout", "w"), ("stderr", "w")):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "isatty"):
            replacement = open(os.devnull, mode, encoding="utf-8", errors="ignore")
            _DEVNULL_STREAMS.append(replacement)
            setattr(sys, name, replacement)

_ensure_standard_streams()

# ---------------------------------------------------------------------------
# PyQt6
# ---------------------------------------------------------------------------
try:
    from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal, QTimer, QSize
    from PyQt6.QtGui import (
        QAction, QFont, QIcon, QImage, QPainter, QPixmap,
        QColor, QTextCursor, QPalette,
    )
    from PyQt6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
        QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
        QPushButton, QScrollArea, QSizePolicy, QSplitter, QStatusBar,
        QTextBrowser, QTextEdit, QVBoxLayout, QWidget, QFrame,
        QStackedWidget, QButtonGroup, QRadioButton,
    )
except ImportError as _e:
    print(f"PyQt6 not found ({_e}). Run: pip install PyQt6", file=sys.stderr)
    sys.exit(1)

try:
    import fitz
except ImportError:
    fitz = None

# ---------------------------------------------------------------------------
# Step-4 wrappers (persistence inside APP_DIR/saved_kgs/)
# ---------------------------------------------------------------------------

def _pdf_hash(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def app_save_kg(pdf_path: Path, G, index: dict, model_name: str = "") -> Path:
    from step4_persistence import save_kg
    return save_kg(pdf_path, G, index,
                   model_name=model_name,
                   save_root=SAVED_KGS_DIR)


def app_load_kg(pdf_path: Path):
    from step4_persistence import load_kg
    return load_kg(pdf_path, save_root=SAVED_KGS_DIR)


def app_list_kgs() -> list[dict]:
    saved = []
    for meta_file in SAVED_KGS_DIR.rglob("metadata.json"):
        if (meta_file.parent / "graph.pkl").exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                # Compute directory size
                total = sum(
                    p.stat().st_size
                    for p in meta_file.parent.rglob("*") if p.is_file()
                )
                sz = total / (1024 ** 2)
                meta["saved_size"] = f"{sz:.1f} MB" if sz >= 1 else f"{total // 1024} KB"
                meta["save_dir"]   = str(meta_file.parent)
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


# ---------------------------------------------------------------------------
# stdout → Qt signal
# ---------------------------------------------------------------------------

class _QtStream:
    def __init__(self, signal):
        self._signal = signal

    def write(self, text: str) -> None:
        if "\r" in text:
            text = text.split("\r")[-1]
        text = text.strip()
        if text:
            self._signal.emit(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Pipeline worker  (step1 + step2)
# ---------------------------------------------------------------------------

class PipelineWorker(QThread):
    log      = pyqtSignal(str)
    finished = pyqtSignal(object, object)   # G, index
    error    = pyqtSignal(str)

    def __init__(self, pdf_path: str):
        super().__init__()
        self.pdf_path = pdf_path

    def run(self) -> None:
        old_stdout = sys.stdout
        sys.stdout = _QtStream(self.log)
        try:
            self.log.emit("Step 1 — Extracting PDF content …")
            from step1_pdf_extraction import extract_pdf
            doc = extract_pdf(self.pdf_path)

            page_count = len(doc.get("pages", {}))
            section_count = sum(
                len(p.get("sections", []))
                for p in doc["pages"].values()
            )
            table_count = sum(
                len(p.get("tables", []))
                for p in doc["pages"].values()
            )
            self.log.emit(
                f"  Extracted {page_count} pages, "
                f"{section_count} sections, {table_count} tables."
            )

            self.log.emit("Step 2 — Building knowledge graph …")
            from step2_graph_construction import build_knowledge_graph, build_index
            G = build_knowledge_graph(doc, self.pdf_path)

            self.log.emit("Step 2 — Building retrieval index …")
            index = build_index(G)

            self.finished.emit(G, index)

        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")
        finally:
            sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# Query + generation worker  (step3 + step5)
# ---------------------------------------------------------------------------

class QueryWorker(QThread):
    results_ready    = pyqtSignal(list)   # step3 top-k results (fast)
    generation_done  = pyqtSignal(dict)   # step5 output (slow)
    generation_error = pyqtSignal(str)
    log              = pyqtSignal(str)

    def __init__(
        self, G, index, query_text: str, top_k: int,
        use_generative: bool, backend: str, model_name: str,
    ):
        super().__init__()
        self.G              = G
        self.index          = index
        self.query_text     = query_text
        self.top_k          = top_k
        self.use_generative = use_generative
        self.backend        = backend
        self.model_name     = model_name

    def run(self) -> None:
        import numpy as np
        old_stdout = sys.stdout
        sys.stdout = _QtStream(self.log)
        try:
            from step3_retrieval import query as step3_query
            results = step3_query(
                self.G, self.index, self.query_text, top_k=self.top_k
            )
            self.results_ready.emit(results)
        except Exception as e:
            self.generation_error.emit(
                f"Retrieval failed:\n{e}\n\n{traceback.format_exc()}"
            )
            sys.stdout = old_stdout
            return

        if not (self.use_generative and results):
            sys.stdout = old_stdout
            return

        try:
            # ── Fix: step3 stores query_embedding as (1, 768) due to
            # broadcasting against corpus_mean (shape (1, 768)).
            # step5.mmr_select calls .unsqueeze(0) expecting 1-D input,
            # producing (1, 1, 768) which causes torch.mm to raise
            # "self must be a matrix".  Flatten to 1-D here as a guard.
            for r in results:
                r["_query_text"] = self.query_text
                qe = r.get("query_embedding")
                if qe is not None:
                    r["query_embedding"] = np.asarray(qe, dtype=np.float32).flatten()

            from step5_generation import generate as step5_generate
            gen = step5_generate(
                self.G, self.index, results,
                backend=self.backend,
                model_name=self.model_name,
            )
            self.generation_done.emit(gen)

        except Exception as e:
            self.generation_error.emit(
                f"Generation failed:\n{e}\n\n{traceback.format_exc()}"
            )
        finally:
            sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# PDF Viewer
# ---------------------------------------------------------------------------

class PDFViewer(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._doc: Optional[object]       = None
        self._cur_page: int               = 0
        self._total_pages: int            = 0
        self._zoom: float                 = 1.5
        self._window_radius: int          = 2
        self._scroll_repositioning: bool  = False
        self._highlight_terms: list[str]  = []
        self._sentence_highlights: dict   = {}
        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        nav = QHBoxLayout()
        nav.setContentsMargins(6, 4, 6, 4)

        self._prev_btn   = QPushButton("◀ Prev")
        self._next_btn   = QPushButton("Next ▶")
        self._page_lbl   = QLabel("No document loaded")
        self._page_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._page_input = QLineEdit()
        self._page_input.setPlaceholderText("Page #")
        self._page_input.setFixedWidth(72)
        self._jump_btn   = QPushButton("Go")
        self._jump_btn.setFixedWidth(46)
        self._zoom_out   = QPushButton("−")
        self._zoom_in    = QPushButton("+")
        self._fit_btn    = QPushButton("Fit width")

        for btn in (self._prev_btn, self._next_btn, self._zoom_out, self._zoom_in):
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

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._pages_widget = QWidget()
        self._pages_layout = QVBoxLayout(self._pages_widget)
        self._pages_layout.setContentsMargins(0, 8, 0, 8)
        self._pages_layout.setSpacing(12)
        self._empty_label = QLabel("<br><br><br>Open a PDF to view it here.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pages_layout.addWidget(self._empty_label)
        self._scroll.setWidget(self._pages_widget)
        root.addWidget(self._scroll, 1)

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
        try:
            self.goto_page(int(self._page_input.text().strip()))
        except ValueError:
            pass

    def zoom_in(self) -> None:
        self._zoom = min(self._zoom * 1.25, 6.0)
        self._render()

    def zoom_out(self) -> None:
        self._zoom = max(self._zoom / 1.25, 0.25)
        self._render()

    def fit_to_width(self) -> None:
        if not self._doc:
            return
        pw = self._scroll.viewport().width() - 4
        page = self._doc[self._cur_page]
        if pw > 0 and page.rect.width > 0:
            self._zoom = pw / page.rect.width
        self._render()

    def set_sentence_highlights(self, page_to_sentences: dict) -> None:
        self._sentence_highlights = page_to_sentences or {}
        if self._doc:
            self._render()

    def set_highlight_terms(self, terms: list[str]) -> None:
        self._highlight_terms = [t.strip() for t in (terms or []) if t and t.strip()]
        if self._doc:
            self._render()

    def _render(self) -> None:
        if not self._doc or self._total_pages == 0:
            return
        self._clear_rendered_pages()
        start = max(0, self._cur_page - self._window_radius)
        end   = min(self._total_pages - 1, self._cur_page + self._window_radius)

        for page_idx in range(start, end + 1):
            page = self._doc[page_idx]
            mat  = fitz.Matrix(self._zoom, self._zoom)
            pix  = page.get_pixmap(matrix=mat)
            img  = QImage(pix.samples, pix.width, pix.height,
                          pix.stride, QImage.Format.Format_RGB888)
            pm   = QPixmap.fromImage(img)

            page_1based      = page_idx + 1
            highlight_rects  = []

            for sentence in self._sentence_highlights.get(page_1based, []):
                if not sentence or len(sentence) < 10:
                    continue
                phrases = re.findall(r'\b[A-Za-z][A-Za-z0-9\-]{4,}\b', sentence)
                for phrase in phrases[:3]:
                    try:
                        for r in page.search_for(phrase, quads=False):
                            highlight_rects.append((r, True))
                    except Exception:
                        pass

            for term in self._highlight_terms:
                if not term:
                    continue
                try:
                    for r in page.search_for(term, quads=False):
                        highlight_rects.append((r, False))
                except Exception:
                    pass

            if highlight_rects:
                painter = QPainter(pm)
                for rect, is_sentence in highlight_rects:
                    x0, y0 = int(rect.x0 * self._zoom), int(rect.y0 * self._zoom)
                    x1, y1 = int(rect.x1 * self._zoom), int(rect.y1 * self._zoom)
                    if is_sentence:
                        painter.setOpacity(0.35)
                        painter.setBrush(QColor(255, 140, 0))
                    else:
                        painter.setOpacity(0.45)
                        painter.setBrush(QColor(255, 255, 0))
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
        self._page_lbl.setText(f"Page {self._cur_page + 1} of {self._total_pages}")
        self._page_input.setText(str(self._cur_page + 1))
        self._update_nav_buttons()

    def _clear_rendered_pages(self) -> None:
        while self._pages_layout.count():
            item = self._pages_layout.takeAt(0)
            w    = item.widget()
            if w is not None:
                w.deleteLater()

    def _set_scroll_ratio(self, ratio: float) -> None:
        bar  = self._scroll.verticalScrollBar()
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
        bar  = self._scroll.verticalScrollBar()
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
            self._doc is not None and self._cur_page < self._total_pages - 1
        )
        loaded = self._doc is not None and self._total_pages > 0
        self._page_input.setEnabled(loaded)
        self._jump_btn.setEnabled(loaded)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._doc:
            QTimer.singleShot(50, self.fit_to_width)


# ---------------------------------------------------------------------------
# Saved KGs dialog
# ---------------------------------------------------------------------------

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
        self._load_btn    = QPushButton("Load selected")
        self._delete_btn  = QPushButton("Delete selected")
        self._refresh_btn = QPushButton("Refresh")
        btn_row.addWidget(self._load_btn)
        btn_row.addWidget(self._delete_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._refresh_btn)
        layout.addLayout(btn_row)

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
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
        return items[0].data(Qt.ItemDataRole.UserRole) if items else None

    def _on_load(self) -> None:
        meta = self._selected_meta()
        if not meta:
            QMessageBox.information(self, "No selection", "Please select a saved KG first.")
            return
        self.load_requested.emit(meta.get("save_dir", ""), meta.get("pdf_path", ""))
        self.accept()

    def _on_delete(self) -> None:
        meta = self._selected_meta()
        if not meta:
            QMessageBox.information(self, "No selection", "Please select a saved KG first.")
            return
        if QMessageBox.question(
            self, "Confirm delete",
            f"Delete saved KG for '{meta.get('pdf_stem','?')}'?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            app_delete_kg(meta.get("save_dir", ""))
            self._refresh()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

_RESULT_GEN_ANCHOR = "<!-- GEN_PLACEHOLDER -->"

_BUILD_BTN_STYLE = (
    "QPushButton { background:#2196F3; color:white; font-weight:bold; "
    "padding:8px; border-radius:5px; }"
    "QPushButton:hover { background:#1976D2; }"
    "QPushButton:disabled { background:#888; }"
)
_SEARCH_BTN_STYLE = (
    "QPushButton { background:#43A047; color:white; font-weight:bold; "
    "padding:6px 18px; border-radius:5px; }"
    "QPushButton:hover { background:#388E3C; }"
    "QPushButton:disabled { background:#888; }"
)


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NLP for PDFs")
        self.setMinimumSize(1100, 700)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        self._pdf_path:       Optional[Path]         = None
        self._G                                       = None
        self._index:          Optional[dict]          = None
        self._worker:         Optional[PipelineWorker] = None
        self._query_worker:   Optional[QueryWorker]   = None
        self._retrieval_html: str                     = ""
        self._log_lines:      list[str]               = []

        self._init_ui()
        self._set_mode("no_pdf")

    # ── UI construction ───────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        menu      = self.menuBar()
        file_menu = menu.addMenu("File")
        for label, shortcut, slot in [
            ("Open PDF …",       "Ctrl+O", self._on_open_pdf),
            ("View saved KGs …", "",       self._on_view_saved),
            ("Quit",             "Ctrl+Q", self.close),
        ]:
            act = QAction(label, self)
            if shortcut:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            file_menu.addAction(act)
            if label == "View saved KGs …":
                file_menu.addSeparator()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        left = self._build_left_panel()
        splitter.addWidget(left)

        self._pdf_viewer = PDFViewer()
        splitter.addWidget(self._pdf_viewer)

        splitter.setStretchFactor(0, 38)
        splitter.setStretchFactor(1, 62)
        splitter.setHandleWidth(4)

    # ── Left panel ────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        container.setMinimumWidth(340)
        container.setMaximumWidth(540)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── File bar (always visible) ─────────────────────────────────────
        file_bar = QFrame()
        file_bar.setFrameShape(QFrame.Shape.StyledPanel)
        file_bar.setStyleSheet("QFrame { background:#f0f0f0; border-bottom:1px solid #ccc; }")
        fl = QHBoxLayout(file_bar)
        fl.setContentsMargins(8, 6, 8, 6)
        fl.setSpacing(6)

        self._file_label = QLabel("No file loaded.")
        self._file_label.setWordWrap(True)
        self._file_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Preferred)
        fl.addWidget(self._file_label, 1)

        self._open_btn  = QPushButton("Open PDF …")
        self._close_btn = QPushButton("Close")
        self._open_btn.setFixedWidth(90)
        self._close_btn.setFixedWidth(60)
        self._open_btn.clicked.connect(self._on_open_pdf)
        self._close_btn.clicked.connect(self._on_close_pdf)
        fl.addWidget(self._open_btn)
        fl.addWidget(self._close_btn)

        file_bar.setFixedHeight(44)
        layout.addWidget(file_bar)

        # ── Stacked widget: build page  /  query page ─────────────────────
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        self._stack.addWidget(self._build_build_page())   # index 0
        self._stack.addWidget(self._build_query_page())   # index 1

        return container

    # ── Build page ────────────────────────────────────────────────────────

    def _build_build_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Options group
        opt_box = QGroupBox("Build options")
        ol = QVBoxLayout(opt_box)

        info_lbl = QLabel(
            "Embedding model: <b>all-mpnet-base-v2</b> (~420 MB)<br>"
            "Split model: <b>all-MiniLM-L6-v2</b> (~80 MB)"
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color:#555; font-size:11px;")
        ol.addWidget(info_lbl)

        layout.addWidget(opt_box)

        # Build log
        log_box = QGroupBox("Build log")
        ll = QVBoxLayout(log_box)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Consolas", 9))
        self._log_text.setPlaceholderText("Build output will appear here …")
        ll.addWidget(self._log_text)
        layout.addWidget(log_box, 1)   # stretches to fill

        # Build button
        self._build_btn = QPushButton("Build Knowledge Graph")
        self._build_btn.setStyleSheet(_BUILD_BTN_STYLE)
        self._build_btn.setFixedHeight(38)
        self._build_btn.clicked.connect(self._on_build_kg)
        layout.addWidget(self._build_btn)

        return page

    # ── Query page ────────────────────────────────────────────────────────

    def _build_query_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Results browser ───────────────────────────────────────────────
        self._results = QTextBrowser()
        self._results.setOpenLinks(False)
        self._results.anchorClicked.connect(self._on_result_link)
        self._results.setPlaceholderText(
            "Query results will appear here.\n"
            "Click a page link to jump to it in the viewer.")
        self._results.setStyleSheet(
            "QTextBrowser { border:2px solid #7B1FA2; border-radius:8px; "
            "background:#fafafa; padding:6px; }"
        )
        layout.addWidget(self._results, 1)

        # ── Query input area ──────────────────────────────────────────────
        query_frame = QFrame()
        query_frame.setStyleSheet(
            "QFrame { background:#f5f5f5; border-radius:8px; "
            "border:1px solid #ddd; }"
        )
        ql = QVBoxLayout(query_frame)
        ql.setContentsMargins(8, 6, 8, 6)
        ql.setSpacing(4)

        input_row = QHBoxLayout()
        self._query_input = QLineEdit()
        self._query_input.setPlaceholderText("Type a query and press Enter …")
        self._query_input.setStyleSheet(
            "QLineEdit { border:1px solid #bbb; border-radius:4px; padding:5px; }"
        )
        self._query_input.returnPressed.connect(self._on_search)
        self._search_btn = QPushButton("Search")
        self._search_btn.setStyleSheet(_SEARCH_BTN_STYLE)
        self._search_btn.setFixedHeight(32)
        self._search_btn.clicked.connect(self._on_search)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedHeight(32)
        self._clear_btn.setFixedWidth(56)
        self._clear_btn.clicked.connect(self._on_clear)
        input_row.addWidget(self._query_input, 1)
        input_row.addWidget(self._search_btn)
        input_row.addWidget(self._clear_btn)
        ql.addLayout(input_row)

        layout.addWidget(query_frame)

        # ── Collapsed settings strip ──────────────────────────────────────
        strip = QFrame()
        strip.setStyleSheet(
            "QFrame { background:#eeeeee; border-radius:6px; "
            "border:1px solid #ccc; }"
        )
        sl = QVBoxLayout(strip)
        sl.setContentsMargins(8, 4, 8, 4)
        sl.setSpacing(4)

        # Row 1: generative toggle + model selector
        gen_row = QHBoxLayout()
        self._gen_chk = QCheckBox("Generate response")
        self._gen_chk.setChecked(False)
        self._gen_chk.setToolTip(
            "Uses step5 to generate a fluent answer from retrieved chunks.\n"
            "Requires Ollama running locally (or llama_cpp)."
        )
        gen_row.addWidget(self._gen_chk)
        gen_row.addStretch()
        gen_row.addWidget(QLabel("Backend:"))
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["ollama", "llama_cpp"])
        self._backend_combo.setFixedWidth(90)
        gen_row.addWidget(self._backend_combo)
        gen_row.addWidget(QLabel("Model:"))
        self._model_input = QLineEdit("qwen2.5:3b")
        self._model_input.setFixedWidth(110)
        gen_row.addWidget(self._model_input)
        sl.addLayout(gen_row)

        # Row 2: build log toggle + save + view saved
        ctrl_row = QHBoxLayout()
        self._log_toggle_btn = QPushButton("▶ Build log")
        self._log_toggle_btn.setFlat(True)
        self._log_toggle_btn.setStyleSheet("font-size:11px; color:#555;")
        self._log_toggle_btn.clicked.connect(self._toggle_build_log)
        ctrl_row.addWidget(self._log_toggle_btn)
        ctrl_row.addStretch()
        self._save_btn = QPushButton("Save KG")
        self._save_btn.setFixedWidth(72)
        self._save_btn.clicked.connect(self._on_save_kg)
        self._view_saved_btn = QPushButton("View saved …")
        self._view_saved_btn.setFixedWidth(90)
        self._view_saved_btn.clicked.connect(self._on_view_saved)
        ctrl_row.addWidget(self._save_btn)
        ctrl_row.addWidget(self._view_saved_btn)
        sl.addLayout(ctrl_row)

        # Collapsible build log (hidden by default in query mode)
        self._mini_log = QTextEdit()
        self._mini_log.setReadOnly(True)
        self._mini_log.setFont(QFont("Consolas", 8))
        self._mini_log.setFixedHeight(80)
        self._mini_log.setVisible(False)
        sl.addWidget(self._mini_log)

        layout.addWidget(strip)

        return page

    # ── Mode switching ────────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        """
        Modes:
          no_pdf       — nothing loaded
          pdf_loaded   — PDF open, KG not built
          building     — KG being built
          kg_ready     — KG ready, query mode
          build_failed — build failed
        """
        self._mode = mode
        no_pdf   = mode == "no_pdf"
        building = mode == "building"
        kg_ready = mode == "kg_ready"

        # File bar buttons
        self._open_btn.setEnabled(not building)
        self._close_btn.setEnabled(not no_pdf and not building)

        if kg_ready:
            self._stack.setCurrentIndex(1)
            self._search_btn.setEnabled(True)
            self._query_input.setEnabled(True)
            self._save_btn.setEnabled(True)
        else:
            self._stack.setCurrentIndex(0)
            if building:
                self._build_btn.setEnabled(False)
                self._build_btn.setText("Building …")
            elif mode in ("pdf_loaded", "build_failed"):
                self._build_btn.setEnabled(True)
                self._build_btn.setText("Build Knowledge Graph")
            else:
                self._build_btn.setEnabled(False)
                self._build_btn.setText("Build Knowledge Graph")

    # ── File management ───────────────────────────────────────────────────

    def _check_permission(self, path: Path) -> bool:
        try:
            path.relative_to(APP_DIR)
            return True
        except ValueError:
            reply = QMessageBox.question(
                self, "File access permission",
                f"Allow access to:\n\n{path}\n\n"
                "(File is outside the application folder.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            return reply == QMessageBox.StandardButton.Yes

    def _on_open_pdf(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", str(Path.home()),
            "PDF files (*.pdf);;All files (*.*)",
        )
        if not path_str:
            return
        path = Path(path_str).resolve()
        if not self._check_permission(path):
            return

        self._pdf_path = path
        self._G        = None
        self._index    = None
        self._results.clear()
        self._clear_log()
        self._file_label.setText(f"<b>{path.name}</b>")

        self._pdf_viewer.load_pdf(str(path))

        saved = app_load_kg(path)
        if saved:
            G, index, meta = saved
            reply = QMessageBox.question(
                self, "Saved KG found",
                f"Found a saved KG for this PDF\n"
                f"(saved {meta.get('saved_at','?')[:19].replace('T',' ')}).\n\n"
                "Load the saved version?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._G     = G
                self._index = index
                self._append_log("Loaded saved knowledge graph.")
                self._set_mode("kg_ready")
                self._status.showMessage(
                    f"Loaded saved KG — {G.number_of_nodes()} nodes")
                return

        self._set_mode("pdf_loaded")
        self._status.showMessage(f"PDF loaded: {path.name}")

    def _on_close_pdf(self) -> None:
        self._pdf_path = None
        self._G        = None
        self._index    = None
        self._file_label.setText("No file loaded.")
        self._results.clear()
        self._clear_log()
        self._pdf_viewer.close_pdf()
        self._set_mode("no_pdf")
        self._status.showMessage("Ready")

    # ── Build ─────────────────────────────────────────────────────────────

    def _on_build_kg(self) -> None:
        if not self._pdf_path:
            return
        self._clear_log()
        self._set_mode("building")
        self._status.showMessage("Building knowledge graph …")

        self._worker = PipelineWorker(str(self._pdf_path))
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_build_finished)
        self._worker.error.connect(self._on_build_error)
        self._worker.start()

    def _on_build_finished(self, G, index: dict) -> None:
        self._G     = G
        self._index = index
        self._mini_log.setPlainText(self._log_text.toPlainText())
        self._set_mode("kg_ready")
        self._status.showMessage(
            f"KG ready — {G.number_of_nodes()} nodes. Enter a query below.")
        self._query_input.setFocus()

    def _on_build_error(self, msg: str) -> None:
        self._set_mode("build_failed")
        self._status.showMessage("Build failed — see log.")
        self._append_log("--- ERROR ---")
        self._append_log(msg)
        QMessageBox.critical(self, "Build error",
                             "An error occurred. See the build log for details.")

    # ── Log helpers ───────────────────────────────────────────────────────

    def _clear_log(self) -> None:
        self._log_lines.clear()
        self._log_text.clear()
        self._mini_log.clear()

    def _append_log(self, msg: str) -> None:
        clean = msg.strip()
        if not clean:
            return
        self._log_lines.append(clean)
        self._log_text.setPlainText("\n".join(self._log_lines))
        self._log_text.moveCursor(QTextCursor.MoveOperation.End)
        # Also update mini log if in query mode
        if self._mode == "kg_ready":
            self._mini_log.setPlainText("\n".join(self._log_lines[-30:]))

    def _toggle_build_log(self) -> None:
        visible = not self._mini_log.isVisible()
        self._mini_log.setVisible(visible)
        self._log_toggle_btn.setText("▼ Build log" if visible else "▶ Build log")

    # ── Query ─────────────────────────────────────────────────────────────

    def _on_search(self) -> None:
        if not self._G or not self._index:
            return
        query_text = self._query_input.text().strip()
        if not query_text:
            return

        self._search_btn.setEnabled(False)
        self._query_input.setEnabled(False)
        self._status.showMessage("Searching …")
        self._results.setHtml(
            f"<p style='color:#555;'>⏳ Retrieving results for: "
            f"<b>{_esc(query_text)}</b> …</p>"
        )

        use_gen    = self._gen_chk.isChecked()
        backend    = self._backend_combo.currentText()
        model_name = self._model_input.text().strip() or "qwen2.5:3b"

        self._query_worker = QueryWorker(
            self._G, self._index, query_text,
            top_k=5,
            use_generative=use_gen,
            backend=backend,
            model_name=model_name,
        )
        self._query_worker.results_ready.connect(
            lambda r: self._on_results_ready(r, query_text, use_gen)
        )
        self._query_worker.generation_done.connect(self._on_generation_done)
        self._query_worker.generation_error.connect(self._on_generation_error)
        self._query_worker.log.connect(self._append_log)
        self._query_worker.finished.connect(self._on_query_worker_finished)
        self._query_worker.start()

    def _on_results_ready(
        self, results: list, query_text: str, use_gen: bool
    ) -> None:
        self._status.showMessage(
            f"Found {len(results)} result(s) for: {query_text!r}"
        )
        self._retrieval_html = self._format_retrieval_html(results, query_text)

        if use_gen and results:
            gen_placeholder = (
                "<div id='gen' style='margin-top:12px; padding:10px; "
                "background:#F3E5F5; border-left:4px solid #7B1FA2; "
                "border-radius:0 6px 6px 0;'>"
                "<b style='color:#7B1FA2;'>💬 Generative response</b><br>"
                "<span style='color:#888;'>⏳ Generating …</span>"
                "</div>"
            )
            full_html = self._retrieval_html + gen_placeholder
        else:
            full_html = self._retrieval_html

        self._results.setHtml(full_html)
        self._results.moveCursor(QTextCursor.MoveOperation.Start)

        # Highlight PDF
        terms = _extract_terms(query_text)
        self._pdf_viewer.set_highlight_terms(terms)
        page_to_sents: dict[int, list[str]] = {}
        for r in results:
            pg    = r.get("page")
            sents = r.get("sentences", [])
            texts = [s["text"] for s in sents] if sents and isinstance(sents[0], dict) else sents
            if pg and texts:
                page_to_sents[pg] = texts[:3]
        self._pdf_viewer.set_sentence_highlights(page_to_sents)

        if results:
            self._pdf_viewer.goto_page(results[0]["page"])

    def _on_generation_done(self, gen: dict) -> None:
        stats   = gen.get("citation_stats", {})
        pages   = gen.get("source_pages", [])
        model   = gen.get("model_name", "")
        cited   = stats.get("cited", 0)
        total   = stats.get("total_claims", 0)
        rate    = stats.get("cite_rate", 0)
        answer_html = self._format_generation_answer_html(gen)

        gen_html = (
            "<div style='margin-top:12px; padding:10px; "
            "background:#F3E5F5; border-left:4px solid #7B1FA2; "
            "border-radius:0 6px 6px 0;'>"
            f"<b style='color:#7B1FA2;'>💬 Generative response</b> "
            f"<span style='color:#888; font-size:11px;'>"
            f"[{_esc(model)} | pages {pages} | "
            f"{cited}/{total} claims cited ({rate:.0%})]</span><br><br>"
            f"<span style='font-size:13px; line-height:1.5;'>{answer_html}</span>"
            "</div>"
        )
        self._results.setHtml(self._retrieval_html + gen_html)
        self._results.moveCursor(QTextCursor.MoveOperation.End)
        self._status.showMessage("Generation complete.")

    def _on_generation_error(self, msg: str) -> None:
        error_html = (
            "<div style='margin-top:12px; padding:10px; "
            "background:#FFEBEE; border-left:4px solid #C62828; "
            "border-radius:0 6px 6px 0;'>"
            "<b style='color:#C62828;'>⚠ Generation failed</b><br>"
            f"<span style='font-size:11px; color:#555;'>{_esc(msg[:300])}</span>"
            "</div>"
        )
        self._results.setHtml(self._retrieval_html + error_html)
        self._append_log(f"Generation error: {msg[:200]}")

    def _on_query_worker_finished(self) -> None:
        self._search_btn.setEnabled(True)
        self._query_input.setEnabled(True)
        self._query_input.setFocus()

    def _on_clear(self) -> None:
        self._results.clear()
        self._query_input.clear()
        self._retrieval_html = ""
        self._status.showMessage("Cleared.")

    def _on_result_link(self, url: QUrl) -> None:
        section = ""
        raw_page = ""
        if url.scheme() == "page":
            raw_page = url.host() or url.path().lstrip("/")
        elif url.scheme() == "cite":
            raw_page = url.host() or url.path().lstrip("/")
            query = url.query()
            if query:
                m = re.search(r"(?:^|&)section=([^&]+)", query)
                if m:
                    section = unquote(m.group(1))
        if not raw_page:
            m = re.search(r"(?:^|://|:)(\d+)(?:/)?$", url.toString())
            if m:
                raw_page = m.group(1)
        try:
            page_num = int(raw_page)
            self._pdf_viewer.goto_page(page_num)
            if section:
                self._status.showMessage(f"Jumped to page {page_num} — {section}")
            else:
                self._status.showMessage(f"Jumped to page {page_num}")
        except (TypeError, ValueError):
            pass

    def _format_generation_answer_html(self, gen: dict) -> str:
        """Render answer text and make citations clickable page links."""
        cited_claims = gen.get("cited_claims") or []
        if cited_claims:
            rendered: list[str] = []
            for claim in cited_claims:
                claim_text = str(claim.get("text", "")).strip()
                if not claim_text:
                    continue
                page = claim.get("page")
                section = str(claim.get("section", "")).strip()
                line_html = _esc(claim_text)
                if page is not None:
                    section_q = quote(section, safe="")
                    line_html += (
                        f' <a href="cite:{page}?section={section_q}" '
                        f'style="color:#1976D2; text-decoration:none; font-weight:bold;">'
                        f'[p. {page}]</a>'
                    )
                rendered.append(line_html)
            if rendered:
                return "<br>".join(rendered)

        answer = str(gen.get("answer", "")).strip()
        if not answer:
            return "<i style='color:#777;'>No generated answer.</i>"
        safe_answer = _esc(answer)
        return re.sub(
            r"\[p\.\s*(\d+)\]",
            lambda m: (
                f'<a href="cite:{m.group(1)}?section=" '
                f'style="color:#1976D2; text-decoration:none; font-weight:bold;">'
                f'[p. {m.group(1)}]</a>'
            ),
            safe_answer,
        ).replace("\n", "<br>")

    # ── Results HTML ──────────────────────────────────────────────────────

    def _format_retrieval_html(self, results: list, query_text: str) -> str:
        if not results:
            return (
                f"<p style='color:#777;'><i>No results found for: "
                f"<b>{_esc(query_text)}</b></i></p>"
            )

        parts = [
            f"<p style='margin:0 0 8px 0;'>"
            f"<b>Query:</b> {_esc(query_text)}&nbsp;&nbsp;"
            f"<span style='color:#888; font-size:11px;'>"
            f"Top {len(results)} results</span></p>"
        ]

        for r in results:
            page_num  = r.get("page")
            page_rng  = r.get("page_range", str(page_num))
            node_type = r.get("node_type", "SemanticChunk")
            score     = r.get("score", 0.0)
            section   = r.get("section", "")
            table_cap = r.get("table_caption", "")
            text      = r.get("chunk_text", r.get("context_block", ""))
            sect_sim  = r.get("section_score", 0.0)

            colour = "#1565C0" if node_type == "SemanticChunk" else "#6A1B9A"
            type_label = "Chunk" if node_type == "SemanticChunk" else "Table"

            page_link = (
                f'<a href="page:{page_num}" '
                f'style="color:#1976D2; font-weight:bold; text-decoration:none;">'
                f'📄 Page {page_rng}</a>'
            ) if page_num is not None else "—"

            block = (
                f'<div style="margin:0 0 8px 0; padding:8px 10px; '
                f'border-left:4px solid {colour}; '
                f'background:#f5f7fa; border-radius:0 5px 5px 0;">'
                f'<b>#{r["rank"]}</b>&nbsp; '
                f'Score: <span style="color:{colour}; font-weight:bold;">'
                f'{score:.4f}</span>&nbsp;|&nbsp;{page_link}'
                f'&nbsp;|&nbsp;<span style="color:#888; font-size:10px;">'
                f'[{type_label}]</span>'
            )

            if section:
                block += (
                    f'<br><span style="font-size:11px; color:#555;">'
                    f'Section: {_esc(section)}'
                    f'&nbsp;<span style="color:#aaa;">'
                    f'(sim={sect_sim:.3f})</span></span>'
                )
            if table_cap:
                block += (
                    f'<br><span style="font-size:11px; color:#6A1B9A;">'
                    f'Table: {_esc(table_cap)}</span>'
                )

            dist = r.get("page_distribution", {})
            if len(dist) > 1:
                dist_str = ", ".join(
                    f"p{pg}:{pct:.0%}" for pg, pct in sorted(dist.items())
                )
                block += (
                    f'<br><span style="font-size:10px; color:#888;">'
                    f'Span: {dist_str}</span>'
                )

            preview = text[:280] + ("…" if len(text) > 280 else "")
            block += (
                f'<br><span style="font-size:12px; color:#333; '
                f'line-height:1.4;">{_esc(preview)}</span>'
                f'</div>'
            )
            parts.append(block)

        return "\n".join(parts)

    # ── Save / load KGs ──────────────────────────────────────────────────

    def _on_save_kg(self) -> None:
        if not self._G or not self._index or not self._pdf_path:
            QMessageBox.information(self, "Nothing to save",
                                    "Build a knowledge graph first.")
            return
        save_dir = app_save_kg(self._pdf_path, self._G, self._index,
                               model_name="all-mpnet-base-v2")
        self._status.showMessage(f"KG saved → {save_dir.name}")
        QMessageBox.information(self, "Saved",
                                f"Knowledge graph saved to:\n{save_dir}")

    def _on_view_saved(self) -> None:
        dlg = SavedKGsDialog(self)
        dlg.load_requested.connect(self._on_load_saved_kg)
        dlg.exec()

    def _on_load_saved_kg(self, save_dir_str: str, pdf_path_str: str) -> None:
        pdf_path = Path(pdf_path_str) if pdf_path_str else None
        saved    = None

        if pdf_path and pdf_path.exists():
            saved = app_load_kg(pdf_path)
        else:
            # Try loading directly from save_dir without hash check
            saved = _load_kg_from_dir(Path(save_dir_str))

        if saved is None:
            QMessageBox.warning(
                self, "Load failed",
                "Could not load the saved KG. "
                "The PDF may have changed or files may be missing.")
            return

        G, index, meta = saved
        self._G         = G
        self._index     = index

        if pdf_path and pdf_path.exists():
            if self._check_permission(pdf_path):
                self._pdf_path = pdf_path
                self._file_label.setText(f"<b>{pdf_path.name}</b>")
                self._pdf_viewer.load_pdf(str(pdf_path))
        else:
            path_str, _ = QFileDialog.getOpenFileName(
                self, "Locate PDF for this KG",
                str(Path.home()), "PDF files (*.pdf)",
            )
            if path_str:
                p = Path(path_str).resolve()
                if self._check_permission(p):
                    self._pdf_path = p
                    self._file_label.setText(f"<b>{p.name}</b>")
                    self._pdf_viewer.load_pdf(str(p))

        self._results.clear()
        self._append_log("Loaded saved knowledge graph.")
        self._set_mode("kg_ready")
        self._status.showMessage(f"Loaded KG: {G.number_of_nodes()} nodes")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Minimal HTML escaping for display."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _extract_terms(query_text: str) -> list[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "what", "which", "with",
    }
    tokens = re.findall(r"[A-Za-z0-9\-_/]+", query_text.lower())
    terms, seen = [], set()
    for tok in tokens:
        if len(tok) < 3 or tok in stop:
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
    return terms[:12]


def _load_kg_from_dir(save_dir: Path):
    """Load G and index directly from a save_dir (no hash check)."""
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

    from step4_persistence import _tokenize
    try:
        from rank_bm25 import BM25Okapi
        ids   = idx_data.get("ids", [])
        types = idx_data.get("types", [])
        texts = [
            G.nodes[nid].get("text", "") if G.has_node(nid) else ""
            for nid in ids
        ]
        tokenized = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized) if tokenized else None
    except Exception:
        bm25  = None
        ids   = idx_data.get("ids", [])
        types = idx_data.get("types", [])
        texts = []

    index = {
        "ids":          ids,
        "types":        types,
        "texts":        texts,
        "embeddings":   embeddings,
        "corpus_mean":  corpus_mean,
        "bm25":         bm25,
    }
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return G, index, meta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("NLPforPDFs")
    app.setApplicationDisplayName("NLP for PDFs")

    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    font = app.font()
    font.setPointSize(max(font.pointSize(), 10))
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()