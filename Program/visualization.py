"""
Step 6 — Visualization
======================
Renders the knowledge graph using:
  - PyQt6 desktop UI for interactive exploration
  - Matplotlib local viewer with click-to-inspect behavior
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pickle
import sys
import argparse

PROGRAM_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = PROGRAM_DIR.parent
for _p in (str(PROJECT_DIR), str(PROGRAM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from Program.step4_persistence import list_saved_kgs
except ImportError:
    from step4_persistence import list_saved_kgs


def _subsample_graph(G, max_nodes):
    if max_nodes is None:
        return G
    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        print(f"  Graph has {len(nodes)} nodes — showing first {max_nodes} for clarity.")
        return G.subgraph(nodes[:max_nodes]).copy()
    return G


def visualize_graph_pyqt(
    G,
    max_nodes=None,
    max_edges=900,
):
    """
    Launch a PyQt6 desktop window with:
      - graph canvas (pan/zoom)
      - clickable nodes
      - right-side panel showing full node attributes
    """
    try:
        from PyQt6.QtCore import Qt  # type: ignore[import-not-found]
        from PyQt6.QtGui import QBrush, QColor, QFont, QPen  # type: ignore[import-not-found]
        from PyQt6.QtWidgets import (  # type: ignore[import-not-found]
            QApplication,
            QGraphicsEllipseItem,
            QGraphicsLineItem,
            QGraphicsScene,
            QGraphicsSimpleTextItem,
            QGraphicsView,
            QMainWindow,
            QPlainTextEdit,
            QSplitter,
            QWidget,
            QVBoxLayout,
            QLabel,
        )
    except ImportError:
        print("  PyQt6 is not installed. Install with: pip install pyqt6")
        return False

    subG = _subsample_graph(G, max_nodes)
    pos = nx.spring_layout(subG, seed=42, k=1.8)
    scale = 900.0
    node_xy = {n: (float(x) * scale, float(y) * scale) for n, (x, y) in pos.items()}

    type_colors = {
        "PDF": "#355C7D",
        "SemanticChunk": "#2A9D8F",
        "Sentence": "#7FB3D5",
        "Section": "#7F77DD",
        "Table": "#1D9E75",
        "Record": "#3B8BD4",
        "Value": "#EF9F27",
        "Condition": "#D85A30",
        "Page": "#4C78A8",
        "Row": "#72B7B2",
        "Column": "#54A24B",
        "Cell": "#E45756",
        "Triple": "#B279A2",
        "Image": "#F58518",
    }

    app = QApplication.instance() or QApplication(sys.argv)

    _IGNORE_TRANSFORM = (
        QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations
    )
    _SELECTABLE = QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable

    class NodeItem(QGraphicsEllipseItem):
        """
        Node drawn at a fixed fraction of the viewport (zoom-independent
        on-screen size). Scene position (x, y) is the graph layout center.
        """

        def __init__(self, node_id, attrs, x, y, on_click):
            super().__init__(-1, -1, 2, 2)
            self.node_id = node_id
            self.attrs = attrs
            self.on_click = on_click
            self.scene_x = x
            self.scene_y = y
            self.setPos(x, y)
            ntype = attrs.get("type", "")
            fill = QColor(type_colors.get(ntype, "#888888"))
            self.setBrush(QBrush(fill))
            self.setPen(QPen(QColor("#222222"), 1))
            self.setZValue(2)
            self.setFlag(_SELECTABLE, True)
            self.setFlag(_IGNORE_TRANSFORM, True)
            self.setToolTip(str(attrs.get("label", node_id)))

        def set_viewport_radius(self, radius: float, pen_width: float) -> None:
            r = max(2.0, float(radius))
            self.setRect(-r, -r, 2 * r, 2 * r)
            pen = self.pen()
            pen.setWidthF(max(0.5, float(pen_width)))
            self.setPen(pen)

        def mousePressEvent(self, event):  # noqa: N802
            self.on_click(self.node_id, self.attrs)
            super().mousePressEvent(event)

    class LabelItem(QGraphicsSimpleTextItem):
        """Label that tracks a node center and scales with viewport."""

        def __init__(self, text: str, scene_x: float, scene_y: float):
            super().__init__(text)
            self.scene_x = scene_x
            self.scene_y = scene_y
            self.setBrush(QBrush(QColor("#202020")))
            self.setZValue(3)
            self.setFlag(_IGNORE_TRANSFORM, True)

        def layout_near_node(self, radius: float, font_px: int) -> None:
            r = max(2.0, float(radius))
            font = QFont()
            font.setPixelSize(max(6, int(font_px)))
            self.setFont(font)
            self.setPos(self.scene_x + r + 2, self.scene_y - r)

    scene = QGraphicsScene()
    scene.setBackgroundBrush(QBrush(QColor("#fafafa")))

    # Draw edges first (behind nodes), capped for responsiveness.
    edge_pen = QPen(QColor("#b0b0b0"))
    edge_pen.setWidthF(0.8)
    edge_pen.setCosmetic(True)
    edge_count = 0
    if isinstance(subG, (nx.MultiDiGraph, nx.MultiGraph)):
        edges_iter = subG.edges(keys=True, data=True)
        for u, v, _k, _attrs in edges_iter:
            if edge_count >= max_edges:
                break
            x1, y1 = node_xy[u]
            x2, y2 = node_xy[v]
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(edge_pen)
            line.setZValue(1)
            scene.addItem(line)
            edge_count += 1
    else:
        for u, v, _attrs in subG.edges(data=True):
            if edge_count >= max_edges:
                break
            x1, y1 = node_xy[u]
            x2, y2 = node_xy[v]
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(edge_pen)
            line.setZValue(1)
            scene.addItem(line)
            edge_count += 1
    if edge_count >= max_edges:
        print(f"  Edge cap reached ({max_edges}); showing subset for responsiveness.")

    details = QPlainTextEdit()
    details.setReadOnly(True)
    details.setPlainText("Click a node to inspect its stored attributes.")

    def on_node_click(node_id, attrs):
        lines = [f"node: {node_id}"]
        for k, v in attrs.items():
            lines.append(f"{k}: {v}")
        details.setPlainText("\n".join(lines))

    # Draw nodes and short labels (viewport-relative size; see GraphView).
    scaled_pairs: list[tuple[NodeItem, LabelItem]] = []
    for n, attrs in subG.nodes(data=True):
        x, y = node_xy[n]
        node_item = NodeItem(n, attrs, x, y, on_node_click)
        scene.addItem(node_item)

        label = str(attrs.get("label", n))[:24]
        text_item = LabelItem(label, x, y)
        scene.addItem(text_item)
        scaled_pairs.append((node_item, text_item))

    class GraphView(QGraphicsView):
        """Zoom/pan graph; node disks + labels keep a constant viewport fraction."""

        def __init__(self, sc, pairs):
            super().__init__(sc)
            self._scaled_pairs = pairs

        def _refresh_scaled_items(self) -> None:
            vp = self.viewport()
            d = float(min(max(vp.width(), 1), max(vp.height(), 1)))
            # Fractions of min(viewport w, h): stable across window shapes.
            radius = max(3.0, d * 0.011)
            pen_w = max(0.6, d * 0.0011)
            font_px = max(7, int(d * 0.016))
            for node_item, text_item in self._scaled_pairs:
                node_item.set_viewport_radius(radius, pen_w)
                text_item.layout_near_node(radius, font_px)

        def resizeEvent(self, event):  # noqa: N802
            super().resizeEvent(event)
            self._refresh_scaled_items()

        def showEvent(self, event):  # noqa: N802
            super().showEvent(event)
            self._refresh_scaled_items()

        def wheelEvent(self, event):  # noqa: N802
            # Force wheel to zoom only (no scrollbar-based panning on scroll).
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            self._refresh_scaled_items()
            event.accept()

    view = GraphView(scene, scaled_pairs)
    view.setRenderHint(view.renderHints())
    view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
    view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
    view.scale(0.7, 0.7)

    window = QMainWindow()
    window.setWindowTitle("KG Viewer (PyQt6)")
    window.resize(1500, 950)

    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.addWidget(QLabel("Node details"))
    right_layout.addWidget(details)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.addWidget(view)
    splitter.addWidget(right_panel)
    splitter.setSizes([1100, 400])

    window.setCentralWidget(splitter)
    window.show()
    app.exec()
    return True


def visualize_graph(G, max_nodes=1000, output_path=None):
    """
    Draw the knowledge graph, colour-coded by node type.

    Args:
        G:           NetworkX graph from build_knowledge_graph()
        max_nodes:   Cap on nodes displayed (default 80)
        output_path: If provided, save image to this path instead of showing
    """
    type_colors = {
        "PDF":       "#355C7D",
        "SemanticChunk": "#2A9D8F",
        "Sentence":  "#7FB3D5",
        "Section":   "#7F77DD",
        "Table":     "#1D9E75",
        "Record":    "#3B8BD4",
        "Value":     "#EF9F27",
        "Condition": "#D85A30",
    }

    subG = _subsample_graph(G, max_nodes)

    colors = [type_colors.get(subG.nodes[n].get("type", ""), "#888") for n in subG.nodes()]
    labels = {n: subG.nodes[n].get("label", n)[:28] for n in subG.nodes()}

    fig, ax = plt.subplots(figsize=(16, 10))
    pos = nx.spring_layout(subG, seed=42, k=1.8)
    node_list = list(subG.nodes())
    node_collection = nx.draw_networkx_nodes(
        subG, pos, nodelist=node_list, node_color=colors, node_size=600, alpha=0.9, ax=ax
    )
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=7, ax=ax)
    nx.draw_networkx_edges(
        subG, pos, alpha=0.4, arrows=True,
        arrowsize=12, edge_color="#aaa",
        connectionstyle="arc3,rad=0.1", ax=ax
    )

    legend = [mpatches.Patch(color=c, label=t) for t, c in type_colors.items()]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    ax.set_title("NLP for PDFs Knowledge Graph", fontsize=14)
    ax.axis("off")

    info_text = ax.text(
        0.01, 0.01,
        "Click a node to inspect its stored attributes.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#cccccc", "alpha": 0.95},
    )

    # Make nodes clickable in interactive mode.
    node_collection.set_picker(True)
    node_collection.set_pickradius(8)

    def _format_node_info(node_id):
        attrs = subG.nodes[node_id]
        lines = [f"node: {node_id}"]
        for k, v in attrs.items():
            value = str(v).replace("\n", " ").strip()
            if len(value) > 140:
                value = value[:137] + "..."
            lines.append(f"{k}: {value}")
        return "\n".join(lines[:14])

    def _on_pick(event):
        if event.artist is not node_collection or not getattr(event, "ind", None):
            return
        idx = int(event.ind[0])
        if idx < 0 or idx >= len(node_list):
            return
        selected_node = node_list[idx]
        info_text.set_text(_format_node_info(selected_node))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", _on_pick)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Graph image saved to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a saved knowledge graph")
    parser.add_argument(
        "--kg-dir",
        type=str,
        default="",
        help="Path to a saved KG directory containing graph.pkl",
    )
    parser.add_argument(
        "--search-dir",
        type=str,
        default="",
        help="Directory to scan for saved KGs when --kg-dir is omitted",
    )
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=10000)
    args = parser.parse_args()

    graph_path: Path | None = None

    if args.kg_dir:
        candidate = Path(args.kg_dir).resolve() / "graph.pkl"
        if candidate.exists():
            graph_path = candidate
    else:
        project_root = Path(__file__).resolve().parents[1]
        search_root = Path(args.search_dir).resolve() if args.search_dir else project_root
        saved = list_saved_kgs(search_root)
        if saved:
            # Prefer the Attention Is All You Need bundle when present.
            preferred_dir: Path | None = None
            for item in saved:
                save_dir = Path(item.get("save_dir", "")).resolve()
                if save_dir.name == "attention_is_all_you_need_kg":
                    preferred_dir = save_dir
                    break

            chosen_dir = preferred_dir or Path(saved[0]["save_dir"]).resolve()
            candidate = chosen_dir / "graph.pkl"
            if candidate.exists():
                graph_path = candidate

    if graph_path is None:
        print("No saved KG found. Pass --kg-dir or build/save a KG first.")
        sys.exit(0)

    try:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
    except Exception as e:
        print(f"Failed to load saved KG from {graph_path}: {e}")
        sys.exit(1)

    print(f"Loaded KG from: {graph_path}")
    launched = visualize_graph_pyqt(
        G,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
    )
    if not launched:
        print("  Failed to launch PyQt6 viewer.")
        sys.exit(1)