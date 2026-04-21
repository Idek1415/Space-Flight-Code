"""
Step 6 — Visualization
=======================
Renders the knowledge graph using matplotlib and NetworkX.
Nodes are colour-coded by type. Large graphs are subsampled
to max_nodes for readability.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_graph(G, max_nodes=80, output_path=None):
    """
    Draw the knowledge graph, colour-coded by node type.

    Args:
        G:           NetworkX graph from build_knowledge_graph()
        max_nodes:   Cap on nodes displayed (default 80)
        output_path: If provided, save image to this path instead of showing
    """
    type_colors = {
        "Section":   "#7F77DD",
        "Table":     "#1D9E75",
        "Record":    "#3B8BD4",
        "Value":     "#EF9F27",
        "Condition": "#D85A30",
    }

    # Subsample if graph is large
    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        print(f"  Graph has {len(nodes)} nodes — showing first {max_nodes} for clarity.")
        nodes = nodes[:max_nodes]
        subG = G.subgraph(nodes)
    else:
        subG = G

    colors = [type_colors.get(subG.nodes[n].get("type", ""), "#888") for n in subG.nodes()]
    labels = {n: subG.nodes[n].get("label", n)[:28] for n in subG.nodes()}

    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(subG, seed=42, k=1.8)
    nx.draw_networkx_nodes(subG, pos, node_color=colors, node_size=600, alpha=0.9)
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=7)
    nx.draw_networkx_edges(subG, pos, alpha=0.4, arrows=True,
                           arrowsize=12, edge_color="#aaa",
                           connectionstyle="arc3,rad=0.1")

    legend = [mpatches.Patch(color=c, label=t) for t, c in type_colors.items()]
    plt.legend(handles=legend, loc="upper left", fontsize=9)
    plt.title("NLP for PDFs Knowledge Graph", fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Graph image saved to: {output_path}")
    else:
        plt.show()
