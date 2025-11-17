import os
from graphviz import Digraph

# Use Graphviz locally (no permanent PATH change)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

dot = Digraph(format="png", comment="Supplementary Figure S1b - Dataset Preprocessing")

# -------------------- Graph layout --------------------
dot.attr(
    rankdir="TB",          # Top -> Bottom
    bgcolor="white",
    splines="spline",
    ranksep="0.7",
    nodesep="0.6",
    pad="0.06",
    dpi="700"
)

# -------------------- Node style --------------------
dot.attr(
    "node",
    shape="box",
    style="rounded,filled",
    color="#9E9E9E",
    fontname="Helvetica",
    fontcolor="#222222",
    fontsize="10",
    margin="0.16,0.12",
    width="3.0",
    height="0.6",
    penwidth="0.9"
)

# -------------------- Edge style (slim triangular arrows) --------------------
dot.attr(
    "edge",
    color="#606060",
    penwidth="0.9",
    arrowsize="0.6",
    arrowhead="normal"
)

# -------------------- Nodes with soft, publication-grade colors --------------------
dot.node("cppsite", "CPPSite 2.0", fillcolor="#E6EEF8")        # soft blue
dot.node("cellppd", "CellPPD", fillcolor="#E6EEF8")
dot.node("merge", "Merge datasets", fillcolor="#EAF6E6")       # mint green
dot.node("dedup", "Remove duplicates", fillcolor="#FFF8E1")    # warm cream
dot.node("esm", "Predict structures\n(ESMFold)", fillcolor="#FCE8EC")  # soft rose
dot.node("final", "Validated CPP dataset", fillcolor="#E6EEF8")# calm blue

# Invisible merge node for neat input convergence
dot.node("hub", "", shape="point", width="0.02", height="0.02", color="#777777")

# -------------------- Edges --------------------
dot.edge("cppsite", "hub")
dot.edge("cellppd", "hub")
dot.edge("hub", "merge")
dot.edge("merge", "dedup")
dot.edge("dedup", "esm")
dot.edge("esm", "final")

# Render PNG
out = dot.render("dataset_pipeline_vertical", cleanup=True)
print("Saved:", out)

