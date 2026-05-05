from graphviz import Digraph

dot = Digraph("BayesianPowerGridGNN", format="png")

dot.attr(rankdir="LR", splines="true", nodesep="0.6", ranksep="1.0")
dot.attr(fontname="Helvetica")

# =========================
# NODE STYLES
# =========================
dot.attr("node", fontname="Helvetica", fontsize="10")

# =========================
# INPUT LAYER
# =========================
with dot.subgraph(name="cluster_input") as c:
    c.attr(label="Input Graph", color="#aaaaaa")

    c.node("v1", "V", shape="circle", style="filled", fillcolor="#A8D5BA")
    c.node("f1", "F", shape="square", style="filled", fillcolor="#F8B195")

    c.edge("v1", "f1")
    c.edge("f1", "v1")

# =========================
# ENCODER
# =========================
with dot.subgraph(name="cluster_encoder") as c:
    c.attr(label="Encoders (→ 32D)", color="#C06C84")

    c.node("v2", "V'", shape="circle", style="filled", fillcolor="#7FB77E")
    c.node("f2", "F'", shape="square", style="filled", fillcolor="#F67280")

# Connections
dot.edge("v1", "v2")
dot.edge("f1", "f2")

# =========================
# MESSAGE PASSING
# =========================
with dot.subgraph(name="cluster_mp") as c:
    c.attr(label="Message Passing ×3 (GAT + Residual)", color="#6C5B7B")

    c.node("v3", "V*", shape="circle", style="filled", fillcolor="#4ECDC4")
    c.node("f3", "F*", shape="square", style="filled", fillcolor="#FF6B6B")

    # bidirectional message passing
    c.edge("v3", "f3", label="GAT")
    c.edge("f3", "v3", label="GAT")

# Connect encoder → MP
dot.edge("v2", "v3")
dot.edge("f2", "f3")

# =========================
# LATENT
# =========================
dot.node(
    "H",
    "h (latent)",
    shape="circle",
    style="filled",
    fillcolor="#355C7D",
    fontcolor="white",
)

dot.edge("v3", "H")

# =========================
# HEADS
# =========================
with dot.subgraph(name="cluster_heads") as c:
    c.attr(label="Bayesian Heads", color="#999999")

    c.node("mu", "μ", shape="ellipse", style="filled", fillcolor="#99B898")

    c.node("var", "σ²", shape="ellipse", style="filled", fillcolor="#F67280")

# Branching
dot.edge("H", "mu")
dot.edge("H", "var", style="dashed", label="MC Dropout")

# =========================
# OUTPUT
# =========================
dot.node(
    "out", "Gaussian\nOutput", shape="box", style="rounded,filled", fillcolor="#F4A261"
)

dot.edge("mu", "out")
dot.edge("var", "out")

# =========================
# EXPORT
# =========================
dot.render("bayesian_gnn_graph_style", format="png", cleanup=True)
dot.render("bayesian_gnn_graph_style", format="pdf", cleanup=True)
