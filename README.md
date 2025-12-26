# Test-Development-using-KNN-and-GNN
Course project for ECE407 CAD for VLSI (University of Cyprus): parses .bench combinational circuits, builds a dependency graph, computes SCOAP testability metrics, then uses KNN and a GCN to predict per-node fault detectability (easy/medium/hard).

This project performs machine-learning–assisted testability analysis for combinational digital circuits described in ISCAS’85/89 .bench format. It starts by parsing a netlist, extracting primary inputs/outputs and gate dependencies, and building a structural DAG (G1 graph) where nodes are signals and directed edges capture functional dependence (input → gate output). The graph is levelized via topological traversal, enabling PI→PO path counting and other structural measurements used later as learning features.

**project407_partA**

On top of the structural model, the pipeline computes SCOAP testability metrics—controllability (CC0/CC1) and observability (CO)—and derives a per-node detectability score **D** for stuck-at faults. Each node is represented by a feature vector combining structural and SCOAP information (e.g., fanin/fanout, PI/PO flags, depth, path count, CC0/CC1), with standardization applied before learning.

**project407_partB**

Two models are implemented and compared:

- **KNN baseline** for (1) regression on detectability (using a `log(1 + D)` target transform) and (2) classification into **EASY / MEDIUM / HARD** difficulty bands derived from circuit-specific percentiles.

- **Graph Neural Network (GCN)** using PyTorch Geometric for node-level regression on detectability, leveraging message passing over the circuit graph, with optional induced **E/M/H** classification for interpretability.

The main script ties everything together, runs the full pipeline, and prints evaluation results (regression metrics and confusion matrices) for direct KNN vs. GNN comparison.
