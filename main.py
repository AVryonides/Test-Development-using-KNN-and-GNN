"""
Project: Test Development using KNN, GNN
Description:
    This project computes SCOAP-based testability measures (controllability,
    observability, and detectability) for the nodes of a digital circuit
    described in .bench format. It extracts structural and SCOAP features
    for each node and uses:

      • KNN (K-Nearest Neighbors) for:
          - Regression: predicting numeric detectability D
          - Classification: EASY / MEDIUM / HARD detectability

      • GNN (Graph Neural Network, specifically a GCN) for:
          - Node-level regression of detectability D
          - Induced classification into EASY / MEDIUM / HARD based on thresholds

    The script prints key tables and metrics comparing KNN and GNN performance,
    including confusion matrices and regression statistics.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from math import inf
from collections import Counter, defaultdict, deque
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

def and_cc(CC0_a, CC1_a, CC0_b, CC1_b):
    CC0_z = min(CC0_a, CC0_b) + 1
    CC1_z = CC1_a + CC1_b + 1
    return CC0_z, CC1_z


def or_cc(CC0_a, CC1_a, CC0_b, CC1_b):
    CC0_z = CC0_a + CC0_b + 1
    CC1_z = min(CC1_a, CC1_b) + 1
    return CC0_z, CC1_z


def nand_cc(CC0_a, CC1_a, CC0_b, CC1_b):
    CC0_z = CC1_a + CC1_b + 1
    CC1_z = min(CC0_a, CC0_b) + 1
    return CC0_z, CC1_z


def nor_cc(CC0_a, CC1_a, CC0_b, CC1_b):
    CC0_z = min(CC1_a, CC1_b) + 1
    CC1_z = CC0_a + CC0_b + 1
    return CC0_z, CC1_z


def not_cc(CC0_a, CC1_a):
    CC0_z = CC1_a + 1
    CC1_z = CC0_a + 1
    return CC0_z, CC1_z


def and_co(CO_z, CC1_a, CC1_b):
    CO_a = CO_z + CC1_b + 1
    CO_b = CO_z + CC1_a + 1
    return CO_a, CO_b


def or_co(CO_z, CC0_a, CC0_b):
    CO_a = CO_z + CC0_b + 1
    CO_b = CO_z + CC0_a + 1
    return CO_a, CO_b


def nand_co(CO_z, CC1_a, CC1_b):
    CO_a = CO_z + CC1_b + 1
    CO_b = CO_z + CC1_a + 1
    return CO_a, CO_b


def nor_co(CO_z, CC0_a, CC0_b):
    CO_a = CO_z + CC0_b + 1
    CO_b = CO_z + CC0_a + 1
    return CO_a, CO_b


def not_co(CO_z):
    CO_a = CO_z + 1
    return CO_a


def fanout_co(*CO_z_list):
    return min(CO_z_list) if CO_z_list else inf


def parse_bench(path):
    nodes = set()
    primary_inputs = set()
    primary_outputs = set()
    edges = []
    gate_info = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("INPUT"):
                name = line[len("INPUT("):-1]
                primary_inputs.add(name)
                nodes.add(name)

            elif line.startswith("OUTPUT"):
                name = line[len("OUTPUT("):-1]
                primary_outputs.add(name)
                nodes.add(name)

            else:
                left, right = line.split("=")
                out = left.strip()

                gate_expr = right.strip()
                gate_type, inputs_part = gate_expr.split("(", 1)
                inputs_part = inputs_part[:-1]
                inputs = [x.strip() for x in inputs_part.split(",")]

                nodes.add(out)
                gate_info[out] = (gate_type.upper(), inputs)

                for inp in inputs:
                    nodes.add(inp)
                    edges.append((inp, out))

    return nodes, edges, primary_inputs, primary_outputs, gate_info


def build_graph_structures(nodes, edges):
    succ = defaultdict(list)
    pred = defaultdict(set)
    for u, v in edges:
        succ[u].append(v)
        pred[v].add(u)
    for v in nodes:
        succ[v] = succ[v]
        pred[v] = pred[v]
    return pred, succ


def compute_depth_from_PI(nodes, primary_inputs, pred):
    depth = {v: None for v in nodes}
    for v in nodes:
        if v in primary_inputs:
            depth[v] = 0

    changed = True
    while changed:
        changed = False
        for v in nodes:
            if v in primary_inputs:
                continue
            if pred[v]:
                if all(depth[p] is not None for p in pred[v]):
                    new_depth = 1 + max(depth[p] for p in pred[v])
                    if depth[v] != new_depth:
                        depth[v] = new_depth
                        changed = True
    return depth


def extract_node_features(nodes, edges, primary_inputs, primary_outputs, p_paths, CC0=None, CC1=None):
    pred, succ = build_graph_structures(nodes, edges)
    depth = compute_depth_from_PI(nodes, primary_inputs, pred)

    node_list = sorted(nodes, key=lambda x: int(x))
    X = []

    for v in node_list:
        fanin = len(pred[v])
        fanout = len(succ[v])
        is_PI = 1 if v in primary_inputs else 0
        is_PO = 1 if v in primary_outputs else 0
        d = depth[v] if depth[v] is not None else -1
        p = p_paths[v]

        features = [fanin, fanout, is_PI, is_PO, d, p]

        if CC0 is not None and CC1 is not None:
            features.append(CC0[v])
            features.append(CC1[v])

        X.append(features)

    return node_list, X


def topological_traversal(nodes, edges, primary_inputs):
    succ = defaultdict(list)
    pred = defaultdict(set)

    for u, v in edges:
        succ[u].append(v)
        pred[v].add(u)

    visited = {v: False for v in nodes}
    Q = deque()
    topo_order = []

    for v in nodes:
        if v in primary_inputs:
            Q.append(v)
            visited[v] = True
        else:
            visited[v] = False

    def can_be_visited(v):
        return all(visited[u] for u in pred[v])

    while Q:
        p = Q.popleft()
        topo_order.append(p)

        print(f"Processing node {p}")

        for v in succ[p]:
            if (not visited[v]) and can_be_visited(v):
                Q.append(v)
                visited[v] = True

    return topo_order


def count_paths(nodes, edges, primary_inputs, primary_outputs):
    succ = defaultdict(list)
    pred = defaultdict(set)

    for u, v in edges:
        succ[u].append(v)
        pred[v].add(u)

    p = {}
    visited = {}
    Q = deque()

    for v in nodes:
        if v in primary_inputs:
            p[v] = 1
            visited[v] = True
            Q.append(v)
        else:
            p[v] = 0
            visited[v] = False

    def can_be_visited(v):
        return all(visited[u] for u in pred[v])

    topo_order = []

    while Q:
        v = Q.popleft()
        topo_order.append(v)

        if v not in primary_inputs:
            p[v] = sum(p[u] for u in pred[v])

        for w in succ[v]:
            if (not visited[w]) and can_be_visited(w):
                visited[w] = True
                Q.append(w)

    total_paths = sum(p[v] for v in primary_outputs)

    return p, total_paths


def compute_scoap(nodes, edges, primary_inputs, primary_outputs, gate_info):
    pred, succ = build_graph_structures(nodes, edges)

    CC0 = {v: inf for v in nodes}
    CC1 = {v: inf for v in nodes}

    for v in primary_inputs:
        CC0[v] = 1
        CC1[v] = 1

    topo = []
    visited = {v: False for v in nodes}
    Q = deque(v for v in primary_inputs)

    for v in primary_inputs:
        visited[v] = True

    def ready(v):
        return all(visited[u] for u in pred[v])

    while Q:
        p_node = Q.popleft()
        topo.append(p_node)

        if p_node in gate_info:
            gate_type, ins = gate_info[p_node]

            if len(ins) == 1:
                a = ins[0]
                if gate_type == "NOT":
                    CC0[p_node], CC1[p_node] = not_cc(CC0[a], CC1[a])
                elif gate_type in ("BUFF", "BUF"):
                    CC0[p_node] = CC0[a] + 1
                    CC1[p_node] = CC1[a] + 1
                else:
                    raise ValueError(f"Unsupported 1-input gate type {gate_type}")

            elif len(ins) >= 2:
                if gate_type in ("AND", "NAND"):
                    cc0_and = min(CC0[x] for x in ins) + 1
                    cc1_and = sum(CC1[x] for x in ins) + 1
                    if gate_type == "AND":
                        CC0[p_node], CC1[p_node] = cc0_and, cc1_and
                    else:
                        CC0[p_node], CC1[p_node] = cc1_and, cc0_and

                elif gate_type in ("OR", "NOR"):
                    cc0_or = sum(CC0[x] for x in ins) + 1
                    cc1_or = min(CC1[x] for x in ins) + 1
                    if gate_type == "OR":
                        CC0[p_node], CC1[p_node] = cc0_or, cc1_or
                    else:
                        CC0[p_node], CC1[p_node] = cc1_or, cc0_or

                else:
                    raise ValueError(f"Unsupported gate type {gate_type}")

        for v in succ[p_node]:
            if not visited[v] and ready(v):
                visited[v] = True
                Q.append(v)

    CO = {v: inf for v in nodes}
    for po in primary_outputs:
        CO[po] = 0

    for p_node in reversed(topo):
        if p_node not in gate_info:
            continue
        gate_type, ins = gate_info[p_node]

        if len(ins) == 1:
            a = ins[0]
            if gate_type in ("NOT", "BUFF", "BUF"):
                new_co = not_co(CO[p_node])
            else:
                raise ValueError(f"Unsupported 1-input gate type {gate_type} in CO")
            CO[a] = min(CO[a], new_co)

        elif len(ins) >= 2:
            if gate_type in ("AND", "NAND"):
                total_cc1 = sum(CC1[x] for x in ins)
                for a in ins:
                    co_a = CO[p_node] + (total_cc1 - CC1[a]) + 1
                    CO[a] = min(CO[a], co_a)

            elif gate_type in ("OR", "NOR"):
                total_cc0 = sum(CC0[x] for x in ins)
                for a in ins:
                    co_a = CO[p_node] + (total_cc0 - CC0[a]) + 1
                    CO[a] = min(CO[a], co_a)

            else:
                raise ValueError(f"Unsupported gate type {gate_type} in CO")

    return CC0, CC1, CO


def euclidean_distance(x, y):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


def knn_predict_one(X_train, y_train, x_query, k):
    distances = []
    for xi, yi in zip(X_train, y_train):
        d = euclidean_distance(xi, x_query)
        distances.append((d, yi))
    distances.sort(key=lambda t: t[0])
    k_nearest = [label for (_, label) in distances[:k]]
    counter = Counter(k_nearest)
    most_common_label, _ = counter.most_common(1)[0]
    return most_common_label


def knn_predict(X_train, y_train, X_test, k):
    return [knn_predict_one(X_train, y_train, x, k) for x in X_test]


def choose_k_rule_of_thumb(N):
    if N <= 0:
        return 1
    k = int(round(N ** 0.5))
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def build_dataset_from_graph(node_list, X, labels):
    X_train, y_train = [], []
    X_test, test_nodes = [], []

    for node, feats in zip(node_list, X):
        if node in labels:
            X_train.append(feats)
            y_train.append(labels[node])
        else:
            X_test.append(feats)
            test_nodes.append(node)

    return X_train, y_train, X_test, test_nodes


def compute_dynamic_thresholds(D_dict, low_quantile=0.33, high_quantile=0.66):
    values = np.array(list(D_dict.values()), dtype=float)
    low_thresh = np.quantile(values, low_quantile)
    high_thresh = np.quantile(values, high_quantile)
    return low_thresh, high_thresh


def classify_detectability(D_value, low_thresh, high_thresh):
    if D_value <= low_thresh:
        return "EASY"
    elif D_value <= high_thresh:
        return "MEDIUM"
    else:
        return "HARD"


if __name__ == "__main__":
    bench_file = "s38417.bench"

    nodes, edges, PIs, POs, gate_info = parse_bench(bench_file)
    pred, succ = build_graph_structures(nodes, edges)

    p, total_paths = count_paths(nodes, edges, PIs, POs)
    CC0, CC1, CO = compute_scoap(nodes, edges, PIs, POs, gate_info)

    D_sa0 = {}
    D_sa1 = {}
    D = {}

    for v in nodes:
        D_sa0[v] = CC1[v] + CO[v]
        D_sa1[v] = CC0[v] + CO[v]
        D[v] = min(D_sa0[v], D_sa1[v])

    low_thresh, high_thresh = compute_dynamic_thresholds(D)
    print(f"\nDynamic thresholds for circuit:", bench_file)
    print(f" EASY   if D <= {low_thresh:.2f}")
    print(f" MEDIUM if {low_thresh:.2f} < D <= {high_thresh:.2f}")
    print(f" HARD   if D > {high_thresh:.2f}")

    detect_class = {}
    for v in sorted(nodes, key=lambda x: int(x)):
        d_val = D[v]
        cls = classify_detectability(d_val, low_thresh, high_thresh)
        detect_class[v] = cls

    #print("\nExtracting node features...")
    node_list, X = extract_node_features(nodes, edges, PIs, POs, p, CC0, CC1)

    print("\nFeature vectors per node:")
    print("(fanin, fanout, is_PI, is_PO, depth_from_PI, path_count, CC0, CC1)")
    #for node, feats in zip(node_list, X):
    #    print(f" node {node}: {feats}")

    y_reg = [D[node] for node in node_list]

    #print("\nLabels (numeric detectability D) per node (same order as features):")
    #for node, d_val in zip(node_list, y_reg):
    #    print(f" node {node}: D = {d_val}")

    y_cls_full = [detect_class[node] for node in node_list]

    #for node, lbl in zip(node_list, y_cls_full):
        #print(f" node {node}: class = {lbl}")

    X_array = np.array(X, dtype=float)
    y_reg_arr = np.array(y_reg, dtype=float)

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, train_nodes, test_nodes = train_test_split(
        X_array,
        y_reg_arr,
        y_cls_full,
        node_list,
        test_size=0.2,
        random_state=0,
        stratify=y_cls_full,
    )

    y_reg_train_raw = y_reg_train.copy()
    y_reg_test_raw = y_reg_test.copy()

    y_reg_train = np.log1p(y_reg_train)
    y_reg_test = np.log1p(y_reg_test)

    scaler_knn = StandardScaler()
    X_train = scaler_knn.fit_transform(X_train)
    X_test = scaler_knn.transform(X_test)

    n_train = X_train.shape[0]

    sqrt_k = int(round(math.sqrt(n_train)))
    if sqrt_k < 1:
        sqrt_k = 1
    if sqrt_k % 2 == 0:
        sqrt_k += 1

    candidate_ks = [3, 5, 7, 9, 11, sqrt_k]
    candidate_ks = sorted(set(k for k in candidate_ks if 1 <= k <= n_train))
    if not candidate_ks:
        candidate_ks = [1]

    #print(f"\nCandidate ks considered: {candidate_ks}")

    best_k_reg = None
    best_r2 = -1e9

    for k_candidate in candidate_ks:
        knn_reg_tmp = KNeighborsRegressor(n_neighbors=k_candidate, weights="distance")
        knn_reg_tmp.fit(X_train, y_reg_train)
        r2_tmp = knn_reg_tmp.score(X_train, y_reg_train)
        if r2_tmp > best_r2:
            best_r2 = r2_tmp
            best_k_reg = k_candidate

    print("\n==================== KNN EXPERIMENT ====================")
    print(
        f"\nBest k for regression (by train R^2 on log(D+1)) = {best_k_reg}, "
        f"R^2_train = {best_r2:.4f}"
    )

    k = best_k_reg
    print(f"Using k = {k} for KNN (both regression & classification)")

    knn_reg = KNeighborsRegressor(n_neighbors=k, weights="distance")
    knn_reg.fit(X_train, y_reg_train)

    y_reg_pred = knn_reg.predict(X_test)

    y_reg_test_orig = np.expm1(y_reg_test)
    y_reg_pred_orig = np.expm1(y_reg_pred)

    mae = mean_absolute_error(y_reg_test_orig, y_reg_pred_orig)
    mse = mean_squared_error(y_reg_test_orig, y_reg_pred_orig)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_reg_test_orig, y_reg_pred_orig)

    knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn_clf.fit(X_train, y_cls_train)

    y_cls_pred = knn_clf.predict(X_test)

    acc = accuracy_score(y_cls_test, y_cls_pred)
    print(f"\nKNN Classification accuracy (E/M/H) = {acc*100:.2f}%")

    labels = ["EASY", "MEDIUM", "HARD"]
    cm = confusion_matrix(y_cls_test, y_cls_pred, labels=labels)
    print("\nKNN Confusion matrix (rows = true, cols = predicted):")
    print("          EASY  MEDIUM  HARD")
    for lbl, row in zip(labels, cm):
        print(f"{lbl:6} {row[0]:7d} {row[1]:7d} {row[2]:7d}")

    print("\nKNN Regression metrics (numeric D, original scale):")
    print(f" MAE  (mean absolute error)      = {mae:.4f}")
    print(f" MSE  (mean squared error)       = {mse:.4f}")
    print(f" RMSE (root mean squared error)  = {rmse:.4f}")
    print(f" R^2  (coefficient of determination) = {r2:.4f}")

    print("\n\n==================== GNN EXPERIMENT ====================")

    X_gnn_np = np.array(X, dtype=float)

    scaler_gnn = StandardScaler()
    X_gnn_np = scaler_gnn.fit_transform(X_gnn_np)

    x = torch.tensor(X_gnn_np, dtype=torch.float32)
    print("Feature matrix shape (GNN):", x.shape)

    idx_of = {name: i for i, name in enumerate(node_list)}
    N = len(node_list)

    src = [idx_of[u] for (u, v) in edges]
    dst = [idx_of[v] for (u, v) in edges]
    src_rev = [idx_of[v] for (u, v) in edges]
    dst_rev = [idx_of[u] for (u, v) in edges]

    edge_index = torch.tensor(
        [src + src_rev, dst + dst_rev],
        dtype=torch.long,
    )

    print("edge_index shape:", edge_index.shape)

    d_values = np.array([D[v] for v in node_list], dtype=float)
    y_log = np.log1p(d_values)
    y = torch.tensor(y_log, dtype=torch.float32)

    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    for name in train_nodes:
        idx = idx_of[name]
        train_mask[idx] = True

    for name in test_nodes:
        idx = idx_of[name]
        test_mask[idx] = True

    print(f"GNN Training nodes: {train_mask.sum().item()}, Test nodes: {test_mask.sum().item()}")

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    class GCN(nn.Module):
        def __init__(self, in_dim, hid_dim=64, dropout=0.2):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hid_dim)
            self.conv2 = GCNConv(hid_dim, hid_dim)
            self.lin = nn.Linear(hid_dim, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.lin(x)
            return x.view(-1)

    torch.manual_seed(0)
    model = GCN(in_dim=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    #print("\nTraining GNN...")
    num_epochs = 1500
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        pred_gnn = model(data.x, data.edge_index)
        loss = loss_fn(pred_gnn[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        #if epoch % 50 == 0:
         #   print(f"Epoch {epoch}, Train Loss (log D space) = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_gnn = model(data.x, data.edge_index)
        pred_np_log = pred_gnn.detach().numpy()
        y_np_log = data.y.detach().numpy()

        pred_np = np.expm1(pred_np_log)
        y_np = np.expm1(y_np_log)

        test_idx = np.where(data.test_mask.numpy())[0]

        test_pred = pred_np[test_idx]
        test_true = y_np[test_idx]

        mse_gnn = mean_squared_error(test_true, test_pred)
        mae_gnn = mean_absolute_error(test_true, test_pred)
        rmse_gnn = math.sqrt(mse_gnn)
        r2_gnn = r2_score(test_true, test_pred)

    print("\n===== GNN Evaluation =====")
    print(f"Test MSE (D space)  = {mse_gnn:.4f}")
    print(f"Test RMSE (D space) = {rmse_gnn:.4f}")
    print(f"Test MAE (D space)  = {mae_gnn:.4f}")
    print(f"Test R^2 (D space)  = {r2_gnn:.4f}")

    #print("\nGNN regression results (predicting D):")
    #for idx, true_d, pred_d in zip(test_idx, test_true, test_pred):
    #    node_name = node_list[idx]

    #print("\nGNN classification results (EASY/MEDIUM/HARD):")

    def classify_predicted_D(pred_value, low, high):
        if pred_value <= low:
            return "EASY"
        elif pred_value <= high:
            return "MEDIUM"
        else:
            return "HARD"

    gnn_cls_pred = []
    gnn_cls_true = []

    for idx, true_d, pred_d in zip(test_idx, test_true, test_pred):
        node_name = node_list[idx]

        true_class = detect_class[node_name]
        pred_class = classify_predicted_D(pred_d, low_thresh, high_thresh)

        gnn_cls_pred.append(pred_class)
        gnn_cls_true.append(true_class)

    acc_gnn = accuracy_score(gnn_cls_true, gnn_cls_pred)
    cm_gnn = confusion_matrix(gnn_cls_true, gnn_cls_pred, labels=labels)

    print(f"\nGNN Classification accuracy (E/M/H) = {acc_gnn*100:.2f}%")
    print("\nGNN Confusion matrix (rows = true, cols = predicted):")
    print("          EASY  MEDIUM  HARD")
    for lbl, row in zip(labels, cm_gnn):
        print(f"{lbl:6} {row[0]:7d} {row[1]:7d} {row[2]:7d}")

    print("\nGNN Regression metrics (numeric D, original scale):")
    print(f" MAE  (mean absolute error)      = {mae_gnn:.4f}")
    print(f" MSE  (mean squared error)       = {mse_gnn:.4f}")
    print(f" RMSE (root mean squared error)  = {rmse_gnn:.4f}")
    print(f" R^2  (coefficient of determination) = {r2_gnn:.4f}")
