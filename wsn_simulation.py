#!/usr/bin/env python3
"""Αναπαραγώγιμη μελέτη WSN, N=100. Εκτέλεση: python wsn_simulation.py

Απαιτεί: numpy, networkx, matplotlib, pandas.
Εξάγει πίνακες CSV, γραφήματα PNG/PDF, δεδομένα NPZ και summary.json.
Q1: shortest paths ελάχιστων hops. Q2/Q3: ελάχιστου γεωμετρικού μήκους.
Τα E2/E3 είναι δείκτες κόστους ανά δευτερόλεπτο, όχι ενέργεια σε Joule.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Αποθήκευση γραφημάτων χωρίς να απαιτείται παράθυρο GUI.
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd

N = 100
POSITION_SEED = 42
TRANSMIT_SEED = 123
SINK_SELECTION_SEED = 456
RC_REFERENCE = 0.20
N_SINKS = 20
N_RADII = 25
T_SECONDS = 2520  # ΕΚΠ(1,...,10): ακριβή πλήθη περιοδικών πακέτων στο (0,T].


def generate_nodes(n=N, seed=POSITION_SEED):
    """Ανεξάρτητες ομοιόμορφες θέσεις στο [0,1)^2, με ρητό PCG64."""
    return np.random.Generator(np.random.PCG64(seed)).random((n, 2))


def assign_transmit_times(n=N, seed=TRANSMIT_SEED):
    """Μία σταθερή περίοδος για κάθε ID, ακόμη κι αν γίνει αργότερα sink."""
    rng = np.random.Generator(np.random.PCG64(seed))
    times = rng.integers(1, 11, size=n, dtype=np.int64)
    return times, 1.0 / times


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))


def distance_matrix(positions):
    """Υπολογισμός μία φορά, ώστε το όριο <= rc να χρησιμοποιεί ίδια floats."""
    n = len(positions)
    distances = np.zeros((n, n))
    for u in range(n):
        for v in range(u + 1, n):
            distances[u, v] = distances[v, u] = euclidean_distance(
                positions[u], positions[v]
            )
    return distances


def build_graph(positions, distances, rc):
    """Απλός, ακατεύθυντος γεωμετρικός γράφος, χωρίς ανοχή στο <= rc."""
    graph = nx.Graph()
    for u, xy in enumerate(positions):
        graph.add_node(u, pos=tuple(xy))
    for u in range(len(positions)):
        for v in range(u + 1, len(positions)):
            if distances[u, v] <= rc:
                graph.add_edge(u, v, length=float(distances[u, v]))
    return graph


def find_minimum_connected_rc(positions, distances):
    """Το μεγαλύτερο μήκος ακμής του ευκλείδειου MST είναι ακριβώς rc_min."""
    complete = build_graph(positions, distances, float(distances.max()))
    mst = nx.minimum_spanning_tree(complete, weight="length", algorithm="kruskal")
    rc_min = max(data["length"] for _, _, data in mst.edges(data=True))
    at_threshold = build_graph(positions, distances, rc_min)
    below = build_graph(positions, distances, np.nextafter(rc_min, -np.inf))
    assert nx.is_connected(at_threshold)
    assert not nx.is_connected(below)
    return rc_min, mst, nx.number_connected_components(below)


def find_bsink_node(graph):
    """Μέγιστος degree. Σε ισοβαθμία επιλέγεται το μικρότερο node ID."""
    return min(graph.nodes, key=lambda u: (-graph.degree[u], u))


def select_sink_nodes(graph, bsink, count=N_SINKS, seed=SINK_SELECTION_SEED):
    """Κάλυψη όσο γίνεται περισσότερων διαφορετικών degrees και των άκρων."""
    rng = np.random.Generator(np.random.PCG64(seed))
    groups = {}
    for u in sorted(graph.nodes):
        groups.setdefault(graph.degree[u], []).append(u)
    degrees = sorted(groups)
    if len(degrees) > count:
        indices = np.rint(np.linspace(0, len(degrees) - 1, count)).astype(int)
        targets = [degrees[i] for i in indices]
    else:
        targets = degrees
    selected = [bsink]
    for degree in targets:
        if degree != graph.degree[bsink]:
            selected.append(int(rng.choice(groups[degree])))
    # Αν τα διαφορετικά degrees είναι <20, συμπληρώνουμε ισορροπημένα.
    while len(selected) < count:
        remaining = {d: [u for u in groups[d] if u not in selected] for d in degrees}
        available = [d for d in degrees if remaining[d]]
        used = {d: sum(graph.degree[u] == d for u in selected) for d in available}
        least = min(used.values())
        degree = int(rng.choice([d for d in available if used[d] == least]))
        selected.append(int(rng.choice(remaining[degree])))
    assert len(set(selected)) == count and bsink in selected
    assert len({graph.degree[u] for u in selected}) == min(count, len(degrees))
    return sorted(selected, key=lambda u: (graph.degree[u], u))


def build_routing_tree(graph, sink, metric="length"):
    """Κάθε u!=s έχει έναν parent προς s, χωρίς τυχαίο μοίρασμα κίνησης.

    Σε ακριβώς ίσα shortest paths διαλέγουμε τον μικρότερο predecessor ID.
    Οι σχεδόν ίσες αποστάσεις δεν μετατρέπονται τεχνητά σε ισοπαλίες.
    """
    n = len(graph)
    graph_hops = dict(nx.single_source_shortest_path_length(graph, sink))
    if len(graph_hops) != n:
        raise ValueError("Απαιτείται συνεκτικός γράφος.")
    if metric == "length":
        predecessors, costs = nx.dijkstra_predecessor_and_distance(
            graph, sink, weight="length"
        )
    elif metric == "hops":
        costs = graph_hops
        predecessors = {
            u: [v for v in graph[u] if costs[v] == costs[u] - 1]
            for u in graph
        }
    else:
        raise ValueError("metric πρέπει να είναι 'length' ή 'hops'.")
    parent = np.full(n, -1, dtype=int)
    for u in graph:
        if u != sink:
            parent[u] = min(predecessors[u])
            assert costs[parent[u]] < costs[u]
    order = sorted(graph, key=lambda u: (costs[u], u))
    x = np.zeros(n)
    route_hops = np.zeros(n, dtype=int)
    for u in order:
        if u != sink:
            p = parent[u]
            x[u] = x[p] + graph[u][p]["length"]
            route_hops[u] = route_hops[p] + 1
    return {"sink": sink, "parent": parent, "order": order, "x": x,
            "route_hops": route_hops,
            "graph_hops": np.array([graph_hops[u] for u in range(n)]),
            "metric": metric}


def effective_rates(base_rates, sink):
    """Μηδενίζεται μόνο ένα ΑΝΤΙΓΡΑΦΟ: ο sink δεν γεννά εξερχόμενη κίνηση."""
    rates = base_rates.copy()
    rates[sink] = 0.0
    return rates


def calculate_cumulative_load(tree, base_rates):
    """Λ(u)=λ(u)+άθροισμα Λ των παιδιών. Το ίδιο το u περιλαμβάνεται."""
    load = effective_rates(base_rates, tree["sink"])
    for u in reversed(tree["order"]):
        if u != tree["sink"]:
            load[tree["parent"][u]] += load[u]
    return load


def calculate_total_transmissions(tree, base_rates):
    """Link transmissions/sec: κάθε πηγή χρεώνεται μία φορά ανά hop."""
    return float(np.dot(effective_rates(base_rates, tree["sink"]), tree["route_hops"]))


def calculate_energy_metric(tree, base_rates, load=None):
    """E2=sum λ*x και κατά γράμμα E3=sum Λ*x, για το ίδιο routing tree."""
    if load is None:
        load = calculate_cumulative_load(tree, base_rates)
    rates = effective_rates(base_rates, tree["sink"])
    return float(np.dot(rates, tree["x"])), float(np.dot(load, tree["x"]))


def validate_tree(graph, tree, base_rates, load):
    """Ανεξάρτητη επαλήθευση με περπάτημα της διαδρομής κάθε πηγής."""
    sink = tree["sink"]
    rates = effective_rates(base_rates, sink)
    direct_load = np.zeros(len(graph))
    tx_direct = e2_direct = e3_direct = link_cost = 0.0
    for source in graph:
        if source == sink:
            continue
        u, visited = source, set()
        while u != sink:
            assert u not in visited
            visited.add(u)
            direct_load[u] += rates[source]
            p = int(tree["parent"][u])
            assert graph.has_edge(u, p)
            tx_direct += rates[source]
            e2_direct += rates[source] * graph[u][p]["length"]
            e3_direct += rates[source] * tree["x"][u]
            u = p
        direct_load[sink] += rates[source]
    for u in graph:
        if u != sink:
            link_cost += load[u] * graph[u][tree["parent"][u]]["length"]
    e2, e3 = calculate_energy_metric(tree, base_rates, load)
    tx = calculate_total_transmissions(tree, base_rates)
    np.testing.assert_allclose(load, direct_load, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose([tx, e2, e3], [tx_direct, e2_direct, e3_direct],
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tx, load.sum() - load[sink], rtol=1e-12)
    np.testing.assert_allclose(e2, link_cost, rtol=1e-12)
    np.testing.assert_allclose(load[sink], rates.sum(), rtol=1e-12)
    assert np.all(tree["route_hops"] >= tree["graph_hops"])
    assert e3 >= e2 - 1e-12
    return abs(e2 - link_cost)


def sink_metrics(graph, sink, rates, times):
    tree = build_routing_tree(graph, sink, "hops")
    load = calculate_cumulative_load(tree, rates)
    validate_tree(graph, tree, rates, load)
    tx = calculate_total_transmissions(tree, rates)
    offered = float(effective_rates(rates, sink).sum())
    packet_counts = T_SECONDS // times
    packet_counts[sink] = 0
    exact_count = int(np.dot(packet_counts, tree["route_hops"]))
    np.testing.assert_allclose(exact_count, T_SECONDS * tx, rtol=1e-12)
    return {"sink": sink, "degree": graph.degree[sink], "tx_per_second": tx,
            "mean_hops": float(tree["route_hops"].sum() / (len(graph) - 1)),
            "source_rate": offered, "traffic_weighted_hops": tx / offered,
            "transmissions_2520s": exact_count}


def hop_level_statistics(nodes, column):
    return nodes.groupby(column)["cumulative_load"].agg(
        count="count", mean="mean", median="median", min="min", max="max"
    ).reset_index()


def save_figure(fig, folder, stem):
    """Ίδιο figure σε PNG για ανάγνωση και vector PDF για την αναφορά."""
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{stem}.{extension}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_network(graph, positions, highlights, folder):
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    segments = [(positions[u], positions[v]) for u, v in graph.edges]
    ax.add_collection(LineCollection(segments, colors="#bbc9d5", linewidths=.6, alpha=.55))
    ax.scatter(*positions.T, c="#547b92", s=25, zorder=2)
    for label, sink, color, marker in highlights:
        ax.scatter(*positions[sink], s=170, color=color, marker=marker,
                   edgecolors="white", linewidths=.8, label=f"{label}: {sink}", zorder=4)
        ax.annotate(str(sink), positions[sink], xytext=(7, 7), textcoords="offset points")
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="x", ylabel="y",
           title=f"Fixed network | N={len(graph)}, rc={RC_REFERENCE:.2f}, edges={graph.number_of_edges()}")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=9)
    save_figure(fig, folder, "07_network_sinks")


def make_plots(q1, radii, nodes, energies, comparison, graph, positions, folder):
    """Όλα τα γραφήματα χρησιμοποιούν τους ίδιους πίνακες που εξάγονται σε CSV."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18})
    blue, orange, green = "#156c94", "#d16b30", "#25856a"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for ax, column, label in zip(axes, ["degree", "mean_hops"],
                                 ["Sink degree", "Mean minimum hops (99 sources)"]):
        ax.scatter(q1[column], q1.tx_per_second, color=blue, s=38, label="Selected sinks")
        b = q1.loc[q1.is_bsink].iloc[0]
        ax.scatter(b[column], b.tx_per_second, color=orange, marker="*", s=200,
                   label=f"bsink = {int(b.sink)}", zorder=3)
        label_offsets = {20: (-18, -13), 22: (-17, 10), 1: (8, 6),
                         93: (5, -13), 39: (7, -13), 3: (7, 11),
                         9: (-13, -13), 0: (-10, 8), 51: (-18, 9)}
        for row in q1.itertuples():
            offset = label_offsets.get(row.sink, (4, 4)) if column == "mean_hops" else (4, 4)
            ax.annotate(str(row.sink), (getattr(row, column), row.tx_per_second),
                        xytext=offset, textcoords="offset points", fontsize=7,
                        arrowprops={"arrowstyle": "-", "color": "#a0a0a0", "lw": .4})
        ax.set(xlabel=label, ylabel="Link transmissions / s")
    axes[0].legend(fontsize=8)
    fig.suptitle("Q1a | Sink choice and transmission load")
    save_figure(fig, folder, "01_sink_transmissions")

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 9), sharex=True, constrained_layout=True)
    for ax, fixed, dynamic, label in [
        (axes[0], "tx_fixed", "tx_dynamic", "Link transmissions / s"),
        (axes[2], "mean_hops_fixed", "mean_hops_dynamic", "Mean minimum hops")]:
        ax.plot(radii.rc, radii[fixed], "o-", color=blue, ms=3, label="Fixed baseline bsink")
        ax.plot(radii.rc, radii[dynamic], "s--", color=orange, ms=3, label="Recomputed max-degree sink")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    axes[1].plot(radii.rc, radii.average_degree, "o-", color=green, ms=3)
    axes[1].set_ylabel("Average degree")
    axes[2].set_xlabel("Communication radius rc (sampled values)")
    fig.suptitle("Q1b | Same positions and transmission periods")
    save_figure(fig, folder, "02_radius_effect")

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    points = ax.scatter(nodes.x_path, nodes.cumulative_load, c=nodes.route_hops,
                        cmap="viridis", s=40, alpha=.85)
    for row in nodes.nlargest(3, "cumulative_load").itertuples():
        ax.annotate(str(row.node), (row.x_path, row.cumulative_load),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    fig.colorbar(points, ax=ax, label="Hops on length-shortest routing tree")
    ax.set(xlabel="x(u, 0): sum of edge lengths on routing path",
           ylabel="Cumulative load (packets / s)", title="Q2a | Sink 0, length-shortest routing")
    save_figure(fig, folder, "03_load_vs_path_length")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    for ax, column, label in zip(axes, ["graph_hops", "route_hops"],
         ["Minimum hops in graph (BFS)", "Actual hops on length-shortest tree"]):
        stats = hop_level_statistics(nodes, column)
        ax.scatter(nodes[column], nodes.cumulative_load, s=28, alpha=.48, color=blue)
        ax.plot(stats[column], stats["mean"], "o-", color=orange, label="Mean")
        ax.plot(stats[column], stats["median"], "s--", color=green, label="Median")
        ax.set(xlabel=label, xticks=stats[column].tolist())
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Cumulative load (packets / s)")
    fig.suptitle("Q2b | Same cumulative loads; two distinct hop distances")
    save_figure(fig, folder, "04_load_vs_hops")

    specials = [("E2 optimum", int(energies.loc[energies.E2.idxmin(), "sink"]), green, "*"),
                ("E3 optimum", int(energies.loc[energies.E3.idxmin(), "sink"]), "#8255a0", "D"),
                ("bsink", find_bsink_node(graph), orange, "s"),
                ("Node 0", 0, "#252b32", "^")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    for ax, metric in zip(axes, ["E2", "E3"]):
        ax.scatter(energies.sink, energies[metric], s=19, color=blue, alpha=.8)
        for label, sink, color, marker in specials:
            row = energies.loc[energies.sink == sink].iloc[0]
            ax.scatter(sink, row[metric], s=95, color=color, marker=marker,
                       label=f"{label}: {sink}", zorder=4)
        ax.set_ylabel(f"{metric} (packet-distance units / s)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("Candidate sink ID (all 100 nodes)")
    fig.suptitle("Q2c / Q3 | Exhaustive sink evaluation")
    save_figure(fig, folder, "05_energy_all_sinks")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    labels = [f"{r.role}\nnode {r.sink}" for r in comparison.itertuples()]
    for ax, metric in zip(axes, ["E2", "E3"]):
        values = comparison[metric] / energies[metric].min()
        bars = ax.bar(np.arange(len(labels)), values, color=blue, width=.65)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.axhline(1, color=orange, linestyle="--", linewidth=1)
        ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
               ylabel=f"{metric} / minimum {metric}", ylim=(0, values.max() * 1.16))
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Sink comparison | Each metric normalized by its own optimum")
    save_figure(fig, folder, "06_energy_comparison")
    plot_network(graph, positions, specials, folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    output = args.output.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    positions = generate_nodes()
    times, rates = assign_transmit_times()
    initial_positions, initial_times, initial_rates = positions.copy(), times.copy(), rates.copy()
    # Προστασία από τυχαία μεταβολή των πρωτογενών δεδομένων.
    positions.setflags(write=False)
    times.setflags(write=False)
    rates.setflags(write=False)
    distances = distance_matrix(positions)
    distances.setflags(write=False)
    assert positions.shape == (N, 2) and np.all((positions >= 0) & (positions < 1))
    assert np.issubdtype(times.dtype, np.integer) and np.all((times >= 1) & (times <= 10))
    np.testing.assert_array_equal(rates, 1.0 / times)
    assert np.all(distances[np.triu_indices(N, 1)] > 0)

    rc_min, mst, below_components = find_minimum_connected_rc(positions, distances)
    graph = build_graph(positions, distances, RC_REFERENCE)
    if not nx.is_connected(graph):
        raise ValueError(f"RC_REFERENCE={RC_REFERENCE} < required rc_min={rc_min}; αυξήστε το.")
    bsink = find_bsink_node(graph)
    assert graph.degree[bsink] == max(dict(graph.degree()).values())
    selected = select_sink_nodes(graph, bsink)

    all_q1 = pd.DataFrame([sink_metrics(graph, s, rates, times) for s in range(N)])
    all_q1["is_bsink"] = all_q1.sink == bsink
    q1 = all_q1.set_index("sink").loc[selected].reset_index()
    tx_optimum = int(all_q1.loc[all_q1.tx_per_second.idxmin(), "sink"])

    radius_values = np.unique(np.r_[np.linspace(rc_min, 1.0, N_RADII), RC_REFERENCE])
    radius_rows = []
    for rc in radius_values:
        g = build_graph(positions, distances, float(rc))
        assert len(g) == N and nx.is_connected(g)
        for u in g:
            np.testing.assert_array_equal(g.nodes[u]["pos"], initial_positions[u])
        dynamic_sink = find_bsink_node(g)
        fixed = sink_metrics(g, bsink, rates, times)
        dynamic = sink_metrics(g, dynamic_sink, rates, times)
        radius_rows.append({"rc": float(rc), "connected": True, "edges": g.number_of_edges(),
            "average_degree": 2 * g.number_of_edges() / N, "fixed_sink": bsink,
            "degree_fixed_sink": g.degree[bsink], "dynamic_bsink": dynamic_sink,
            "degree_dynamic_bsink": g.degree[dynamic_sink], "tx_fixed": fixed["tx_per_second"],
            "tx_dynamic": dynamic["tx_per_second"], "mean_hops_fixed": fixed["mean_hops"],
            "mean_hops_dynamic": dynamic["mean_hops"],
            "source_rate_fixed": fixed["source_rate"], "source_rate_dynamic": dynamic["source_rate"]})
    radii = pd.DataFrame(radius_rows)
    assert np.all(np.diff(radii.tx_fixed) <= 1e-10)
    assert np.all(np.diff(radii.mean_hops_fixed) <= 1e-12)
    assert np.all(np.diff(radii.edges) >= 0)

    energy_rows, node_rows, errors, tree_rows = [], [], [], []
    for sink in range(N):
        # Κρίσιμο: νέο routing tree ΚΑΙ νέα Λ για κάθε υποψήφιο sink.
        tree = build_routing_tree(graph, sink, "length")
        load = calculate_cumulative_load(tree, rates)
        errors.append(validate_tree(graph, tree, rates, load))
        e2, e3 = calculate_energy_metric(tree, rates, load)
        energy_rows.append({"sink": sink, "degree": graph.degree[sink],
            "pos_x": positions[sink, 0], "pos_y": positions[sink, 1],
            "E2": e2, "E3": e3, "tx_length_routing": calculate_total_transmissions(tree, rates),
            "mean_route_hops": tree["route_hops"].sum() / (N - 1),
            "linear_link_cost": sum(load[u] * graph[u][tree["parent"][u]]["length"]
                                    for u in graph if u != sink)})
        for u in graph:
            tree_rows.append({"sink": sink, "node": u, "parent": int(tree["parent"][u]),
                "x_path": tree["x"][u], "route_hops": int(tree["route_hops"][u]),
                "graph_hops": int(tree["graph_hops"][u]), "cumulative_load": load[u]})
            if sink == 0 and u != 0:
                node_rows.append({"node": u, "parent": int(tree["parent"][u]),
                    "lambda": rates[u], "cumulative_load": load[u], "x_path": tree["x"][u],
                    "straight_distance": distances[u, 0],
                    "graph_hops": int(tree["graph_hops"][u]),
                    "route_hops": int(tree["route_hops"][u])})
    energies = pd.DataFrame(energy_rows)
    nodes = pd.DataFrame(node_rows)
    s2 = int(energies.loc[energies.E2.idxmin(), "sink"])
    s3 = int(energies.loc[energies.E3.idxmin(), "sink"])
    assert len(energies) == N and energies.sink.nunique() == N
    assert energies.loc[s2, "E2"] == min(energies.E2)
    assert energies.loc[s3, "E3"] == min(energies.E3)
    # Μικρότερο ID σε ακριβή ισοβαθμία, επειδή οι πίνακες είναι σε σειρά ID.
    center_sink = int(np.argmin(np.linalg.norm(positions - np.array([.5, .5]), axis=1)))
    roles = [("E2 optimum", s2), ("E3 optimum", s3), ("Max degree", bsink),
             ("Node 0", 0), ("Field center", center_sink)]
    comparison = pd.DataFrame([{"role": role, **energies.loc[s].to_dict()}
                               for role, s in roles])
    comparison["sink"] = comparison.sink.astype(int)
    comparison["degree"] = comparison.degree.astype(int)

    np.testing.assert_array_equal(positions, initial_positions)
    np.testing.assert_array_equal(times, initial_times)
    np.testing.assert_array_equal(rates, initial_rates)
    np.savez(output / "fixed_inputs.npz", positions=positions, t_transmit=times,
             base_rates=rates, distances=distances)
    inputs = pd.DataFrame({"node": range(N), "x": positions[:, 0], "y": positions[:, 1],
                           "t_transmit": times, "lambda": rates})
    edges = pd.DataFrame([{"u": u, "v": v, "length": d["length"]}
                           for u, v, d in graph.edges(data=True)])
    tables = {"nodes": inputs, "reference_edges": edges, "q1a_selected_sinks": q1,
              "q1a_all_sinks": all_q1, "q1b_radius_sweep": radii, "q2_nodes_sink0": nodes,
              "q2b_graph_hop_stats": hop_level_statistics(nodes, "graph_hops"),
              "q2b_route_hop_stats": hop_level_statistics(nodes, "route_hops"),
              "q2c_q3_all_sinks": energies, "sink_comparison": comparison,
              "all_weighted_routing_trees": pd.DataFrame(tree_rows)}
    for name, frame in tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        pd.testing.assert_frame_equal(frame, pd.read_csv(path), check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-12)
    make_plots(q1, radii, nodes, energies, comparison, graph, positions, figures)

    summary = {"N": N, "position_seed": POSITION_SEED, "transmit_seed": TRANSMIT_SEED,
        "sink_selection_seed": SINK_SELECTION_SEED, "rng": "Generator(PCG64)",
        "rc_reference": RC_REFERENCE, "rc_min": rc_min,
        "components_immediately_below_rc_min": below_components,
        "rc_min_edges": build_graph(positions, distances, rc_min).number_of_edges(),
        "mst_bottleneck_edges": [[u, v, d["length"]] for u, v, d in mst.edges(data=True)
                                  if d["length"] == rc_min],
        "reference_edges": graph.number_of_edges(), "reference_average_degree": 2 * graph.number_of_edges()/N,
        "bsink": bsink, "bsink_degree": graph.degree[bsink], "bsink_position": positions[bsink].tolist(),
        "max_degree_ties": [u for u in graph if graph.degree[u] == graph.degree[bsink]],
        "selected_sinks": selected, "distinct_degrees": sorted(set(dict(graph.degree()).values())),
        "best_transmission_sink": tx_optimum, "best_transmissions": float(all_q1.loc[tx_optimum, "tx_per_second"]),
        "bsink_transmissions": float(all_q1.loc[bsink, "tx_per_second"]),
        "E2_optimum": s2, "E2_minimum": float(energies.E2.min()),
        "E3_optimum": s3, "E3_minimum": float(energies.E3.min()),
        "E2_runner_up_gap": float(np.sort(energies.E2)[1] - energies.E2.min()),
        "E3_runner_up_gap": float(np.sort(energies.E3)[1] - energies.E3.min()),
        "nearest_field_center": center_sink, "base_rate_sum": float(rates.sum()),
        "sink0_source_rate": float(effective_rates(rates, 0).sum()),
        "sink0_nodes_with_extra_route_hops": int((nodes.route_hops > nodes.graph_hops).sum()),
        "sink0_max_load_node": int(nodes.loc[nodes.cumulative_load.idxmax(), "node"]),
        "sink0_max_load": float(nodes.cumulative_load.max()),
        "corr_x_load": float(nodes.x_path.corr(nodes.cumulative_load)),
        "corr_graph_hops_load": float(nodes.graph_hops.corr(nodes.cumulative_load)),
        "corr_route_hops_load": float(nodes.route_hops.corr(nodes.cumulative_load)),
        "max_abs_E2_link_identity_error": max(errors), "all_checks_passed": True,
        "fixed_inputs_sha256": hashlib.sha256(positions.tobytes()+times.tobytes()+rates.tobytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "networkx": nx.__version__, "matplotlib": matplotlib.__version__, "pandas": pd.__version__}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = [
        "PASS: 100 nodes; coordinates in the unit square.",
        "PASS: fixed positions and periods; read-only arrays and exact before/after equality.",
        "PASS: integer tTransmit in 1..10 and lambda=1/tTransmit.",
        "PASS: all analyzed reference/sweep networks are connected.",
        "PASS: rc_min from MST; disconnected at the preceding float, connected at rc_min.",
        "PASS: maximum-degree bsink, deterministic tie rule; 20 distinct selected sinks.",
        "PASS: maximum achievable number of represented degree values.",
        "PASS: every route reaches its sink without cycles.",
        "PASS: cumulative loads match independent source-by-source path accumulation.",
        "PASS: transmission sum equals source-rate times hops and sum of non-sink loads.",
        "PASS: exact periodic packet counts at T=2520 s match the analytic rate.",
        "PASS: all 100 candidates evaluated for E2 and E3; minima checked.",
        "PASS: E2 equals sum of load times NEXT-LINK length, for every sink.",
        "PASS: E3 matches independent sum of remaining distances per source.",
        "PASS: fixed-sink hops/transmissions nonincreasing as radius increases.",
        "PASS: all CSVs round-trip numerically; plots consume those same tables."]
    (output / "validation.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nQ1a - 20 selected sinks:\n" + q1.to_string(index=False))
    print("\nSink comparison:\n" + comparison.to_string(index=False))
    print(f"\nAll checks passed. Results: {output}")


if __name__ == "__main__":
    main()
