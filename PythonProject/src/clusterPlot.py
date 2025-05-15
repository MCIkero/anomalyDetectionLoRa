import os

import pandas as pd
import ast
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import gridspec
from matplotlib_venn import venn2
from matplotlib_venn.layout.venn2 import DefaultLayoutAlgorithm
from matplotlib.animation import FuncAnimation

def get_color_map(labels):
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    color_mapping = {
        label: cmap(i) if label != -1 else (0.6, 0.6, 0.6, 1.0)  # Grau für Rauschen
        for i, label in enumerate(unique_labels)
    }

    return [color_mapping[label] for label in labels]

def plot_3d_clusters_animation(df, x, y, z, label_column, title="3D Plot", interval=50, save_path=None, frame_range=360):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    #Farbzuordnung wie bei der statischen Version
    color_map = {
        7: "#1f77b4",  # Blau
        8: "#ff7f0e",  # Orange
        9: "#2ca02c",  # Grün
        10: "#17becf", # Türkis
        11: "#9467bd", # Violett
        12: "#8c564b"  # Braun
    }

    anomaly_points = df[df["anomaly"] == -1] if "anomaly" in df.columns else pd.DataFrame()

    def update(frame):
        ax.clear()

        #Normale Punkte pro SF
        for sf, color in color_map.items():
            subset = df[df[z] == sf]
            normal_points = subset[subset.get("anomaly", 1) != -1]
            ax.scatter(
                normal_points[x],
                normal_points[y],
                normal_points[z],
                color=color,
                s=10,
                alpha=0.4,
                label=f"SF {sf}"
            )

        #Anomalien extra in Rot drüberzeichnen
        if not anomaly_points.empty:
            ax.scatter(
                anomaly_points[x],
                anomaly_points[y],
                anomaly_points[z],
                color="#FF0000",
                edgecolors="black",
                linewidths=0.5,
                s=10,
                alpha=1.0,
                label="Anomalien"
            )

        ax.set_xlabel(x.upper())
        ax.set_ylabel(y.upper())
        ax.set_zlabel(z.upper())
        ax.set_title(title)
        ax.view_init(elev=30, azim=frame - 60)

        #Legende immer nur einmal erstellen
        handles, labels_ = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Spreading Factor", loc='upper left', bbox_to_anchor=(1.1, 1))

    ani = FuncAnimation(fig, update, frames=range(0, frame_range, 1), interval=interval)

    if save_path:
        ani.save(save_path, writer="ffmpeg", dpi=200)
        print(f"Animation gespeichert unter: {save_path}")
    else:
        plt.show()
        plt.close()


def plot_static_3d_clusters(df, x, y, z, label_column, title="3D Cluster Plot", save_path=None):

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    fig.subplots_adjust(left=0.05, right=0.75)
    ax.set_zlim(7, 12)

    #Farbzuweisung für SF-Layer
    color_map = {
        7: "#1f77b4",  # Blau
        8: "#ff7f0e",  # Orange
        9: "#2ca02c",  # Grün
        10: "#17becf",  # Türkis
        11: "#9467bd",  # Violett
        12: "#8c564b"  # Braun
    }

    for sf, color in color_map.items():
        subset = df[df[z] == sf]

        #Normale Punkte
        normal_points = subset[subset.get("anomaly", 1) != -1]
        ax.scatter(
            normal_points[x],
            normal_points[y],
            normal_points[z],
            color=color,
            s=15,
            alpha=0.4,
            label=f"SF {sf}"
        )


        #Anomalien immer zuletzt rot drüberzeichnen
    if "anomaly" in df.columns:
        anomalies = df[df["anomaly"] == -1]
        if not anomalies.empty:
            ax.scatter(
                anomalies[x],
                anomalies[y],
                anomalies[z],
                color="#FF0000",       # leuchtendes Rot
                edgecolors="black",
                linewidths=0.5,
                s=15,
                alpha=1.0,
                label="Anomalien"
            )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Spreading Factor", loc='upper left', bbox_to_anchor=(1.1, 1))

    ax.set_xlabel(x.upper())
    ax.set_ylabel(y.upper())
    ax.set_zlabel(z.upper())
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot gespeichert unter: {save_path}")
    else:
        plt.show(block=False)
        plt.pause(1)

    plt.close()


def plot_isolationforest_summary(perf_log_path, save_path=None):
    df = pd.read_csv(perf_log_path)

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Contamination')
    ax1.set_ylabel('Anomalien erkannt', color=color)
    ax1.plot(df["contamination"], df["anomalies_detected"], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Laufzeit (ms)', color=color)
    ax2.plot(df["contamination"], df["runtime_milliseconds"], color=color, linestyle='--', marker='x')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Isolation Forest – Anomalien & Laufzeit vs. Contamination")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot gespeichert unter: {save_path}")
    else:
        plt.show()


def plot_dbscan_summary(perf_log_path, save_path=None):
    df = pd.read_csv(perf_log_path)

    # Clustergrößen-Strings in echte Dictionaries konvertieren
    df["cluster_sizes"] = df["cluster_sizes"].apply(ast.literal_eval)

    cluster_counts = []
    noise_ratios = []
    total_points = []

    for clusters in df["cluster_sizes"]:
        n_clusters = sum(1 for label in clusters if int(label) != -1)
        n_noise = clusters.get('-1', 0)
        total = sum(clusters.values())
        noise_ratio = n_noise / total if total > 0 else 0

        cluster_counts.append(n_clusters)
        noise_ratios.append(noise_ratio * 100)  # in Prozent
        total_points.append(total)

    df["n_clusters"] = cluster_counts
    df["noise_ratio"] = noise_ratios

    fig, ax1 = plt.subplots()

    color1 = 'tab:blue'
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Anzahl Cluster', color=color1)
    ax1.plot(df["eps"], df["n_clusters"], color=color1, marker='o', label="Clusteranzahl")
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Rauschanteil (%)', color=color2)
    ax2.plot(df["eps"], df["noise_ratio"], color=color2, marker='x', linestyle='--', label="Noise (%)")
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("DBSCAN – Clusteranzahl & Rauschanteil vs. Epsilon")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot gespeichert unter: {save_path}")
    else:
        plt.show()

def plot_nearestneighbor_summary(perf_log_path, save_path=None):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(perf_log_path)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Radius')
    ax1.set_ylabel('Anomalien erkannt', color=color1)
    ax1.plot(df["radius"], df["anomalies_detected"], color=color1, marker='o', label='Anomalien erkannt')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Laufzeit (ms)', color=color2)
    ax2.plot(df["radius"], df["runtime_ms"], color=color2, marker='x', linestyle='--', label='Laufzeit')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Nearest Neighbor – Anomalien & Laufzeit vs. Radius")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot gespeichert unter: {save_path}")
    else:
        plt.show()

def plot_point_anomalies_plotly(data, sigma_threshold=3, save_path=None):

    data = data.dropna(subset=["rssi", "snr", "time"])
    data = data.sort_values("time")

    #Optional: Downsampling
    if len(data) > 500_000:
        data = data.sample(n=500_000, random_state=42).sort_values("time")

    #Berechne Mittelwert und Standardabweichung
    rssi_mean, rssi_std = data["rssi"].mean(), data["rssi"].std()
    snr_mean, snr_std = data["snr"].mean(), data["snr"].std()

    #Schwellwerte
    rssi_upper = rssi_mean + sigma_threshold * rssi_std
    rssi_lower = rssi_mean - sigma_threshold * rssi_std

    #Anomalien
    rssi_anomalies = data[(data["rssi"] > rssi_upper) | (data["rssi"] < rssi_lower)]

    fig = go.Figure()

    # RSSI-Linie
    fig.add_trace(go.Scatter(x=data["time"], y=data["rssi"],
                             mode='lines',
                             name='RSSI',
                             line=dict(color='green')))

    fig.update_layout(
        title="Punktanomalien in RSSI",
        xaxis_title="Zeit",
        yaxis_title="Wert",
        template="plotly_white",
        legend=dict(x=0, y=1.1, orientation="h")
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Interaktiver Plot gespeichert unter: {save_path}")
    else:
        fig.show()



def plot_anomaly_overlap(df, rule_col="rule_flag", ml_col="ml_flag", save_path=None, type="Venn"):
    rule_set = set(df[df[rule_col]].index)
    ml_set = set(df[df[ml_col]].index)

    only_rule = len(rule_set - ml_set)
    only_ml = len(ml_set - rule_set)
    both = len(rule_set & ml_set)

    # Layout mit Grid: 2 Spalten (Text links, Venn rechts)
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])

    # Linke Textbox
    ax_text = plt.subplot(gs[0])
    ax_text.axis("off")
    ax_text.text(0, 1,
                 f"Regelbasiert(Rot): {only_rule + both}\n"
                 f"Machine Learning(grün): {only_ml + both}\n"
                 f"Hybride (Gelb): {both}",
                 fontsize=14, va='top', ha='left')

    # Rechte Venn-Diagramm-Achse
    ax_venn = plt.subplot(gs[1])
    venn = venn2(
        [rule_set, ml_set],
        ax=ax_venn,
        layout_algorithm=DefaultLayoutAlgorithm(2.0)
    )

    venn.get_patch_by_id('10').set_color('#e74c3c')  # kräftiges Rot
    venn.get_patch_by_id('01').set_color('#27ae60')  # kräftiges Grün
    if venn.get_patch_by_id('11') is not None:
        venn.get_patch_by_id('11').set_color('#f1c40f')  # Gelb (Hybrid)

    for subset in ('10', '01'):
        label = venn.get_label_by_id(subset)
        if label:
            label.set_visible(False)

    # Schriftgrößen angleichen
    for text in venn.set_labels:
        if text:
            text.set_fontsize(14)
    for text in venn.subset_labels:
        if text:
            text.set_fontsize(14)

    plt.suptitle(f"Überlappung erkannter Anomalien mit {type}",
                 fontsize=12, fontweight='bold', y=1.02)
    ax_venn.set_axis_off()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Venn-Diagramm gespeichert unter: {save_path}")
    else:
        plt.show()


def load_and_prepare(filepath, x_param):
    df = pd.read_csv(filepath)

    #Spaltennamen klein schreiben
    df.columns = [col.lower() for col in df.columns]

    # Sicherstellen, dass x_param existiert
    if x_param not in df.columns:
        raise ValueError(f"Fehlende X-Parameter-Spalte '{x_param}' in Datei: {filepath}")

    #Spalten, die in allen drei Dateien existieren
    required_columns = ['runtime_ms', 'cpu_time_s', 'peak_ram_mb', x_param]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Fehlende Spalten {missing_columns} in Datei: {filepath}")

    return df



def plot_model_performance(csv_path, param_col, title="Model Performance", save_path=None):

    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Achse 1 – Laufzeit
    color1 = 'tab:blue'
    ax1.set_xlabel(param_col)
    ax1.set_ylabel('Runtime (ms)', color=color1)
    ax1.plot(df[param_col], df["runtime_ms"], color=color1, marker='o', label="Runtime")
    ax1.tick_params(axis='y', labelcolor=color1)

    # Achse 2 – RAM
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Peak RAM (MB)', color=color2)
    ax2.plot(df[param_col], df["peak_ram_mb"], color=color2, linestyle='--', marker='s', label="Peak RAM")
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(title)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot gespeichert unter: {save_path}")
    else:
        plt.show()

def plot_evaluation_curve(df_eval, title="NN Performance vs Radius", para="radius", save_path=None):
    plt.figure(figsize=(10, 6))

    #Precision
    plt.plot(df_eval[para], df_eval["precision"], label="Precision", marker='o')
    #F1
    plt.plot(df_eval[para], df_eval["f1"], label="F1 Score", marker='s')
    #Recall (falls vorhanden)
    if "recall" in df_eval.columns:
        plt.plot(df_eval[para], df_eval["recall"], label="Recall", marker='^')

    plt.title(title)
    plt.xlabel(para)
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")
    else:
        plt.show()