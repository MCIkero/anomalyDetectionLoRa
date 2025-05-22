
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
import os
import time
from collections import Counter
from clusterPlot import plot_static_3d_clusters, plot_3d_clusters_animation
from src.others import prepare_output_dir
import psutil
import tracemalloc

def DbscanAlgorithm(file, outputdir, eps_range=(1.0, 3.1, 0.5), min_samples=50, save_video=False, show_plot=True):

    print(f"DBSCAN Algorithmus im gestartet")

    outputdir = os.path.join(outputdir, "DBSCAN")
    prepare_output_dir(outputdir)
    video_dir = os.path.join(outputdir, "videos")
    prepare_output_dir(video_dir)

    features = ["rssi", "snr", "spreading_factor", "time_difference", "frame_counter_diff", "rssi_mean5", "snr_mean5"]
    #, "time_difference", "frame_counter_diff", "rssi_mean5", "snr_mean5"

    #Drop rows with NaNs in the selected features
    file = file.dropna(subset=features)

    #Skalierung
    data_scaled = StandardScaler().fit_transform(file[features])

    performance_log = []
    all_results = []

    for eps in np.arange(*eps_range):
        eps_rounded = round(eps, 5)
        print(f"\nEps: {eps_rounded}")

        #Start Tracking
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        cpu_start = process.cpu_times().user
        start_time = time.time()

        model = DBSCAN(eps=eps_rounded, min_samples=min_samples)
        labels = model.fit_predict(data_scaled)

        #Stop Tracking
        elapsed = (time.time() - start_time) * 1000
        cpu_end = process.cpu_times().user
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        df_result = file.copy()
        df_result["cluster"] = labels
        df_result["anomaly"] = [-1 if label == -1 else 0 for label in labels]
        df_result["eps"] = eps_rounded
        all_results.append(df_result)

        # Clustergrößen analysieren
        cluster_sizes = dict(Counter(labels))

        #Dateinamen vorbereiten
        eps_tag = str(eps_rounded).replace('.', '_')
        filename_static = os.path.join(outputdir, f"dbscan_eps_{eps_tag}.png")
        filename_video = os.path.join(outputdir, f"videos/dbscan_eps_{eps_tag}.mp4")

        #Statischer Plot
        if show_plot:
            plot_static_3d_clusters(
                df=df_result,
                x="rssi",
                y="snr",
                z="spreading_factor",
                label_column="anomaly",
                title=f"DBSCAN – Eps {eps_rounded}",
                save_path=filename_static
            )

        #Animation
        if save_video:
            plot_3d_clusters_animation(
                df=df_result,
                x="rssi",
                y="snr",
                z="spreading_factor",
                label_column="anomaly",
                title=f"DBSCAN – Eps {eps_rounded}",
                interval=50,
                save_path=filename_video,
                frame_range=360
            )

        #Performance
        performance_log.append({
            "eps": eps_rounded,
            "min_samples": min_samples,
            "runtime_ms": round(elapsed, 2),
            "cpu_time_s": round(cpu_end - cpu_start, 4),
            "peak_ram_mb": round(peak / 1024 / 1024, 2),
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "n_noise": (labels == -1).sum(),
            "anomalies_detected": (df_result["anomaly"] == -1).sum(),  # HINZUFÜGEN!
            "normals_detected": (df_result["anomaly"] == 0).sum(),     # HINZUFÜGEN!
            "cluster_sizes": cluster_sizes
        })

    #Log speichern
    perf_df = pd.DataFrame(performance_log)
    perf_df.to_csv(os.path.join(outputdir, "performance_log_DBSCAN.csv"), index=False)
    print("\nPerformance-Log gespeichert unter:", os.path.join(outputdir, "performance_log_DBSCAN.csv"))

    return all_results, perf_df

def evaluate_dbscan_results(all_results, df_ground_truth):
    """
    Berechne F1 und Precision für alle DBSCAN-Ergebnisse vs Ground Truth.

    Args:
        all_results (list of pd.DataFrame): Liste mit Vorhersage-DataFrames aus DBSCAN.
        df_ground_truth (pd.DataFrame): DataFrame mit Ground Truth. Muss 'fCnt' und 'anomaly' enthalten.

    Returns:
        pd.DataFrame mit eps, precision, f1, count
    """
    eval_log = []

    # Spaltennamen ggf. korrigieren
    df_ground_truth.columns = df_ground_truth.columns.str.strip()
    if "fCnt" not in df_ground_truth.columns and "fcnt" in df_ground_truth.columns:
        df_ground_truth.rename(columns={"fcnt": "fCnt"}, inplace=True)

    for df_result in all_results:
        eps = df_result["eps"].iloc[0]

        if "fCnt" not in df_result.columns:
            print(f"Warnung: fCnt fehlt in einem Ergebnis mit eps={eps}")
            continue

        merged = df_result.merge(df_ground_truth[["fCnt", "anomaly"]], on="fCnt", how="inner")

        if merged.empty:
            continue

        y_true = merged["anomaly_y"].astype(bool)
        y_pred = merged["anomaly_x"] == -1

        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        eval_log.append({
            "eps": eps,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": len(merged)
        })

    return pd.DataFrame(eval_log)

def DbscanGridSearch(file, outputdir, eps_range=(0.1, 1.02, 0.5), min_samples_list=[10, 30, 50, 70], df_ground_truth=None):
    all_logs = []

    for min_samples in min_samples_list:
        print(f"\nStarte DBSCAN mit min_samples = {min_samples}")

        dbscan_results, _ = DbscanAlgorithm(
            file.copy(),
            outputdir=outputdir,
            eps_range=eps_range,
            min_samples=min_samples,
            save_video=False,
            show_plot=False
        )

        eval_df = evaluate_dbscan_results(dbscan_results, df_ground_truth)
        eval_df["min_samples"] = min_samples
        all_logs.append(eval_df)

    combined_eval = pd.concat(all_logs, ignore_index=True)
    best_row = combined_eval.loc[combined_eval["f1"].idxmax()]
    print("\nBestes Ergebnis:")
    print(best_row)