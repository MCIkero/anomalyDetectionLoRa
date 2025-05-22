import numpy as np
import pandas as pd
import psutil
import os
import time
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, f1_score, recall_score

from clusterPlot import plot_3d_clusters_animation, plot_static_3d_clusters
from src.others import prepare_output_dir

def NearestNeighbor(file, outputdir, radius_range=(3.0, 5.01, 0.5), min_neighbors=10, save_video=False, show_plot=True, mode="train"):

    print(f"Nearest Neighbor Algorithmus im {mode} Modus gestartet")

    outputdir = os.path.join(outputdir, "NearestNeighbor")
    video_dir = os.path.join(outputdir, "videos")

    features = ["rssi", "snr", "spreading_factor", "time_difference", "frame_counter_diff", "rssi_mean5", "snr_mean5"]
    #, "time_difference", "frame_counter_diff", "rssi_mean5", "snr_mean5"
    performance_log = []
    all_results = []

    file = file.dropna(subset=features)

    if mode == "train":
        prepare_output_dir(outputdir)
        prepare_output_dir(video_dir)

        print("Train-Modus aktiviert")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(file[features])
        joblib.dump(scaler, os.path.join(outputdir, "scaler.pkl"))

        model = NearestNeighbors(radius=1.0)
        model.fit(data_scaled)
        joblib.dump(model, os.path.join(outputdir, "nn_model_train.pkl"))
        print("Modell und Scaler gespeichert.")
        return None, None

    elif mode == "predict":
        print("Predict-Modus aktiviert")

        scaler = joblib.load(os.path.join(outputdir, "scaler.pkl"))
        model = joblib.load(os.path.join(outputdir, "nn_model_train.pkl"))
        data_scaled = scaler.transform(file[features])
        process = psutil.Process(os.getpid())

        for radius in np.arange(*radius_range):
            print(f"\nRadius {round(radius, 3)}:")

            cpu_start = time.process_time()
            ram_before = process.memory_info().rss / 1024 ** 2
            wall_start = time.time()

            neighbors = model.radius_neighbors(data_scaled, radius=radius, return_distance=False)
            all_anomaly_flags = []

            for k in range(min_neighbors-1, min_neighbors, 1):
                anomaly_flags = [1 if len(n) >= k else -1 for n in neighbors]

                cols_to_keep = ["fCnt", "rssi", "snr", "spreading_factor"]
                missing_cols = [col for col in cols_to_keep if col not in file.columns]
                if missing_cols:
                    raise ValueError(f"Spalten fehlen im Eingabe-DataFrame: {missing_cols}")

                df_result = file[cols_to_keep].copy()
                df_result["anomaly"] = anomaly_flags
                df_result["radius"] = round(radius, 3)

                all_results.append(df_result)
                all_anomaly_flags.append(anomaly_flags)

                radius_tag = str(round(radius, 3)).replace('.', '_')
                filename_static = os.path.join(outputdir, f"nn_radius_{radius_tag}_{k}.png")
                filename_video = os.path.join(outputdir, f"videos/nn_radius_{radius_tag}.mp4")

                if show_plot:
                    plot_static_3d_clusters(
                        df=df_result,
                        x="rssi", y="snr", z="spreading_factor",
                        label_column="anomaly",
                        title=f"Nearest Neighbors – Radius {round(radius, 3)}",
                        save_path=filename_static
                    )

                if save_video:
                    plot_3d_clusters_animation(
                        df=df_result,
                        x="rssi", y="snr", z="spreading_factor",
                        label_column="anomaly",
                        title=f"Nearest Neighbors – Radius {round(radius, 3)}",
                        interval=50,
                        save_path=filename_video,
                        frame_range=360
                    )

            elapsed = (time.time() - wall_start) * 1000
            cpu_time = time.process_time() - cpu_start
            ram_after = process.memory_info().rss / 1024 ** 2
            peak_ram = max(ram_before, ram_after)

            last_anomalies = (df_result["anomaly"] == -1).sum()
            last_normals = (df_result["anomaly"] == 1).sum()

            performance_log.append({
                "radius": round(radius, 3),
                "runtime_ms": round(elapsed, 2),
                "cpu_time_s": round(cpu_time, 4),
                "peak_ram_mb": round(peak_ram, 2),
                "anomalies_detected": last_anomalies,
                "normals_detected": last_normals
            })

        perf_df = pd.DataFrame(performance_log)
        perf_df.to_csv(os.path.join(outputdir, "performance_log_nn.csv"), index=False)
        print("\nPerformance-Log gespeichert unter:", os.path.join(outputdir, "performance_log_nn.csv"))

        return all_results, perf_df
    return None


#Evaluation-Funktion für F1 & Precision
def evaluate_nn_results(all_results, df_ground_truth):
    eval_log = []

    for df_result in all_results:
        radius = df_result["radius"].iloc[0]

        #Sicherstellen, dass Spaltennamen übereinstimmen
        df_ground_truth.columns = df_ground_truth.columns.str.strip()
        if "fCnt" not in df_ground_truth.columns and "fcnt" in df_ground_truth.columns:
            df_ground_truth.rename(columns={"fcnt": "fCnt"}, inplace=True)

        merged = df_result.merge(df_ground_truth[["fCnt", "anomaly"]], on="fCnt", how="inner")

        if merged.empty:
            continue

        y_true = merged["anomaly_y"].astype(bool)
        y_pred = merged["anomaly_x"] == -1

        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        eval_log.append({
            "radius": radius,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": len(merged)
        })

    return pd.DataFrame(eval_log)


def NearestNeighborGridSearch(
        file,
        outputdir,
        radius_range=(1.0, 8.1, 0.5),
        min_neighbors_range=(2, 10),
        df_ground_truth=None
):
    print("Starte erweiterte Nearest Neighbor Grid Search")

    all_evaluations = []

    for min_n in range(min_neighbors_range[0], min_neighbors_range[1]):
        print(f"Teste min_neighbors = {min_n}")

        all_results, _ = NearestNeighbor(
            file=file,
            outputdir=outputdir,
            radius_range=radius_range,
            min_neighbors=min_n,
            save_video=False,
            show_plot=False,
            mode="predict"
        )

        eval_df = evaluate_nn_results(all_results, df_ground_truth)
        eval_df["min_neighbors"] = min_n  # mitloggen
        all_evaluations.append(eval_df)

    # Alle Ergebnisse zu einem großen DataFrame zusammenführen
    combined_eval_df = pd.concat(all_evaluations, ignore_index=True)

    # Beste Kombination nach F1-Score finden
    best_row = combined_eval_df.loc[combined_eval_df["f1"].idxmax()]

    print("\nBestes Ergebnis:")
    print(best_row)

    return combined_eval_df, best_row

