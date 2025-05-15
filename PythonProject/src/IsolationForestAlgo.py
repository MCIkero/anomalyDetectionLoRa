import os
import time

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.clusterPlot import plot_static_3d_clusters, plot_3d_clusters_animation
from src.others import prepare_output_dir


def IsolationForestExperiment(file, outputdir, contamination_range=(0.005, 0.05, 0.005), save_video=False, show_plot=True, mode="train"):

    print(f"Isolation Forest Algorithmus im {mode} Modus gestartet")

    outputdir = os.path.join(outputdir, "IsolationForest")
    video_dir = os.path.join(outputdir, "videos")

    features = ["rssi", "snr", "spreading_factor", "time_difference", "frame_counter_diff", "rssi_mean5", "snr_mean5"]
    # , "time_difference", "frame_counter_diff", "rssi_mean5", "snr_mean5"
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

        model = IsolationForest()
        model.fit(data_scaled)
        joblib.dump(model, os.path.join(outputdir, "iforest_model_train.pkl"))
        print("Modell und Scaler gespeichert.")
        return None, None

    elif mode == "predict":
        print("Predict-Modus aktiviert")
        scaler = joblib.load(os.path.join(outputdir, "scaler.pkl"))
        model = joblib.load(os.path.join(outputdir, "iforest_model_train.pkl"))
        data_scaled = scaler.transform(file[features])

        process = psutil.Process(os.getpid())

        for contamination in np.arange(*contamination_range):
            contamination_rounded = round(contamination, 3)
            print(f"\nContamination {contamination_rounded}:")
            model.set_params(contamination=contamination_rounded)

            #Zeit & Ressourcenmessung
            cpu_start = time.process_time()
            ram_before = process.memory_info().rss / 1024 ** 2  # MB
            start_time = time.time()

            preds = model.fit_predict(data_scaled)

            cpu_end = time.process_time()
            ram_after = process.memory_info().rss / 1024 ** 2  # MB
            peak_ram = max(ram_before, ram_after)
            elapsed_ms = (time.time() - start_time) * 1000
            cpu_time = cpu_end - cpu_start

            df_result = file.copy()
            df_result["anomaly"] = preds
            df_result["contamination"] = contamination_rounded
            all_results.append(df_result)

            #Plot
            contamination_tag = str(contamination_rounded).replace('.', '_')
            filename_static = os.path.join(outputdir, f"iforest_contam_{contamination_tag}.png")
            filename_video = os.path.join(outputdir, f"videos/iforest_contam_{contamination_tag}.mp4")

            if show_plot:
                plot_static_3d_clusters(
                    df=df_result,
                    x="rssi",
                    y="snr",
                    z="spreading_factor",
                    label_column="spreading_factor",
                    title=f"Isolation Forest – Contamination {contamination_rounded}",
                    save_path=filename_static
                )

            if save_video:
                plot_3d_clusters_animation(
                    df=df_result,
                    x="rssi",
                    y="snr",
                    z="spreading_factor",
                    label_column="spreading_factor",
                    title=f"Isolation Forest – Contamination {contamination_rounded}",
                    interval=50,
                    save_path=filename_video,
                    frame_range=360
                )

            #Performance-Messung speichern
            performance_log.append({
                "contamination": contamination_rounded,
                "runtime_ms": round(elapsed_ms, 2),
                "cpu_time_s": round(cpu_time, 4),
                "peak_ram_mb": round(peak_ram, 2),
                "anomalies_detected": (df_result["anomaly"] == -1).sum(),
                "normals_detected": (df_result["anomaly"] == 1).sum()
            })

        perf_df = pd.DataFrame(performance_log)
        perf_df.to_csv(os.path.join(outputdir, "performance_log_IF.csv"), index=False)
        print("\nPerformance-Log gespeichert unter:", os.path.join(outputdir, "performance_log.csv"))
        return all_results, perf_df

    else:
        raise ValueError("Ungültiger Modus – verwende 'train' oder 'predict'")


def evaluate_iforest_results(all_results, df_ground_truth):
    """
    Berechne F1 und Precision für alle Isolation Forest Ergebnisse vs Ground Truth.

    Args:
        all_results (list of pd.DataFrame): Liste mit Vorhersage-DataFrames aus Isolation Forest.
        df_ground_truth (pd.DataFrame): DataFrame mit Ground Truth. Muss 'fCnt' und 'anomaly' enthalten.

    Returns:
        pd.DataFrame mit contamination, precision, f1, count
    """
    eval_log = []

    # Fix mögliche falsche Spaltennamen
    df_ground_truth.columns = df_ground_truth.columns.str.strip()
    if "fCnt" not in df_ground_truth.columns and "fcnt" in df_ground_truth.columns:
        df_ground_truth.rename(columns={"fcnt": "fCnt"}, inplace=True)

    for df_result in all_results:
        contamination = df_result["contamination"].iloc[0]

        if "fCnt" not in df_result.columns:
            print(f"Warnung: fCnt fehlt in einem Ergebnis mit contamination={contamination}")
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
            "contamination": contamination,
            "precision": precision,
            "recall": recall,  #hier
            "f1": f1,
            "count": len(merged)
        })

    return pd.DataFrame(eval_log)