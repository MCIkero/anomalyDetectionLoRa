import os

import pandas as pd

from src.DBScanAlgo import DbscanAlgorithm, evaluate_dbscan_results, DbscanGridSearch
from src.IsolationForestAlgo import IsolationForestExperiment, evaluate_iforest_results
from src.NearestNeighbor import NearestNeighbor, evaluate_nn_results, NearestNeighborGridSearch
from src.Timedif import K_MeansTimedif
from src.clusterPlot import plot_point_anomalies_plotly, plot_anomaly_overlap, load_and_prepare, plot_evaluation_curve, \
    plot_model_performance, plot_static_3d_clusters
from src.filtering import filteredData
from src.others import create_interactive_layered_map
from src.rulebased import duplicatePayload, rule_based_filter, find_best_rule_thresholds_with_heatmap

csvfilePredict = "../Data/LoED_LoRaWAN_at_edge_dataset/20_02_2019.csv"
folderPredict = "../Data/predictData/"
folderTrain = "../Data/trainData/"

signalNormal1 = "../Data/trainData/signalNormal1.txt"
signalNormal2 = "../Data/trainData/signalNormal2.txt"
jammingNormal = "../Data/trainData/jammingNormal.csv"

outputdir = "../Data/OutputData/"
IKBPrediction = "../Data/InfluxDataIKB.json"

signalPrediction = "../Data/predictData/signalPrediction.txt"
jammingPrediction = "../Data/predictData/jammingPrediction.csv"
jammingPredictionJRZ = "../Data/predictData/jammingPredictionJRZ.txt"


def check(flag = ""):
    print("Regelbasierte Anomalien:", df["rule_flag"].sum())
    print("ML-Anomalien:", df[flag].sum())
    print("Überlappung:", (df["rule_flag"] & df[flag]).sum())

def heatmap(frame=None, outputdir=None, ml_col = "ml_flag"):
    #Regeloptimierung gegen ML-Ergebnisse
    frame, rule_eval_df, best_rule = find_best_rule_thresholds_with_heatmap(
        frame=frame,
        ml_col=ml_col,
        plot_path=os.path.join(outputdir, "rule_heatmap.png")
    )

    #Venn mit optimierter Regel
    plot_anomaly_overlap(
        frame,
        rule_col="rule_flag_optimized",
        ml_col=ml_col,
        save_path=os.path.join(outputdir, "venn_rule_vs_ml.png")
    )

    #Beste Regel speichern
    with open(os.path.join(outputdir, "best_rule.txt"), "w") as f:
        f.write(f"RSSI < {best_rule[0]}, SNR < {best_rule[1]}, F1 = {round(best_rule[2], 4)}")

def startnn():
    #Ground Truth laden
    df_ground_truth = pd.read_json(signalPrediction, lines=True)

    # Optional: CSV-Format unterstützen
    if "Jamming" in df_ground_truth.columns:
        df_ground_truth.rename(columns={"Jamming": "anomaly"}, inplace=True)
        df_ground_truth["anomaly"] = df_ground_truth["anomaly"].apply(lambda x: 1 if x == -1 else 0)

    if "anomaly" in df_ground_truth.columns:
        df_ground_truth["anomaly"] = df_ground_truth["anomaly"].astype(int)
    else:
        raise ValueError("Ground Truth enthält keine 'anomaly'-Spalte!")

    #Nearest Neighbor Anomalien
    #NearestNeighbor(df_train, outputdir, save_video=False, mode="train")

    df_eval_nn, best_radius_config = NearestNeighborGridSearch(
        file=df,
        outputdir=outputdir,
        radius_range=(3.0, 3.01, 0.01),
        min_neighbors_range=(10, 16),
        df_ground_truth=df_ground_truth
    )

    nn_results, nn_perf = NearestNeighbor(df, outputdir, save_video=False, mode="predict")

    if nn_results is not None:
        best_nn_result = nn_results[1]
        nn_anomalies = best_nn_result[best_nn_result["anomaly"] == -1]

        #Flags im Original-DataFrame setzen
        df["ml_flag_nn"] = df.index.isin(nn_anomalies.index)
        check("ml_flag_nn")

        #Venn-Diagramm erzeugen
        plot_anomaly_overlap(df, ml_col="ml_flag_nn", save_path=os.path.join(outputdir, "NearestNeighbor/venn_nn.png"), type="Nearest Neighbor")

        #Bewertung aller NN-Ergebnisse mit Ground Truth
        df_eval = evaluate_nn_results(nn_results, df_ground_truth)
        plot_evaluation_curve(df_eval, title="NN Performance vs Radius", para = "radius",save_path=outputdir+"NearestNeighbor/performance_log_nn.png")



        #Optionaler Plot
        plot_evaluation_curve(df_eval_nn, title="NN Performance vs Radius", para="radius",
                              save_path=os.path.join(outputdir, "nn_grid_plot.png"))

    else:
        print("Keine Nearest Neighbor Ergebnisse gefunden.")


def startDBSCAN():
    df_ground_truth = pd.read_json(signalPrediction, lines=True)

    if "Jamming" in df_ground_truth.columns:
        df_ground_truth.rename(columns={"Jamming": "anomaly"}, inplace=True)
        df_ground_truth["anomaly"] = df_ground_truth["anomaly"].apply(lambda x: 1 if x == -1 else 0)

    if "anomaly" in df_ground_truth.columns:
        df_ground_truth["anomaly"] = df_ground_truth["anomaly"].astype(int)
    else:
        raise ValueError("Ground Truth enthält keine 'anomaly'-Spalte!")

    DbscanGridSearch(
        file=df,
        outputdir=outputdir,
        eps_range=(1.0, 2.01, 0.5),
        min_samples_list=[10, 30, 50, 70, 90],
        df_ground_truth=df_ground_truth
    )

    #DBSCAN Anomalien
    dbscan_results, dbscan_perf = DbscanAlgorithm(df, outputdir, save_video=False)
    df_eval_dbscan = evaluate_dbscan_results(dbscan_results, df_ground_truth)
    if dbscan_results is not None:
        best_dbscan_result = dbscan_results[0]
        dbscan_anomalies = best_dbscan_result[best_dbscan_result["cluster"] == -1]

        #ML-Flag zusätzlich mit DBSCAN-Anomalien kombinieren
        df["ml_flag_dbscan"] = df.index.isin(dbscan_anomalies.index)

        print(f"DBSCAN-Anomalien erkannt: {len(dbscan_anomalies)}")

        check("ml_flag_dbscan")

        plot_anomaly_overlap(df, ml_col="ml_flag_dbscan", save_path=os.path.join(outputdir, "DBSCAN/venn_dbscan.png"), type="DBSCAN")

        plot_evaluation_curve(df_eval_dbscan, title="DBSCAN Performance vs eps", para = "eps",save_path=outputdir+"DBSCAN/performance_log_DBSCAN.png")

    else:
        print("Keine DBSCAN Ergebnisse gefunden.")


def startIsolation():
    # Ground Truth laden
    df_ground_truth = pd.read_json(signalPrediction, lines=True)
    #df_ground_truth = pd.read_csv(jammingPrediction)

    # Optional: CSV-Unterstützung
    if "Jamming" in df_ground_truth.columns:
        df_ground_truth.rename(columns={"Jamming": "anomaly"}, inplace=True)
        df_ground_truth["anomaly"] = df_ground_truth["anomaly"].apply(lambda x: 1 if x == -1 else 0)

    if "anomaly" in df_ground_truth.columns:
        df_ground_truth["anomaly"] = df_ground_truth["anomaly"].astype(int)
    else:
        raise ValueError("Ground Truth enthält keine 'anomaly'-Spalte!")

    # Modell trainieren und vorhersagen
    #IsolationForestExperiment(df_train, outputdir, mode="train")
    iso_results, iso_perf = IsolationForestExperiment(df, outputdir, save_video=False, mode="predict")

    df_eval_if = evaluate_iforest_results(iso_results, df_ground_truth)
    print(df_eval_if.head())
    print(df_eval_if.columns)

    # Index-Abgleich und Evaluierung
    if iso_results is not None:
        pred_df = iso_results[2]
        common_idx = pred_df.index.intersection(df_ground_truth.index)
        if len(common_idx) < len(pred_df):
            print(f"Nur {len(common_idx)} von {len(pred_df)} Vorhersagen stimmen mit Ground Truth überein.")

        y_true = df_ground_truth.loc[common_idx, "anomaly"]
        y_pred = (pred_df.loc[common_idx, "anomaly"] == -1).astype(int)

        from sklearn.metrics import classification_report
        print("\nKlassifikationsbericht für Isolation Forest:")
        print(classification_report(y_true, y_pred, zero_division=0))

        # Ergebnisse übernehmen
        iso_anomalies = pred_df[pred_df["anomaly"] == -1]
        df["ml_flag_iforest"] = df.index.isin(iso_anomalies.index)

        print(f"IsolationForest-Anomalien erkannt: {len(iso_anomalies)}")

        check("ml_flag_iforest")

        plot_anomaly_overlap(
            df,
            ml_col="ml_flag_iforest",
            save_path=os.path.join(outputdir, "IsolationForest/venn_iforest.png"),
            type="IsolationForest"
        )

        plot_evaluation_curve(
            df_eval_if,
            title="IF Performance vs Contamination",
            para="contamination",
            save_path=os.path.join(outputdir, "IsolationForest/performance_log_IF.png")
        )

    else:
        print("Keine Isolation Forest Ergebnisse gefunden.")


if __name__ == '__main__':
    df = filteredData(single_file=IKBPrediction, dir=outputdir)
    df_train = filteredData(folder_path=folderTrain, dir=outputdir)

    if "fcnt" in df.columns and "fCnt" not in df.columns:
        df.rename(columns={"fcnt": "fCnt"}, inplace=True)

    duplicatePayload(df, outputdir)
    K_MeansTimedif(df, outputdir)

    plot_point_anomalies_plotly(df, sigma_threshold=5.7, save_path=os.path.join(outputdir, "point_anomalies.html"))

    # Regelbasierte Anomalien
    df_rules = rule_based_filter(df,outputdir)
    df_rules.to_csv(os.path.join(outputdir, "rule_based_anomalies.csv"), index=False)

    startnn()
    #heatmap(frame=df, outputdir=outputdir+"NearestNeighbor/", ml_col="ml_flag_nn")
    startDBSCAN()
    #heatmap(frame=df, outputdir=outputdir+"DBSCAN/",ml_col="ml_flag_dbscan")
    startIsolation()
    #heatmap(frame=df, outputdir=outputdir+"IsolationForest/",ml_col="ml_flag_iforest")

    df["ml_flag"] = df[["ml_flag_nn", "ml_flag_dbscan", "ml_flag_iforest"]].any(axis=1)
    df["anomaly_combined"] = df["rule_flag"] & df["ml_flag_iforest"]

    create_interactive_layered_map(df, os.path.join(outputdir, "interactive_layered.html"))

    df["hybrid_anomaly_plot"] = df["anomaly_combined"].apply(lambda x: -1 if x == 1 else 1)

    plot_static_3d_clusters(
        df=df,
        x="rssi",
        y="snr",
        z="spreading_factor",
        label_column="hybrid_anomaly_plot",
        title="Hybride Anomalieerkennung – Regelbasiert & ML kombiniert",
        save_path=os.path.join(outputdir, "hybrid_detection_3d_IKB.png")
    )

    load_and_prepare("../Data/OutputData/DBSCAN/performance_log_DBSCAN.csv", "eps")
    load_and_prepare("../Data/OutputData/IsolationForest/performance_log_IF.csv", "contamination")
    load_and_prepare("../Data/OutputData/NearestNeighbor/performance_log_nn.csv", "radius")

    plot_model_performance(
         "../Data/OutputData/IsolationForest/performance_log_IF.csv","contamination",title="Model Performance Isolation Forest",save_path=
        os.path.join(outputdir, "IsolationForest/model_performance.png")
    )
    plot_model_performance(
        "../Data/OutputData/DBSCAN/performance_log_DBSCAN.csv","eps",title="Model Performance DBSCAN",save_path=
        os.path.join(outputdir, "DBSCAN/model_performance.png")
    )
    plot_model_performance(
        "../Data/OutputData/NearestNeighbor/performance_log_nn.csv","radius",title="Model Performance Nearest Neighbor",save_path=
        os.path.join(outputdir, "NearestNeighbor/model_performance.png")
    )
