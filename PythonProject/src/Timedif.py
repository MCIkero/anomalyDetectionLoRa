import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from src.others import prepare_output_dir


def K_MeansTimedif(file, outputdir):
    outputdir = os.path.join(outputdir, "Timedif")
    prepare_output_dir(outputdir)
    video_dir = os.path.join(outputdir, "videos")
    prepare_output_dir(video_dir)

    time_diff = file[['time_difference']].values

    inertia = []
    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(time_diff)
        inertia.append(kmeans.inertia_)

    plt.figure()
    plt.plot(k_values, inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.savefig(os.path.join(outputdir, "ElbowTimedif.png"), dpi=300)
    plt.close()

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    file.loc[:, 'cluster'] = kmeans.fit_predict(time_diff)

    # Durchschnittliche Zeitdifferenz pro Cluster
    cluster_means = file.groupby("cluster")["time_difference"].mean()

    # Den Cluster mit der höchsten durchschnittlichen Zeitdifferenz als "Anomalie-Cluster"
    anomaly_cluster = cluster_means.idxmax()

    file["rule_timedif_cluster"] = (file["cluster"] == anomaly_cluster)

    y_values = np.random.uniform(-1, 1, size=len(file))
    plt.scatter(file["time_difference"], y_values, c=file["cluster"], cmap="viridis", marker="o")
    plt.title("KMeans Clustering")
    plt.xlabel("Zeitdifferenz (s)")
    plt.ylabel("Relative Position (zufällig)")
    plt.colorbar(label="Cluster")
    plt.savefig(os.path.join(outputdir, "Timedif.png"), dpi=200)
    plt.close()

    file.to_csv(os.path.join(outputdir, 'ZeitdifferenzAnomalien.csv'), index=False)

    return file