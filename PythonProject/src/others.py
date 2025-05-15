import os
import shutil
import stat
import folium
import pandas as pd
from folium import FeatureGroup
from folium.plugins import MarkerCluster, HeatMap


def handle_remove_readonly(func, path, exc):
    """ Wird aufgerufen, wenn eine Datei oder ein Ordner schreibgeschützt oder blockiert ist. """
    print(f"Zugriff verweigert bei: {path} – versuche Schreibschutz zu entfernen...")
    os.chmod(path, stat.S_IWRITE)  # Schreibschutz entfernen
    try:
        func(path)  # Erneuter Löschversuch
    except Exception as e:
        print(f"Löschen fehlgeschlagen für: {path}\nFehler: {e}")

def prepare_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, onerror=handle_remove_readonly)
        except Exception as e:
            print(f"Fehler beim Entfernen von {file_path}: {e}")


def create_interactive_layered_map(df, save_path="anomalien_interaktiv.html"):
    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=10, control_scale=True)

    anomaly_types = {
        "Regelbasiert": "rule_flag",
        "DBSCAN": "ml_flag_dbscan",
        "Isolation Forest": "ml_flag_iforest",
        "Nearest Neighbor": "ml_flag_nn"
    }

    # Einzellayer
    for name, column in anomaly_types.items():
        if column not in df.columns:
            continue

        sub_df = df[(df[column] == True) & df["latitude"].notnull() & df["longitude"].notnull()]
        if sub_df.empty:
            continue

        marker_group = FeatureGroup(name=f"{name} Marker")
        marker_cluster = MarkerCluster().add_to(marker_group)

        grouped = sub_df.groupby(["device_address", "latitude", "longitude"])
        for (device, lat, lon), group in grouped:
            popup = f"""
            <b>Device:</b> {device}<br>
            <b>{name} Anomalien:</b> {len(group)}
            """
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(marker_cluster)

        marker_group.add_to(m)

        heat_data = list(zip(sub_df["latitude"], sub_df["longitude"]))
        heat_group = FeatureGroup(name=f"{name} Heatmap")
        HeatMap(heat_data).add_to(heat_group)
        heat_group.add_to(m)

    # Kombinierte Layer: Regelbasiert + je ein ML-Verfahren
    ml_methods = ["DBSCAN", "Isolation Forest", "Nearest Neighbor"]
    for ml in ml_methods:
        rule_col = anomaly_types["Regelbasiert"]
        ml_col = anomaly_types[ml]

        if rule_col not in df.columns or ml_col not in df.columns:
            continue

        combined_df = df[
            (df[rule_col] == True) &
            (df[ml_col] == True) &
            df["latitude"].notnull() & df["longitude"].notnull()
        ]
        if combined_df.empty:
            continue

        # Kombinierte Marker
        marker_group = FeatureGroup(name=f"Regelbasiert + {ml} Marker")
        marker_cluster = MarkerCluster().add_to(marker_group)

        grouped = combined_df.groupby(["device_address", "latitude", "longitude"])
        for (device, lat, lon), group in grouped:
            popup = f"""
            <b>Device:</b> {device}<br>
            <b>Gefunden von:</b> Regelbasiert + {ml}<br>
            <b>Anzahl Einträge:</b> {len(group)}
            """
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                icon=folium.Icon(color="darkred", icon="flag")
            ).add_to(marker_cluster)

        marker_group.add_to(m)

        # Kombinierte Heatmap
        heat_data = list(zip(combined_df["latitude"], combined_df["longitude"]))
        heat_group = FeatureGroup(name=f"Regelbasiert + {ml} Heatmap")
        HeatMap(heat_data).add_to(heat_group)
        heat_group.add_to(m)

    #Kartenlayer
    folium.TileLayer('openstreetmap', name='Standard').add_to(m)
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='&copy; OpenStreetMap contributors & CartoDB',
        name='CartoDB Positron'
    ).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='&copy; Esri — Source: Esri, Maxar, Earthstar Geographics',
        name='Esri Satellite'
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    m.save(save_path)
    print(f"Interaktive Karte mit Layern gespeichert unter: {save_path}")



