# Bachelorarbeit – **Anomalie Erkennung in einem LoRaWANNetzwerk**

> *Python 3.12.7 · entwickelt in PyCharm 2024.1*

---

## Projektüberblick <a name="projektüberblick"></a>

Diese Arbeit demonstriert eine **hybride Detektions‑Pipeline** zur Erkennung von Jamming‑, Signal‑ und Zeit­anomalien in LoRaWAN‑Sensornetzen.
Regelbasierte Heuristiken (CRC‑Fehler, RSSI/SNR‑Schwellen u. a.) werden mit **Machine‑Learning‑Verfahren** (Isolation Forest, DBSCAN, Radius‑Nearest‑Neighbor, K‑Means) kombiniert, um sowohl punktuelle Ausreißer als auch Cluster‑Anomalien zuverlässig zu identifizieren.
Das Verfahren wird auf dem **LabData – im Laborbetrieb erzeugte Datensätze** sowie „IKBPrediction“–Echt­messungen evaluiert; Ergebnisse werden als CSV‑Logs, 3‑D‑Plots und interaktive Karten ausgegeben.

---

## Ordnerstruktur <a name="ordnerstruktur"></a>

```
PythonProject/
├─ .idea/                    # PyCharm‑Projektdateien
├─ Data/                     # Roh‑ & Ausgabedaten
│   ├─ trainData/            # Trainings­datensätze
│   ├─ predictData/          # Zu prognostizierende Daten
│   ├─ anomalies/            # Ground‑Truth‑Labels
│   ├─ Jamming Data/         # Spezielle Jamming‑Messreihen
│   ├─ OutputData/           # Generierte Ergebnisse
│   └─ LabData/              # Labor‑Messreihen
├─ src/                      # Python‑Quellcode
│   ├─ main.py               # Entry‑Point – orchestriert gesamten Workflow
│   ├─ anomalyDeclaration.py # Definition & Verwaltung der Anomalie‑Objekte
│   ├─ filtering.py          # Einlesen & Feature‑Engineering
│   ├─ IsolationForestAlgo.py
│   ├─ DBScanAlgo.py
│   ├─ NearestNeighbor.py
│   ├─ Timedif.py            # K‑Means auf Zeitdifferenzen
│   ├─ rulebased.py          # Heuristische Regeln + Heatmap‑Optimierung
│   ├─ clusterPlot.py        # Plot‑Utilities
│   ├─ HttpServer.py         # Mini‑Flask‑Server für REST‑Ausgabe (optional)
│   └─ others.py             # Hilfsfunktionen (I/O, Karten …)
```

---

## Systemvoraussetzungen <a name="systemvoraussetzungen"></a>

| Komponente | Version / Empfehlung                     |
| ---------- | ---------------------------------------- |
| **Python** | 3.12.7 (getestet)                        |
| **pip**    | ≥ 23                                     |
| **FFmpeg** | Für GIF/MP4‑Animationen                  |
| **OS**     | Windows 10/11 oder Linux (Ubuntu 22 LTS) |

### Python‑Abhängigkeiten

Manuelle Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly folium matplotlib-venn psutil joblib tqdm
```

---

## Installation <a name="installation"></a>

```bash
# 1 – Repo klonen / Dateien kopieren
# 2 – virtuelle Umgebung anlegen (optional aber empfohlen)
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate         # Windows

# 3 – Pakete installieren
pip install -r requirements.txt  # oder manuell, s. oben
```

---

## Datenverwaltung <a name="datenverwaltung"></a>

| Pfadkonstante in *main.py* | Bedeutung                                                      |
| -------------------------- | -------------------------------------------------------------- |
| `folderTrain`              | Ordner mit Trainingsdateien (Ground‑Truth bekannt)             |
| `folderPredict`            | Ordner mit Dateien für Vorhersagen                             |
| `IKBPrediction`            | Einzeldatei mit Feldmessungen der Innsbrucker Kommunalbetriebe |
| `outputdir`                | Wurzelverzeichnis für *alle* Ergebnisse                        |

> **Hinweis:** Pfade bei Bedarf in *main.py* an lokale Verzeichnisse anpassen.

---

## Ausführung <a name="ausfuehrung"></a>

```bash
python src/main.py         # startet vollständige Pipeline
```

Standardmäßig werden:

1. **Daten gefiltert & Features erzeugt** (`filtering.filteredData`).
2. **Regelbasierte Anomalien** markiert (`rulebased.rule_based_filter`).
3. **K‑Means** auf Zeitdifferenzen ausgeführt (`Timedif.K_MeansTimedif`).
4. **ML‑Modelle** trainiert/geladen & Predictions erzeugt

   * Isolation Forest
   * Radius‑Nearest‑Neighbor
   * DBSCAN
5. **Hybridlogik** kombiniert Regel‑ und ML‑Flags.
6. **Visualisierungen & Berichte** gespeichert (Plots, Heatmaps, Performance‑Logs, HTML‑Karte).

### Moduswahl

* Alle ML‑Algorithmen können sowohl **train** als auch **predict**.
  Beispiel (nur Isolation Forest trainieren):

  ```python
  IsolationForestExperiment(df_train, outputdir, mode="train")
  ```
* Grid‑Search‑Wrapper (`NearestNeighborGridSearch`, `DbscanGridSearch`) evaluieren Hyper­parameter.

---

## Wichtige Skripte & Module <a name="module"></a>

| Datei                      | Zweck                                                                                                            |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **filtering.py**           | Erkennt Eingabeformat (CSV, NDJSON), bereitet Features (RSSI/SNR‑Mittel, Frame‑Counter‑Diff …) auf               |
| **rulebased.py**           | CRC‑Fehler, RSSI/SNR‑Schwellen, Zeitdeltas … → boolean `rule_flag`; Heatmap‑Optimierung für bestes Schwellenpaar |
| **IsolationForestAlgo.py** | Classic Isolation Forest, optional 3‑D‑Animation, Performance‑Logging                                            |
| **NearestNeighbor.py**     | Radius‑Nearest‑Neighbor zur Cluster‑Outlier‑Detektion                                                            |
| **DBScanAlgo.py**          | DBSCAN mit Grid‑Search & Cluster‑Statistiken                                                                     |
| **Timedif.py**             | K‑Means‑Clustering auf Zeitdifferenzen je Gerät                                                                  |
| **clusterPlot.py**         | Statische & animierte 3‑D‑Plots, Venn‑Diagramme, KPI‑Kurven                                                      |
| **others.py**              | Map‑Erstellung (Folium), Output‑Verzeichnis‑Handling                                                             |

---

## Ergebnisse & Artefakte <a name="ergebnisse"></a>

Nach Abschluss liegen im *OutputData*‑Ordner u. a.:

* **`*/performance_log_*.csv`** – Laufzeit & RAM pro Hyperparameter
* **`.png`‑Plotte** – 3‑D‑Cluster, Heatmaps, KPI‑Kurven, Venn‑Diagramme
* **`.mp4`/`.gif`** – optionale Rotation der 3‑D‑Plots
* **`interactive_layered.html`** – interaktive Karte mit Layer‑Control für einzelne / kombinierte Methoden

---

## Reproduzierbarkeit

1. Lade den oben genannten Datensatz in die entsprechenden *Data*‑Unterordner.
2. Passe Pfadkonstanten in *main.py* an.
3. Führe das Skript aus.
4. Vergleiche erzeugte Logs & Plots mit im Thesis‑Anhang dokumentierten Referenzwerten.

---

## Lizenz / Rechte

Dieses Projekt ist Teil der Bachelorarbeit an der **MCI Management Center Innsbruck**.
Der Code ist für akademische Zwecke freigegeben (CC‑BY‑NC 4.0).
Die verwendeten Datensätze unterliegen den jeweiligen Original­lizenzen.

---

## Kontakt

Fragen & Feedback gern an **\[Vorname Nachname]** (\<student@ mci.edu>).
