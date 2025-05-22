import os

from src.others import prepare_output_dir
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score
from itertools import product


def rule_based_filter(df, save_path=None):
    # CRC-Fehler, niedriger RSSI/SNR
    basic_condition = (
            (df["snr"] < -20) |
            (df["rssi"] < -130) |
            (df["crc_error"] == True)
    )
    save_path = os.path.join(save_path, "rulebased")

    # Frame Counter Rücksprünge
    frame_jump_condition = df["frame_counter_diff"] < 0

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["time_diff_device"] = df.groupby("device_address")["time"].diff().dt.total_seconds().fillna(0)

    if "rule_timedif_cluster" in df.columns:
        df["rule_time"] = df["rule_timedif_cluster"]
    else:
        df["rule_time"] = False

    # 99. Perzentil pro Gerät berechnen und jeder Zeile zuweisen
    df["time_thresh_device"] = df.groupby("device_address")["time_diff_device"].transform(lambda x: x.quantile(0.999))

    # Bedingung: Zeitdifferenz == 0 oder größer als gerätespezifisches 99%-Perzentil
    time_condition = (df["time_diff_device"] > df["time_thresh_device"])

    # Verwaiste Geräte (nur 1 Eintrag)
    device_counts = df["device_address"].value_counts()
    orphan_devices = device_counts[device_counts == 1].index
    orphan_condition = df["device_address"].isin(orphan_devices)

    # Spreading Factor-Wechsel (gerätebasiert)
    df["sf_shift"] = df.groupby("device_address")["spreading_factor"].shift(1)
    sf_change_condition = (df["spreading_factor"] != df["sf_shift"])

    # Einzelne Regel-Flags für Analyse
    df["rule_crc"] = (df["crc_error"] == True)
    df["rule_rssi"] = df["rssi"] < -130
    df["rule_snr"] = df["snr"] < -20
    df["rule_fcnt"] = frame_jump_condition
    #df["rule_time"] = time_condition
    df["rule_orphan"] = orphan_condition
    df["rule_sf_change"] = sf_change_condition

    # Gesamtes Flag
    df["rule_flag"] = (
            df["rule_crc"] |
            df["rule_rssi"] |
            df["rule_snr"] |
            df["rule_fcnt"] |
            df["rule_orphan"] |
            df["rule_sf_change"] |
            df["rule_time"]
    )

    # Analyse und Plot
    top_rules = analyze_rule_contributions(df)
    plot_rule_contributions(top_rules, save_path)

    return df[df["rule_flag"] == True].copy()


def duplicatePayload(file, outputdir=None):
    outputdir = os.path.join(outputdir, "rulebased")
    prepare_output_dir(outputdir)
    video_dir = os.path.join(outputdir, "videos")
    prepare_output_dir(video_dir)

    if "physical_payload" not in file.columns:
        print("Spalte 'physical_payload' nicht vorhanden – Duplicate-Check übersprungen.")
        return

    duplicates = file[file.duplicated(subset=['physical_payload'], keep=False)].copy()

    if not duplicates.empty:
        print("Doppelte Einträge für 'physical_payload' gefunden:")
        duplicates.loc[:, 'same_device'] = duplicates.groupby('physical_payload')['device_address'].transform(
            lambda x: x.nunique() == 1)
        duplicates.to_csv(os.path.join(outputdir, "duplicates.csv"), index=False)

        differing_devices = duplicates[~duplicates['same_device']]
        if not differing_devices.empty:
            print("Einige Payloads stammen von unterschiedlichen Geräten:")
            print(differing_devices)
        else:
            print("Alle doppelten Payloads stammen vom selben Gerät.")
    else:
        print("Keine doppelten Einträge für 'physical_payload' gefunden.")


def find_best_rule_thresholds_with_heatmap(
        frame=None,
        rssi_range=(-130, -80, 5),
        snr_range=(-20, 5, 2),
        ml_col="ml_flag",
        plot_path=None,
        return_flag=True,
        verbose=True
):
    """
    Findet die beste Regelkombination aus RSSI und SNR vs. ML-Ergebnis
    und zeigt eine Heatmap der F1-Scores.
    """

    results = []

    for rssi_thresh, snr_thresh in product(range(*rssi_range), range(*snr_range)):
        rule_flag = (
                (frame["rssi"] < rssi_thresh) |
                (frame["snr"] < snr_thresh) |
                (frame["crc_error"] == True)
        )

        precision = precision_score(frame[ml_col], rule_flag, zero_division=0)
        recall = recall_score(frame[ml_col], rule_flag, zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "rssi_thresh": rssi_thresh,
            "snr_thresh": snr_thresh,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values("f1_score", ascending=False).iloc[0]
    rssi_best, snr_best, best_score = best_row["rssi_thresh"], best_row["snr_thresh"], best_row["f1_score"]

    if verbose:
        print("Beste Regel-Kombination:")
        print(f"   - RSSI < {rssi_best}")
        print(f"   - SNR  < {snr_best}")
        print(f"   - F1-Score: {round(best_score, 4)}")

    # Heatmap erstellen
    pivot = results_df.pivot(index="snr_thresh", columns="rssi_thresh", values="f1_score")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "F1-Score"})
    plt.title("F1-Score für Regelkombinationen (vs. ML)")
    plt.xlabel("RSSI-Schwelle")
    plt.ylabel("SNR-Schwelle")
    plt.tight_layout()

    if plot_path:
        plt.savefig(plot_path, dpi=300)
        print(f"Heatmap gespeichert unter: {plot_path}")
    else:
        plt.show()

    if return_flag:
        frame["rule_flag_optimized"] = (
                (frame["rssi"] < rssi_best) |
                (frame["snr"] < snr_best) |
                (frame["crc_error"] == True)
        )

        return frame, results_df, (rssi_best, snr_best, best_score)

    return results_df, (rssi_best, snr_best, best_score)


def analyze_rule_contributions(df):
    rule_columns = [
        "rule_crc", "rule_rssi", "rule_snr",
        "rule_fcnt", "rule_time", "rule_orphan", "rule_sf_change"
    ]
    rule_counts = {col: df[col].sum() for col in rule_columns}
    rule_counts_sorted = dict(sorted(rule_counts.items(), key=lambda x: x[1], reverse=True))

    print("\nRegelbeiträge zur Anomalie-Erkennung:")
    for rule, count in rule_counts_sorted.items():
        print(f" {rule}: {count} Einträge")

    return rule_counts_sorted


def plot_rule_contributions(rule_counts, save_path=None):
    labels = [r.replace("rule_", "").upper() for r in rule_counts.keys()]
    values = list(rule_counts.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color="cornflowerblue")
    plt.title("Regelbeiträge zur Anomalie-Erkennung")
    plt.ylabel("Anzahl erkannter Einträge")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 5, f"{int(height)}", ha='center', va='bottom')

    save_path = os.path.join(save_path, "rulebased_statistic.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot gespeichert unter: {save_path}")
    plt.close()
