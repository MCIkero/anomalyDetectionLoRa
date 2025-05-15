import json
import pandas as pd
import os

import os
import pandas as pd

def parse_file(file_path: str) -> pd.DataFrame:
    extension = os.path.splitext(file_path)[1].lower()

    try:
        if extension in [".json", ".txt"]:
            print(f"Verarbeite JSON/NDJSON/Text-Datei: {file_path}")
            try:
                df_try = pd.read_csv(file_path, low_memory=False)
                if '_value' in df_try.columns:
                    print("Datei scheint CSV mit eingebettetem JSON zu sein.")
                    df = preprocess_json_column(df_try)
                else:
                    print("Datei scheint echtes NDJSON zu sein.")
                    df = preprocess_ndjson_file(file_path)
            except Exception:
                df = preprocess_ndjson_file(file_path)

        elif extension == ".csv":
            print(f"Verarbeite CSV-Datei: {file_path}")
            df = pd.read_csv(file_path, low_memory=False, encoding="utf-8-sig")

            if "Local Time" in df.columns and "Jamming" in df.columns:
                df.rename(columns={
                    "Local Time": "time",
                    "deveui": "device_address",
                    "RSSI(dBm)": "rssi",
                    "snrSNR(dB)": "snr",
                    "SF": "spreading_factor",
                    "Temperature (F)": "temperature",
                    "Jamming": "anomaly"
                }, inplace=True)
                df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)
            else:
                if "time" not in df.columns:
                    raise ValueError(f"'{file_path}' hat keine 'time'-Spalte – manuelle Anpassung nötig.")
                if "anomaly" not in df.columns:
                    df["anomaly"] = 0
                if "device_address" not in df.columns:
                    df["device_address"] = df.get("deveui", "-1")
                if "spreading_factor" not in df.columns:
                    df["spreading_factor"] = 7
                if "rssi" not in df.columns:
                    df["rssi"] = -999
                if "snr" not in df.columns:
                    df["snr"] = 0
        else:
            raise ValueError(f"Nicht unterstütztes Dateiformat: {file_path}")

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        if "crc_status" not in df.columns:
            df["crc_status"] = 1
        if "device_address" not in df.columns:
            df["device_address"] = "-1"

        return df

    except Exception as e:
        raise RuntimeError(f"Fehler beim Verarbeiten von {file_path}: {e}")


def filteredData(single_file=None, folder_path=None, dir=None):
    combined_data = []

    if single_file:
        try:
            df_parsed = parse_file(single_file)

            filtered = df_parsed[
                (df_parsed["crc_status"] > -1) &
                (df_parsed["device_address"] != "-1")
                ].copy()

            if not filtered.empty:
                combined_data.append(filtered)

        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei: {e}")


    elif folder_path and os.path.isdir(folder_path):
        print(f"📁 Verarbeite Ordner: {folder_path}")
        all_files = [f for f in os.listdir(folder_path) if f.endswith(('.csv', '.txt', '.json'))]

        for filename in all_files:
            try:
                path = os.path.join(folder_path, filename)
                df = parse_file(path)

                filtered = df[
                    (df["crc_status"] > -1) &
                    (df["device_address"] != "-1")
                    ].copy()

                if not filtered.empty:
                    combined_data.append(filtered)

            except Exception as e:
                print(f"Fehler in Datei {filename}: {e}")
    else:
        print("⚠️ Kein gültiger Pfad übergeben.")
        exit(1)

    if not combined_data:
        print("Keine gültigen Daten vorhanden.")
        exit(1)

    df_combined = pd.concat(combined_data, ignore_index=True)
    print(f"Insgesamt {len(df_combined)} Datensätze verarbeitet.")
    return add_features(df_combined, dir=dir)


def preprocess_json_column(df):
    records = []

    for _, row in df.iterrows():
        try:
            payload = json.loads(row['_value'])
            time = payload.get("time")
            fcnt = payload.get("fCnt")
            spreading_factor = payload.get("txInfo", {}).get("modulation", {}).get("lora", {}).get("spreadingFactor")
            devAddr = payload.get("devAddr")
            object_data = payload.get("object", {})
            temperature = object_data.get("temperature")

            for rx in payload.get("rxInfo", []):
                rssi = rx.get("rssi")
                snr = rx.get("snr")
                crc = rx.get("crcStatus") == "CRC_OK"
                loc = rx.get("location", {})
                lat = loc.get("latitude")
                lon = loc.get("longitude")

                records.append({
                    "time": time,
                    "fcnt": fcnt,
                    "spreading_factor": spreading_factor,
                    "rssi": rssi,
                    "snr": snr,
                    "crc_status": int(crc),
                    "latitude": lat,
                    "longitude": lon,
                    "temperature": temperature,
                    "device_address": devAddr,
                    "anomaly": 0
                })

        except Exception as e:
            print(f"Fehler beim Parsen: {e}")

    return pd.DataFrame(records)

def preprocess_ndjson_file(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                payload = json.loads(line.strip())
                time = payload.get("time")
                fcnt = payload.get("fCnt")
                spreading_factor = payload.get("txInfo", {}).get("dataRate", {}).get("spreadFactor")
                devAddr = payload.get("devEUI")
                anomaly = payload.get("anomaly", False)

                for rx in payload.get("rxInfo", []):
                    rssi = rx.get("rssi")
                    snr = rx.get("loRaSNR")
                    lat = rx.get("latitude")
                    lon = rx.get("longitude")

                    records.append({
                        "time": time,
                        "fcnt": fcnt,
                        "spreading_factor": spreading_factor,
                        "rssi": rssi,
                        "snr": snr,
                        "crc_status": 1,
                        "latitude": lat,
                        "longitude": lon,
                        "temperature": None,
                        "device_address": devAddr,
                        "anomaly": int(anomaly)
                    })
            except Exception as e:
                print(f"Fehler beim Parsen der NDJSON-Zeile: {e}")
    return pd.DataFrame(records)

def add_features(data, dir=None):
    data.loc[:, 'time'] = pd.to_datetime(data['time'], errors='coerce')
    data = data.dropna(subset=["time"])

    if "anomaly" not in data.columns:
        data["anomaly"] = 0

    data.loc[:, 'time_difference'] = data['time'].diff().dt.total_seconds().fillna(0)

    for col in ['rssi', 'snr', 'fcnt', 'crc_status']:
        data.loc[:, col] = pd.to_numeric(data[col], errors='coerce')

    data.loc[:, 'frame_counter_diff'] = data.groupby('device_address')['fcnt'].diff().fillna(0)
    data["rssi_mean5"] = data.groupby("device_address")["rssi"].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    data["snr_mean5"] = data.groupby("device_address")["snr"].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    data.loc[:, 'crc_error'] = data['crc_status'] == 0

    if dir:
        os.makedirs(dir, exist_ok=True)
        data.to_csv(os.path.join(dir, 'EnhancedData.csv'), index=False)

    return data
