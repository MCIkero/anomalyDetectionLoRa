import json

setting = False
file = "../Data/trainData/signalNormal2.txt"

# Datei einlesen
with open(file, "r") as f:
    data_entries = [json.loads(line) for line in f]

# anomaly-Feld setzen
for entry in data_entries:
    entry["anomaly"] = setting

# Wieder in dieselbe Datei schreiben (überschreibt Inhalt!)
with open(file, "w") as f:
    for entry in data_entries:
        json.dump(entry, f)
        f.write("\n")

print(f"Alle Einträge wurden mit 'anomaly': {setting} versehen.")