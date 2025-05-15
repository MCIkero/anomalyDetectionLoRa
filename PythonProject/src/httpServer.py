import time

from flask import Flask, request, jsonify
import json

app = Flask(__name__)

data_file = "data_"+time.strftime("%Y%m%d-%H%M%S")+".txt"

@app.route('/jrz', methods=['POST'])
def receive_data():
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            raise ValueError("Leere oder ungültige JSON-Daten erhalten.")

        with open(data_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(data, ensure_ascii=False) + "\n")

        print("Empfangene Daten gespeichert:", data)
        return jsonify({"status": "success", "received": data}), 200

    except Exception as e:
        return jsonify({"error": "Invalid JSON", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host="192.168.23.100", port=8000)