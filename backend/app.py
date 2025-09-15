from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, numpy as np, joblib
from datetime import datetime
import tensorflow as tf

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(BASE, "landmarks")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(LANDMARKS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "landmarks_model.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")

def save_landmarks(label, landmarks):
    label_dir = os.path.join(LANDMARKS_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    fname = os.path.join(label_dir, f"{label}_{ts}.npy")
    arr = np.array(landmarks, dtype=np.float32)
    np.save(fname, arr)
    return fname

@app.route('/')
def index():
    # serve frontend index (useful if you want Flask to serve static files)
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except Exception:
        return jsonify({"status":"backend running"})

@app.route('/upload_landmarks', methods=['POST'])
def upload_landmarks():
    data = request.get_json()
    if not data or 'label' not in data or 'landmarks' not in data:
        return jsonify({'error':'label and landmarks required'}), 400
    label = str(data['label']).strip()
    landmarks = data['landmarks']
    try:
        save_landmarks(label, landmarks)
        return jsonify({'message':'saved','label':label}), 200
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/count', methods=['GET'])
def count():
    d = {}
    for label in os.listdir(LANDMARKS_DIR):
        p = os.path.join(LANDMARKS_DIR, label)
        if os.path.isdir(p):
            d[label] = len([f for f in os.listdir(p) if f.endswith('.npy')])
    return jsonify(d), 200

@app.route('/train_landmarks', methods=['POST'])
def train_landmarks():
    X, y = [], []
    labels = sorted([d for d in os.listdir(LANDMARKS_DIR) if os.path.isdir(os.path.join(LANDMARKS_DIR, d))])
    if len(labels) < 2:
        return jsonify({'error':'Need at least 2 labels with samples'}), 400
    label_map = {i: labels[i] for i in range(len(labels))}
    for idx, label in enumerate(labels):
        p = os.path.join(LANDMARKS_DIR, label)
        for f in os.listdir(p):
            if f.endswith('.npy'):
                arr = np.load(os.path.join(p,f))
                X.append(arr.flatten())
                y.append(idx)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=30, batch_size=16, verbose=1)
    model.save(MODEL_PATH)
    joblib.dump(label_map, LABEL_MAP_PATH)
    return jsonify({'message':'trained','classes':label_map}), 200

@app.route('/predict_landmarks', methods=['POST'])
def predict_landmarks():
    data = request.get_json()
    if not data or 'landmarks' not in data:
        return jsonify({'error':'landmarks required'}), 400
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        return jsonify({'error':'model not trained'}), 400
    model = tf.keras.models.load_model(MODEL_PATH)
    label_map = joblib.load(LABEL_MAP_PATH)
    lm = np.array(data['landmarks'], dtype=np.float32).flatten().reshape(1,-1)
    preds = model.predict(lm)[0]
    idx = int(preds.argmax())
    return jsonify({'prediction': label_map.get(idx, str(idx)), 'confidence': float(preds[idx])}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render usa PORT, local usa 5000
    app.run(host="0.0.0.0", port=port, debug=True)
