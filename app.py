from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and vectorizer once, at startup
with open("Model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Please provide a 'message' field"}), 400

    message = data["message"]
    vec = tfidf.transform([message])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]

    return jsonify({
        "message": message,
        "prediction": "spam" if prediction == 1 else "ham",
        "spam_probability": round(float(probability[1]), 4)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)