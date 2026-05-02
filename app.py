from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string

# -----------------------------
# Create Flask app FIRST
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# Text cleaning function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

# -----------------------------
# Home route
# -----------------------------
@app.route('/')
def home():
    return "Fake Review Detection API is running"

# -----------------------------
# Prediction route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']

    review_lower = review.lower()

    # 🔹 Rule-based keyword detection
    promo_keywords = [
        "buy now", "buy immediately", "limited offer", "hurry", "offer",
    "discount", "grab", "deal", "exclusive", "order now", "dont wait",
    "best product ever", "guaranteed", "limited stock", "special discount",
    "act now", "click now", "everyone is buying", "must buy", "dont miss"
    ]










    if any(word in review_lower for word in promo_keywords):
        return jsonify({"result": "Fake"})

    # -----------------------------
    # ML Prediction
    # -----------------------------
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    result = "Genuine" if prediction == 1 else "Fake"

    return jsonify({"result": result})

# -----------------------------
# Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)