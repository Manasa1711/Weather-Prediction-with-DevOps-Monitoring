from flask import Flask, request, render_template, Response
import numpy as np
import joblib
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Load models
models = {
    "KNN": joblib.load("knn_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "Decision Tree": joblib.load("decision_tree_model.pkl"),
    "SVM": joblib.load("svm_model.pkl")
}

scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

# ------------------ PROMETHEUS METRICS ------------------
REQUEST_COUNT = Counter('app_requests_total', 'Total Requests')
ERROR_COUNT = Counter('app_errors_total', 'Total Errors')
REQUEST_TIME = Histogram('app_response_time_seconds', 'Response Time')
ACTIVE_REQUESTS = Gauge('active_requests', 'Active Requests')

# ------------------ ROUTES ------------------

@app.route('/')
def home():
    REQUEST_COUNT.inc()
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    REQUEST_COUNT.inc()
    ACTIVE_REQUESTS.inc()

    try:
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind = float(request.form['wind'])

        # Feature engineering
        temp_diff = 0
        humidity_index = humidity * wind
        rolling = humidity

        features = np.array([[temp, humidity, wind, temp_diff, humidity_index, rolling]])
        scaled = scaler.transform(features)

        model = models["SVM"]   # or allow selection
        prediction = model.predict(scaled)[0]

        result = f"Prediction: {prediction}"

    except Exception as e:
        ERROR_COUNT.inc()
        result = "Error occurred"

    finally:
        REQUEST_TIME.observe(time.time() - start)
        ACTIVE_REQUESTS.dec()

    return render_template('index.html', result=result)


@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)