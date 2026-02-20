from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("attendance_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    total = int(request.form['total'])
    attended = int(request.form['attended'])
    previous = float(request.form['previous'])

    bunk_rate = (total - attended) / total

    data = np.array([[total, attended, previous, bunk_rate]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "⚠️ Student likely to fall below 75% attendance."
    else:
        result = "✅ Student likely to maintain 75% attendance."

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)