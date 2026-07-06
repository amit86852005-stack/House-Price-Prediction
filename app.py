from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


# INR Format Function
def format_inr(amount):
    if amount >= 10000000:
        return f"₹ {amount/10000000:.2f} Crore"
    elif amount >= 100000:
        return f"₹ {amount/100000:.2f} Lakh"
    else:
        return f"₹ {amount:,.2f}"


@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        overall=5,
        area=1500,
        garage=2,
        basement=800,
        year=2000
    )


@app.route("/predict", methods=["POST"])
def predict():

    overall = float(request.form["overall"])
    area = float(request.form["area"])
    garage = float(request.form["garage"])
    basement = float(request.form["basement"])
    year = float(request.form["year"])

    features = np.array([[overall, area, garage, basement, year]])

    prediction = model.predict(features)[0]

    # Prevent negative value
    prediction = max(0, prediction)

    # USD to INR
    prediction = prediction * 86

    prediction = format_inr(prediction)

    return render_template(
        "index.html",
        prediction=f"Estimated House Price: {prediction}",
        overall=overall,
        area=area,
        garage=garage,
        basement=basement,
        year=year
    )


if __name__ == "__main__":
    app.run(debug=True)