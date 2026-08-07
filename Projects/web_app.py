from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder="4.3_Templates")

# Importing the model and scaler
model = pickle.load(open("Projects/4.2_Models/ridge.pkl", "rb"))
scaler = pickle.load(open("Projects/4.2_Models/scaler.pkl", "rb"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_data", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        new_scaled_data = scaler.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )
        result = model.predict(new_scaled_data)
        return render_template('predict.html',results=result[0])

    else:
        return render_template("predict.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
