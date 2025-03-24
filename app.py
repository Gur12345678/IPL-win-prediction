from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# List of IPL teams
teams = [
    "Sunrisers Hyderabad",
    "Mumbai Indians",
    "Kolkata Knight Riders",
    "Chennai Super Kings",
    "Rajasthan Royals",
    "Delhi Capitals",
    "Lucknow Super Giants",
    "Gujarat Titans"
]

# List of IPL cities
cities = ["Mumbai", "Kolkata", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Jaipur", "Ahmedabad"]

@app.route("/")
def home():
    return render_template("index.html", teams=teams, cities=cities)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        batting_team = request.args.get("batting_team")
        bowling_team = request.args.get("bowling_team")
        city = request.args.get("city")
        runs_left = int(request.args.get("runs_left"))
        balls_left = int(request.args.get("balls_left"))
        wickets = int(request.args.get("wickets"))
        total_runs = int(request.args.get("total_runs"))
        crr = float(request.args.get("crr"))
        rrr = float(request.args.get("rrr"))

        # Impossible scenarios
        if runs_left > balls_left * 6:
            return jsonify({"win": 0.0, "lose": 100.0, "graph": ""})

        # Convert input to DataFrame
        columns = ["batting_team", "bowling_team", "city", "runs_left", "balls_left", "wickets", "total_runs_x", "crr", "rrr"]
        features_df = pd.DataFrame([[batting_team, bowling_team, city, runs_left, balls_left, wickets, total_runs, crr, rrr]], columns=columns)

        # Predict probabilities
        prediction = model.predict_proba(features_df)[0]
        win_prob = round(prediction[1] * 100, 2)
        lose_prob = round(prediction[0] * 100, 2)

        # Adjust for extreme cases
        if runs_left > balls_left * 6:
            win_prob = 0.0
            lose_prob = 100.0

        if runs_left == 0:
            win_prob = 100.0
            lose_prob = 0.0

        # Generate graph for 20 overs only (T20 format)
        overs = list(range(1, 21))
        win_progress = np.linspace(max(0, win_prob - 10), min(100, win_prob + 10), len(overs))
        lose_progress = np.linspace(min(100, lose_prob + 10), max(0, lose_prob - 10), len(overs))

        fig, ax = plt.subplots()
        ax.plot(overs, win_progress, label='Win Probability', color='green')
        ax.plot(overs, lose_progress, label='Lose Probability', color='red')
        ax.set_xlabel("Overs")
        ax.set_ylabel("Probability (%)")
        ax.set_title("Win/Loss Probability Progression (T20)")
        ax.legend()

        # Convert graph to image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        graph_url = f"data:image/png;base64,{graph_url}"

        return jsonify({"win": win_prob, "lose": lose_prob, "graph": graph_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
