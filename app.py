from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "spam_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "spam_model.pkl was not found. First run: python train_model.py"
    )

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            result = int(model.predict([message])[0])
            prediction = "Spam" if result == 1 else "Not Spam"

    return render_template(
        "index.html", prediction=prediction, message=message
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
