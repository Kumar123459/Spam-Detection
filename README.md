---
title: Email Spam Detection
emoji: 📧
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Email Spam Detection Using Machine Learning

A beginner-friendly project that classifies SMS/email text as **Spam** or **Not Spam** using TF-IDF and Multinomial Naive Bayes, with a Flask web interface.

## Requirements

- Python 3.9 or newer
- A labeled dataset saved as `spam.csv`

## Dataset

Download the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) and place the CSV file in this folder as `spam.csv`.

The training script supports either:
- columns named `label` and `message`, or
- common SMS dataset columns `v1` and `v2` (`ham`/`spam` labels).

If your downloaded file is named `SMSSpamCollection` and is tab-separated rather than CSV, convert it to CSV first, with columns `label` and `message`.

## Setup

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
`
### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Project structure

```text
email-spam-detection/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── templates/
    └── index.html
```

## How it works

1. Pandas loads and validates the labeled messages.
2. TF-IDF converts text into numerical features.
3. Multinomial Naive Bayes learns to classify spam and ham.
4. The script prints accuracy, precision, recall, and F1-score.
5. Joblib saves the fitted pipeline as `spam_model.pkl`.
6. Flask loads the model and serves the prediction page.

**Note:** Accuracy depends on the dataset and its quality. The model is a learning/demo project, not a guaranteed email security filter.
