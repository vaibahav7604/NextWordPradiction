from flask import Flask, render_template, request
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model
model = load_model("next_word_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_len = 167   # same value used during training


def predict_next_words(text, top_k=5):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return []

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len-1,
        padding='pre'
    )

    predictions = model.predict(token_list, verbose=0)[0]

    top_indices = np.argsort(predictions)[-top_k:][::-1]

    results = []

    for index in top_indices:
        word = tokenizer.index_word.get(index, "Unknown")
        prob = predictions[index]
        results.append((word, float(prob)))

    return results


@app.route("/", methods=["GET", "POST"])
def index():

    predictions = []

    if request.method == "POST":
        text = request.form["text"]
        predictions = predict_next_words(text)

    return render_template("index.html", predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)