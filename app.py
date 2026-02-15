from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
app.secret_key = "asl_secret_key"

# Load model
model = tf.keras.models.load_model("asl_model.h5")

class_labels = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'DEL','NOTHING','SPACE'
]

@app.route('/', methods=['GET', 'POST'])
def home():

    # Initialize session variables
    if "word" not in session:
        session["word"] = ""
    if "last_prediction" not in session:
        session["last_prediction"] = None
    if "last_confidence" not in session:
        session["last_confidence"] = None

    if request.method == 'POST':
        action = request.form.get("action")

        # Clear word
        if action == "clear":
            session["word"] = ""
            session["last_prediction"] = None
            return redirect(url_for("home"))

        # Add to word using stored prediction
        if action == "add":
            letter = session.get("last_prediction")

            if letter:
                if letter == "SPACE":
                    session["word"] += " "
                elif letter == "DEL":
                    session["word"] = session["word"][:-1]
                elif letter != "NOTHING":
                    session["word"] += letter

            return redirect(url_for("home"))

        # Predict action
        if action == "predict":
            file = request.files.get('image')

            if file:
                img = Image.open(file).resize((64, 64))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                pred = model.predict(img_array)
                predicted_class = np.argmax(pred)
                confidence = float(np.max(pred))

                letter = class_labels[predicted_class]

                # Store in session
                session["last_prediction"] = letter
                session["last_confidence"] = confidence

            return redirect(url_for("home"))

    return render_template(
        'index.html',
        prediction=session.get("last_prediction"),
        confidence=session.get("last_confidence"),
        word=session.get("word")
    )

if __name__ == '__main__':
    app.run(debug=True)
