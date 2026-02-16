from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)
app.secret_key = "asl_secret_key"

# CRITICAL FIXES FOR RENDER - Add these lines
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

# Load model with error handling
try:
    model = tf.keras.models.load_model("asl_model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

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
        print(f"🔵 Action received: {action}")
        
        # Check if this is an AJAX request (coming from your JavaScript)
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        # Clear word
        if action == "clear":
            session["word"] = ""
            session["last_prediction"] = None
            session["last_confidence"] = None
            if is_ajax:
                return jsonify({'success': True, 'word': ''})
            return redirect(url_for("home"))

        # Add to word using stored prediction
        if action == "add":
            letter = session.get("last_prediction")
            print(f"🔵 Adding letter: {letter}")

            if letter:
                if letter == "SPACE":
                    session["word"] += " "
                elif letter == "DEL":
                    session["word"] = session["word"][:-1]
                elif letter != "NOTHING":
                    session["word"] += letter

            if is_ajax:
                return jsonify({
                    'success': True, 
                    'word': session["word"],
                    'letter': letter
                })
            return redirect(url_for("home"))

        # Predict action - THIS IS THE CRITICAL PART
        if action == "predict":
            file = request.files.get('image')
            print(f"🔵 File received: {file.filename if file else 'NO FILE'}")

            if file and model is not None:
                try:
                    # Process image
                    img = Image.open(file).convert('RGB')
                    img = img.resize((64, 64))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Predict
                    pred = model.predict(img_array, verbose=0)
                    predicted_class = int(np.argmax(pred))
                    confidence = float(np.max(pred))

                    letter = class_labels[predicted_class]
                    confidence_percent = f"{confidence*100:.1f}%"
                    
                    print(f"✅ Prediction: {letter} with confidence {confidence_percent}")

                    # Store in session
                    session["last_prediction"] = letter
                    session["last_confidence"] = confidence

                    # FOR AJAX REQUESTS - Return JSON
                    if is_ajax:
                        return jsonify({
                            'success': True,
                            'prediction': letter,
                            'confidence': confidence_percent,
                            'word': session.get("word", "")
                        })
                    
                except Exception as e:
                    print(f"❌ Prediction error: {e}")
                    if is_ajax:
                        return jsonify({
                            'success': False,
                            'error': str(e)
                        }), 500

            # For non-AJAX requests or errors, redirect
            return redirect(url_for("home"))

    # For GET requests
    confidence_display = None
    if session.get("last_confidence"):
        confidence_display = f"{session['last_confidence']*100:.1f}%"
    
    return render_template(
        'index.html',
        prediction=session.get("last_prediction"),
        confidence=confidence_display,
        word=session.get("word")
    )

if __name__ == '__main__':
    # Important for Render - bind to 0.0.0.0 and use PORT env variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
