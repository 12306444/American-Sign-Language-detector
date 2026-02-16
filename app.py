from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)
app.secret_key = "asl_secret_key"

# Load model
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
        print(f"\n🔵 Action received: {action}")
        print(f"🔵 Session before: word='{session.get('word')}', prediction={session.get('last_prediction')}")

        # Clear word
        if action == "clear":
            print("🔵 Clearing word")
            session["word"] = ""
            session["last_prediction"] = None
            session["last_confidence"] = None
            print(f"✅ Session after clear: word='{session.get('word')}'")
            return redirect(url_for("home"))

        # Add to word using stored prediction
        if action == "add":
            letter = session.get("last_prediction")
            print(f"🔵 Adding letter: {letter}")
            
            if letter:
                if letter == "SPACE":
                    session["word"] += " "
                    print("✅ Added SPACE")
                elif letter == "DEL":
                    session["word"] = session["word"][:-1]
                    print("✅ Deleted last character")
                elif letter != "NOTHING":
                    session["word"] += letter
                    print(f"✅ Added letter: {letter}")
                else:
                    print("ℹ️ NOTHING detected - no action")
            else:
                print("⚠️ No letter to add")
            
            print(f"✅ Word now: '{session['word']}'")
            return redirect(url_for("home"))

        # Predict action
        if action == "predict":
            file = request.files.get('image')
            print(f"🔵 File received: {file.filename if file else 'NO FILE'}")
            
            if file:
                try:
                    # Check file size
                    file.seek(0, os.SEEK_END)
                    file_length = file.tell()
                    file.seek(0)
                    print(f"📁 File size: {file_length} bytes")
                    
                    # Open and preprocess image
                    print("🖼️ Opening image...")
                    img = Image.open(file).convert('RGB')
                    print(f"✅ Image opened: {img.size}, mode: {img.mode}")
                    
                    img = img.resize((64, 64))
                    print(f"✅ Image resized to: {img.size}")
                    
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    print(f"✅ Image array shape: {img_array.shape}, min: {img_array.min()}, max: {img_array.max()}")
                    
                    # Predict
                    print("🤖 Running prediction...")
                    pred = model.predict(img_array, verbose=0)
                    print(f"✅ Prediction raw output shape: {pred.shape}")
                    
                    predicted_class = int(np.argmax(pred))
                    confidence = float(np.max(pred))
                    
                    print(f"📊 Predicted class index: {predicted_class}")
                    print(f"📊 Top 5 predictions:")
                    top5 = np.argsort(pred[0])[-5:][::-1]
                    for i, idx in enumerate(top5):
                        print(f"   {i+1}. {class_labels[idx]}: {pred[0][idx]:.4f}")
                    
                    letter = class_labels[predicted_class]
                    confidence_percent = f"{confidence*100:.1f}%"
                    
                    print(f"🎯 FINAL PREDICTION: {letter} with confidence {confidence_percent}")
                    
                    # Store in session
                    session["last_prediction"] = letter
                    session["last_confidence"] = confidence
                    print(f"✅ Session updated: prediction={letter}, confidence={confidence}")
                    
                except Exception as e:
                    print(f"❌❌❌ ERROR during prediction: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ No file uploaded")
            
            print(f"🔄 Redirecting to home")
            return redirect(url_for("home"))

    # For GET requests, render template
    confidence_display = None
    if session.get("last_confidence"):
        confidence_display = f"{session['last_confidence']*100:.1f}%"
        print(f"📤 Rendering with prediction: {session.get('last_prediction')}, confidence: {confidence_display}")
    
    return render_template(
        'index.html',
        prediction=session.get("last_prediction"),
        confidence=confidence_display,
        word=session.get("word")
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
