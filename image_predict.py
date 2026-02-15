import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog

# Load model
model = tf.keras.models.load_model("asl_model.h5")

# Class labels
class_labels = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'DEL','NOTHING','SPACE'
]

# Hide root window
Tk().withdraw()

output_word = ""

print("\nASL Image Word Builder")
print("Select images one by one.")
print("Press Cancel to stop.\n")

while True:
    file_path = filedialog.askopenfilename(
        title="Select ASL image",
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )

    # Stop if user cancels
    if not file_path:
        break

    # Load and preprocess image
    img = image.load_img(file_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    letter = class_labels[predicted_class]

    print(f"Predicted: {letter} (Confidence: {confidence:.2f})")

    # Word logic
    if letter == "SPACE":
        output_word += " "
    elif letter == "DEL":
        output_word = output_word[:-1]
    elif letter != "NOTHING":
        output_word += letter

    print(f"Current word: {output_word}\n")

print("Final word:", output_word)
