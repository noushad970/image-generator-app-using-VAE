from flask import Flask, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np
import base64
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow frontend access

# Load trained GAN generator
generator = tf.keras.models.load_model("generator.h5")

LATENT_DIM = 100

@app.route("/generate", methods=["GET"])
def generate_anime():
    # Generate random noise
    noise = tf.random.normal([1, LATENT_DIM])
    g_img = generator(noise, training=False)

    # Convert back to normal image range
    g_img = (g_img * 127.5) + 127.5
    img = array_to_img(g_img[0])

    # Convert image -> Base64 so browser can read it
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({"image": img_str})

app.run(debug=True, host="127.0.0.1", port=5000)
