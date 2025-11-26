# app.py
from flask import Flask, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, load_img
import numpy as np
import base64
from io import BytesIO
from flask_cors import CORS
import time
import os
import random

# === CONFIG ===
BASE_DIR = "Datasets/anim_images"   # update if your dataset folder is elsewhere
MODEL_PATH = "generator.h5"
LATENT_DIM = 100
IMG_RESOLUTION = (64, 64)   # generator output resolution
CHANNELS = 3
MODEL_NAME = "DCGAN Anime Generator"

# === APP / CORS ===
app = Flask(__name__)
CORS(app)

# === Load model once on startup ===
generator = tf.keras.models.load_model(MODEL_PATH)

def pil_to_base64(img_pil, fmt="PNG"):
    buf = BytesIO()
    img_pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def tensor_to_pil(img_tensor):
    # img_tensor expected in pixels 0..255 (float or int) shape (H,W,C)
    img_tensor = tf.clip_by_value(img_tensor, 0.0, 255.0)
    arr = tf.cast(img_tensor, tf.uint8).numpy()
    # Pillow expects HWC and RGB
    from PIL import Image
    return Image.fromarray(arr)

def compute_metrics(orig, gen):
    # orig, gen: float32 tensors in [0,255], shape (H,W,C)
    # ensure same dtype
    orig = tf.cast(orig, tf.float32)
    gen  = tf.cast(gen, tf.float32)

    # MSE
    mse = tf.reduce_mean(tf.square(orig - gen))
    # Normalize MSE to maximum possible (255^2)
    normalized_mse = mse / (255.0**2)
    accuracy = 1.0 - normalized_mse
    accuracy = tf.clip_by_value(accuracy, 0.0, 1.0)

    # PSNR and SSIM use 0..255 range
    psnr = tf.image.psnr(orig, gen, max_val=255.0)
    ssim = tf.image.ssim(orig, gen, max_val=255.0)

    return {
        "mse": float(mse.numpy()),
        "accuracy": float(accuracy.numpy()),   # 0..1
        "psnr": float(psnr.numpy()),
        "ssim": float(ssim.numpy()),
    }

@app.route("/generate", methods=["GET"])
def generate_only():
    """
    Generates an image and returns base64 image plus metadata.
    """
    start_time = time.time()

    noise = tf.random.normal([1, LATENT_DIM])
    g_img = generator(noise, training=False)           # range [-1,1] if trained like that
    # denormalize to 0..255
    g_img = (g_img * 127.5) + 127.5
    g_img = tf.squeeze(g_img, axis=0)                 # shape (H,W,C)

    img_pil = tensor_to_pil(g_img)
    img_b64 = pil_to_base64(img_pil, fmt="PNG")

    end_time = time.time()
    return jsonify({
        "image": img_b64,
        "model_name": MODEL_NAME,
        "latent_dim": LATENT_DIM,
        "resolution": f"{IMG_RESOLUTION[0]}x{IMG_RESOLUTION[1]}",
        "channels": CHANNELS,
        "generation_time": round(end_time - start_time, 4)
    })


@app.route("/compare", methods=["GET"])
def compare_with_random_original():
    """
    Picks a random image from BASE_DIR, loads it, resizes to generator resolution,
    generates an image, computes difference & metrics, returns three base64 images
    (original, generated, diff) and metrics.
    """
    start_time = time.time()

    # pick a random image from BASE_DIR
    if not os.path.isdir(BASE_DIR):
        return jsonify({"error": f"BASE_DIR not found: {BASE_DIR}"}), 400

    files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(files) == 0:
        return jsonify({"error": "No image files found in dataset folder."}), 400

    chosen = random.choice(files)
    chosen_path = os.path.join(BASE_DIR, chosen)

    # Load original image and resize to target resolution
    pil_orig = load_img(chosen_path, target_size=IMG_RESOLUTION)  # Pillow image
    orig_arr = np.array(pil_orig).astype("float32")               # shape H,W,C in 0..255

    # Generate image
    noise = tf.random.normal([1, LATENT_DIM])
    g_img = generator(noise, training=False)
    g_img = (g_img * 127.5) + 127.5
    g_img = tf.squeeze(g_img, axis=0)    # shape H,W,C

    # If generator output shape differs, resize generated to IMG_RESOLUTION
    g_img_resized = tf.image.resize(g_img, IMG_RESOLUTION, method="bilinear")

    # Ensure orig and gen have same shape & dtype float32
    orig_tensor = tf.cast(orig_arr, tf.float32)
    gen_tensor = tf.cast(g_img_resized, tf.float32)

    # Compute absolute difference map (0..255)
    abs_diff = tf.abs(orig_tensor - gen_tensor)
    # For visualization, amplify differences slightly (optional)
    diff_vis = tf.clip_by_value(abs_diff * 3.0, 0.0, 255.0)  # scale to make differences clearer

    # Compute metrics
    metrics = compute_metrics(orig_tensor, gen_tensor)
    metrics["accuracy_percent"] = round(metrics["accuracy"] * 100.0, 3)
    metrics["psnr_db"] = round(metrics["psnr"], 3)
    metrics["ssim"] = round(metrics["ssim"], 4)

    # Convert to base64 images
    gen_pil = tensor_to_pil(gen_tensor)
    diff_pil = tensor_to_pil(diff_vis)

    orig_b64 = pil_to_base64(pil_orig, fmt="PNG")
    gen_b64  = pil_to_base64(gen_pil, fmt="PNG")
    diff_b64 = pil_to_base64(diff_pil, fmt="PNG")

    end_time = time.time()
    generation_time = round(end_time - start_time, 4)

    return jsonify({
        "original_image": orig_b64,
        "generated_image": gen_b64,
        "diff_image": diff_b64,
        "chosen_filename": chosen,
        "metrics": {
            "mse": round(metrics["mse"], 3),
            "accuracy_percent": metrics["accuracy_percent"],
            "psnr_db": metrics["psnr_db"],
            "ssim": metrics["ssim"]
        },
        "generation_time": generation_time,
        "model_name": MODEL_NAME,
        "latent_dim": LATENT_DIM,
        "resolution": f"{IMG_RESOLUTION[0]}x{IMG_RESOLUTION[1]}",
        "channels": CHANNELS
    })


if __name__ == "__main__":
    # Run on localhost 5000
    print("Starting server... Make sure generator.h5 and dataset folder exist.")
    app.run(debug=True, host="127.0.0.1", port=5000)
