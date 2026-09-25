import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "conv5_block16_concat"

MODEL_1_PATH = os.path.join("models", "best_densenet121_phase2.keras")
MODEL_2_PATH = os.path.join("models", "best_densenet121_model2_phase2.keras")

CLASS_NAMES_MOD1 = ["NORMAL", "PNEUMONIA"]
CLASS_NAMES_MOD2 = ["BACTERIAL", "VIRAL"]


def load_ai_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelul nu există la calea: {model_path}")

    return tf.keras.models.load_model(model_path, compile=False)


model1_triage = load_ai_model(MODEL_1_PATH)
model2_type = load_ai_model(MODEL_2_PATH)


def prepare_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imaginea nu există: {image_path}")

    # Imagine pentru model (224x224)
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    processed_img = preprocess_input(img_array.copy())
    processed_img = np.expand_dims(processed_img, axis=0).astype(np.float32)

    # Imagine pentru afișare și Grad-CAM (dimensiunea originală)
    original_bgr = cv2.imread(image_path)

    if original_bgr is None:
        raise ValueError(f"Nu pot citi imaginea: {image_path}")

    display_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    return processed_img, display_img


def make_gradcam_heatmap(active_model, img_array, pred_index=None):
    try:
        densenet_model = active_model.get_layer("densenet121")
    except Exception:
        raise ValueError(
            "Nu am găsit layer-ul 'densenet121' în model. "
            "Verifică numele backbone-ului din model."
        )

    try:
        last_conv_layer = densenet_model.get_layer(LAST_CONV_LAYER_NAME)
    except Exception:
        raise ValueError(
            f"Nu am găsit layer-ul Grad-CAM: {LAST_CONV_LAYER_NAME}. "
            "Pentru DenseNet121, de obicei este 'conv5_block16_concat'."
        )

    grad_model = tf.keras.models.Model(
        inputs=densenet_model.input,
        outputs=[last_conv_layer.output, densenet_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, densenet_output = grad_model(img_array, training=False)

        x = densenet_output

        dense_start_found = False
        for layer in active_model.layers:
            if layer.name == "densenet121":
                dense_start_found = True
                continue

            if dense_start_found:
                x = layer(x, training=False)

        predictions = x

        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError("Grad-CAM nu a putut calcula gradientul pentru imaginea curentă.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy()


def save_gradcam_overlay(display_img, heatmap, save_path, alpha=0.4):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    heatmap_resized = cv2.resize(
        heatmap,
        (display_img.shape[1], display_img.shape[0])
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = heatmap_color * alpha + display_img.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    cv2.imwrite(
        save_path,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    return save_path


def analyze_image(image_path, gradcam_path):
    processed_img, display_img = prepare_image(image_path)

    pred_1 = model1_triage.predict(processed_img, verbose=0)[0]

    prob_normal = float(pred_1[0])
    prob_pneumonia = float(pred_1[1])

    prob_bacterial = 0.0
    prob_viral = 0.0

    if prob_normal >= prob_pneumonia:
        pred_label = "NORMAL"
        confidence = prob_normal
        active_model = model1_triage
        gradcam_pred_idx = 0

    else:
        pred_2 = model2_type.predict(processed_img, verbose=0)[0]

        prob_bacterial_raw = float(pred_2[0])
        prob_viral_raw = float(pred_2[1])

        prob_bacterial = prob_bacterial_raw * prob_pneumonia
        prob_viral = prob_viral_raw * prob_pneumonia

        if prob_bacterial >= prob_viral:
            pred_label = "BACTERIAL"
            confidence = prob_bacterial
            gradcam_pred_idx = 0
        else:
            pred_label = "VIRAL"
            confidence = prob_viral
            gradcam_pred_idx = 1

        active_model = model2_type

    probabilities = {
        "NORMAL": prob_normal,
        "VIRAL": prob_viral,
        "BACTERIAL": prob_bacterial
    }

    heatmap = make_gradcam_heatmap(
        active_model=active_model,
        img_array=processed_img,
        pred_index=gradcam_pred_idx
    )

    save_gradcam_overlay(
        display_img=display_img,
        heatmap=heatmap,
        save_path=gradcam_path
    )

    return {
        "pred_label": pred_label,
        "confidence": float(confidence),
        "probabilities": probabilities,
        "gradcam_path": gradcam_path
    }