import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# ===================== METRICS =====================

def dice_coefficient(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2 * intersection + 1) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_coefficient(y_true, y_pred)
    return bce + (1 - dice)

def iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    intersection = np.sum(y_true * y_pred)

    dice = (2 * intersection + 1) / (np.sum(y_true) + np.sum(y_pred) + 1)
    iou_score = intersection / (np.sum(y_true) + np.sum(y_pred) - intersection + 1e-7)

    return dice, iou_score

def f1_score(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return 2 * (precision * recall) / (precision + recall + 1e-7)

# ===================== LOAD MODEL =====================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "final_model.keras",
        custom_objects={
            "combined_loss": combined_loss,
            "dice_coefficient": dice_coefficient,
            "iou": iou,
            "f1_score": f1_score
        }
    )

model = load_model()

# ===================== UI =====================

st.title("🎬 Scene Cast AI - Face Segmentation")

st.markdown("---")
st.subheader("🎯 Model Info")

st.write("""
- Architecture: U-Net (MobileNetV2 Encoder)
- Input Size: 256×256
- Task: Face Segmentation
""")

st.divider()

# ===================== MODE =====================

mode = st.radio("Select Mode", ["Demo Mode", "Evaluation Mode"])

st.subheader("📊 Metrics")
col1, col2 = st.columns(2)

# default values
col1.metric("Dice Score", "N/A")
col2.metric("IoU Score", "N/A")

if mode == "Demo Mode":
    st.info("Upload a ground truth mask to compute Dice & IoU.")

st.divider()

# ===================== FILE UPLOAD =====================

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

mask_file = None
if mode == "Evaluation Mode":
    mask_file = st.file_uploader("Upload Ground Truth Mask", type=["jpg", "png"])

# ===================== FUNCTIONS =====================

def preprocess(image):
    img = cv2.resize(image, (256, 256))
    img = img / 255.0
    return img

def predict(image):
    img = preprocess(image)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    return np.clip(pred.squeeze(), 0, 1)

def overlay(image, mask):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = mask.squeeze()

    overlay_img = image.copy()
    overlay_img[mask == 1] = [0, 255, 0]

    return overlay_img

# ===================== MAIN =====================

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pred = predict(image)
    pred_mask = (pred > 0.5).astype(np.uint8)

    result = overlay(image, pred_mask)

    st.image(image, caption="Original Image")
    st.image(pred, caption="Predicted Mask")
    st.image(result, caption="Overlay Output")

    # ===================== METRICS =====================

    if mode == "Evaluation Mode" and mask_file:
        mask_bytes = np.asarray(bytearray(mask_file.read()), dtype=np.uint8)
        true_mask = cv2.imdecode(mask_bytes, 0)

        if true_mask is not None:
            true_mask = cv2.resize(true_mask, (256, 256))
            true_mask = (true_mask > 127).astype(np.uint8)

            dice, iou_score = compute_metrics(true_mask, pred_mask)

            col1.metric("Dice Score", f"{dice:.4f}")
            col2.metric("IoU Score", f"{iou_score:.4f}")
        else:
            st.error("Invalid mask file uploaded.")