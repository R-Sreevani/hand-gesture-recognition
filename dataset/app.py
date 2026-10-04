import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image
import mediapipe as mp
import os

# ============================================================
#                    LUXURY UI  (UNCHANGED)
# ============================================================
st.set_page_config(
    page_title="GestureAI • Luxury Hand Recognition",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- REALISTIC BACKGROUND ---------- */
    .stApp {
        background:
            linear-gradient(rgba(0, 0, 0, 0.82), rgba(0, 0, 0, 0.92)),
            url('https://images.unsplash.com/photo-1519681393784-d120267933ba?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #f5f5f0;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background:
            radial-gradient(circle at 20% 20%, rgba(212, 175, 55, 0.10) 0%, transparent 45%),
            radial-gradient(circle at 80% 80%, rgba(212, 175, 55, 0.08) 0%, transparent 45%);
        pointer-events: none;
        z-index: 0;
    }

    header[data-testid="stHeader"] { background: transparent; }
    footer, #MainMenu { visibility: hidden; }

    /* ---------- HERO ---------- */
    .hero {
        text-align: center;
        padding: 50px 20px 20px 20px;
        animation: fadeDown 1s ease-out;
        position: relative;
        z-index: 1;
    }

    @keyframes fadeDown {
        from { opacity: 0; transform: translateY(-25px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .badge {
        display: inline-block;
        padding: 8px 22px;
        border-radius: 100px;
        background: rgba(212, 175, 55, 0.10);
        border: 1px solid rgba(212, 175, 55, 0.55);
        color: #d4af37;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 26px;
        box-shadow: 0 0 25px rgba(212, 175, 55, 0.25);
        backdrop-filter: blur(10px);
    }

    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: 5rem;
        font-weight: 900;
        line-height: 1.05;
        margin: 0;
        background: linear-gradient(135deg,
            #ffffff 0%, #f4e4bc 25%, #d4af37 50%, #f4e4bc 75%, #ffffff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        text-shadow: 0 0 60px rgba(212, 175, 55, 0.35);
        animation: shine 6s linear infinite;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .hero p {
        font-size: 1.15rem;
        color: #b8b8b0;
        max-width: 620px;
        margin: 22px auto 0 auto;
        line-height: 1.7;
        font-weight: 300;
    }

    .divider {
        width: 120px;
        height: 2px;
        margin: 30px auto;
        background: linear-gradient(90deg, transparent, #d4af37, transparent);
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.7);
    }

    /* ---------- STATS ---------- */
    .stats {
        display: flex;
        justify-content: center;
        gap: 80px;
        margin: 30px auto 40px auto;
        animation: fadeUp 1s ease-out 0.2s both;
        position: relative;
        z-index: 1;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .stat-item { text-align: center; }
    .stat-value {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f4e4bc, #d4af37, #f4e4bc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(212, 175, 55, 0.4);
    }
    .stat-label {
        font-size: 0.7rem;
        color: #8a8a80;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-top: 6px;
        font-weight: 500;
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        justify-content: center;
        background: transparent;
        border-bottom: 1px solid rgba(212, 175, 55, 0.25);
        padding-bottom: 0;
        margin-bottom: 25px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 15, 15, 0.6);
        border: 1px solid rgba(212, 175, 55, 0.35);
        border-radius: 12px 12px 0 0;
        color: #d8d4c0 !important;
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 2px;
        padding: 14px 32px !important;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(212, 175, 55, 0.15);
        border-color: #d4af37;
        color: #d4af37 !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.25), rgba(184, 148, 31, 0.15)) !important;
        border-color: #d4af37 !important;
        color: #f4e4bc !important;
        box-shadow: 0 -4px 30px rgba(212, 175, 55, 0.4);
    }

    /* ---------- UPLOAD CARD ---------- */
    [data-testid="stFileUploader"] {
        background: rgba(15, 15, 15, 0.75);
        border: 1.5px solid rgba(212, 175, 55, 0.45);
        border-radius: 20px;
        padding: 30px;
        backdrop-filter: blur(20px);
        transition: all 0.35s ease;
        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.6),
            0 0 40px rgba(212, 175, 55, 0.15);
        position: relative;
        z-index: 1;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #d4af37;
        box-shadow:
            0 20px 80px rgba(0, 0, 0, 0.7),
            0 0 60px rgba(212, 175, 55, 0.45);
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div {
        color: #d8d4c0 !important;
    }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #d4af37, #b8941f) !important;
        color: #0a0a0a !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 26px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 1px;
        box-shadow: 0 4px 25px rgba(212, 175, 55, 0.5);
    }

    /* ---------- CAMERA INPUT ---------- */
    [data-testid="stCameraInput"] {
        background: rgba(15, 15, 15, 0.75);
        border: 1.5px solid rgba(212, 175, 55, 0.45);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(20px);
        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.6),
            0 0 40px rgba(212, 175, 55, 0.15);
    }

    [data-testid="stCameraInput"] button {
        background: linear-gradient(135deg, #d4af37, #b8941f) !important;
        color: #0a0a0a !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        box-shadow: 0 4px 25px rgba(212, 175, 55, 0.5);
    }

    /* ---------- IMAGE FRAME ---------- */
    [data-testid="stImage"] img {
        border-radius: 16px;
        border: 2px solid rgba(212, 175, 55, 0.55);
        box-shadow:
            0 25px 60px rgba(0, 0, 0, 0.7),
            0 0 50px rgba(212, 175, 55, 0.35),
            0 0 0 6px rgba(212, 175, 55, 0.08);
        padding: 4px;
        background: #0a0a0a;
    }

    [data-testid="stCaptionContainer"] {
        color: #a8a89e !important;
        text-align: center;
        font-size: 0.8rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 500;
        margin-top: 12px;
    }

    /* ---------- RESULT PANEL ---------- */
    [data-testid="stAlert"] {
        background:
            linear-gradient(135deg,
                rgba(20, 18, 10, 0.95) 0%,
                rgba(40, 34, 15, 0.95) 50%,
                rgba(20, 18, 10, 0.95) 100%) !important;
        color: #f4e4bc !important;
        border: 2px solid #d4af37 !important;
        border-radius: 18px !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        letter-spacing: 2px;
        text-align: center;
        padding: 34px 24px !important;
        backdrop-filter: blur(20px);
        box-shadow:
            0 0 0 1px rgba(212, 175, 55, 0.4),
            0 0 40px rgba(212, 175, 55, 0.55),
            0 0 90px rgba(212, 175, 55, 0.25);
        text-shadow: 0 0 20px rgba(212, 175, 55, 0.6);
        animation: goldPulse 2.5s ease-in-out infinite;
    }

    [data-testid="stAlert"]::before {
        content: "◆  RESULT  ◆";
        display: block;
        font-size: 0.7rem;
        letter-spacing: 6px;
        color: #d4af37;
        margin-bottom: 16px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }

    @keyframes goldPulse {
        0%, 100% {
            box-shadow:
                0 0 0 1px rgba(212, 175, 55, 0.4),
                0 0 30px rgba(212, 175, 55, 0.4);
        }
        50% {
            box-shadow:
                0 0 0 1px rgba(212, 175, 55, 0.75),
                0 0 60px rgba(212, 175, 55, 0.8),
                0 0 120px rgba(212, 175, 55, 0.4);
        }
    }

    h4 {
        color: #d4af37 !important;
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        letter-spacing: 2px;
        font-size: 1.2rem !important;
        margin-bottom: 14px !important;
    }

    .block-container {
        padding-top: 1rem !important;
        max-width: 1150px;
        position: relative;
        z-index: 1;
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(#d4af37, #8b6f1e);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)
# ============================================================
#                 END LUXURY UI
# ============================================================


# ============================================================
#         HAND DETECTION SETUP (MediaPipe)
# ============================================================
mp_hands = mp.solutions.hands


def extract_landmarks(image_bgr):
    """
    Returns a 63-dim landmark vector or None.
    Creates a FRESH detector per call to avoid state caching.
    """
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    )
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return coords
    finally:
        hands_detector.close()


def detect_and_crop_hand(image_bgr):
    """
    Detect hand in image and return cropped hand region.
    Falls back to full image if no hand found.
    """
    h, w = image_bgr.shape[:2]
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    )
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]

            pad = 0.2
            x_min = max(0, int((min(xs) - pad * (max(xs) - min(xs))) * w))
            x_max = min(w, int((max(xs) + pad * (max(xs) - min(xs))) * w))
            y_min = max(0, int((min(ys) - pad * (max(ys) - min(ys))) * h))
            y_max = min(h, int((max(ys) + pad * (max(ys) - min(ys))) * h))

            if x_max > x_min and y_max > y_min:
                return image_bgr[y_min:y_max, x_min:x_max], True

        return image_bgr, False
    finally:
        hands_detector.close()


def extract_features(image_bgr):
    hand_crop, found = detect_and_crop_hand(image_bgr)
    hand_crop = cv2.resize(hand_crop, (128, 128))
    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2)
    )

    return features, found


# Load trained landmark model
model = joblib.load(r"D:\Hand_Gesture_Project\gesture_model_landmarks_v2.pkl")


# Helper: predict from a PIL image using landmarks
def predict_gesture(pil_image):
    image_array = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    landmarks = extract_landmarks(image_bgr)

    if landmarks is None:
        return None, False

    prediction = model.predict([landmarks])[0]
    gesture = str(prediction)

    return gesture, True


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="badge">◆  DIP & ML BASED GESTURE CLASSIFIER  ◆</div>
    <h1>Gesture Recognition</h1>
    <div class="divider"></div>
    <p>Upload or capture a hand gesture and let our intelligent system identify it instantly with precision and elegance.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats">
    <div class="stat-item"><div class="stat-value">4</div><div class="stat-label">Gestures</div></div>
    <div class="stat-item"><div class="stat-value">21</div><div class="stat-label">Landmarks</div></div>
    <div class="stat-item"><div class="stat-value">SVM</div><div class="stat-label">Model</div></div>
    <div class="stat-item"><div class="stat-value">95%+</div><div class="stat-label">Accuracy</div></div>
</div>
""", unsafe_allow_html=True)


# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📤  UPLOAD IMAGE", "📷  LIVE CAMERA"])


# ============================================================
#                TAB 1 — UPLOAD
# ============================================================
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Drop your gesture image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="uploader"
        )

    with col2:
        st.markdown("#### 🎯 Prediction")
        result_upload = st.empty()

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        gesture, hand_found = predict_gesture(image)

        with result_upload.container():
            if not hand_found:
                st.warning("⚠️ No hand detected. Please upload a clear hand gesture image.")
            else:
                st.success("Predicted Gesture: " + gesture.replace("_", " ").title())


# ============================================================
#                TAB 2 — LIVE CAMERA
# ============================================================
with tab2:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📷 Capture Gesture")
        camera_image = st.camera_input("Take a picture", key="camera")

    with col2:
        st.markdown("#### 🎯 Prediction")
        result_camera = st.empty()

    if camera_image is not None:
        cam_image = Image.open(camera_image)
        st.image(cam_image, caption="Captured Image", use_container_width=True)

        gesture, hand_found = predict_gesture(cam_image)

        with result_camera.container():
            if not hand_found:
                st.warning("⚠️ No hand detected. Please show your hand clearly to the camera.")
            else:
                st.success("Predicted Gesture: " + gesture.replace("_", " ").title())