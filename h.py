import streamlit as st
import pandas as pd
import os
from datetime import datetime
import traceback
import time
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr
import requests # Needed to download models

# --- 1. GLOBAL CONFIGURATION ---
st.set_page_config(
    page_title="Threat_Detection AI",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CONFIGURATION CONSTANTS ---
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY" 

# --- 2. AUTHENTICATION SYSTEM (Modified with Registration and Styling) ---
DB_FILE = "users_db.csv"
ERROR_LOG = "login_error.log"

def log_error(exc: Exception):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().isoformat()}] {repr(exc)}\n")
        f.write(traceback.format_exc())

def load_users():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["aadhaar", "dob"], dtype=str)
        df.to_csv(DB_FILE, index=False)
        return df
    return pd.read_csv(DB_FILE, dtype=str).fillna("").astype(str)

def save_user(aadhaar, dob):
    df = load_users()
    if aadhaar in df["aadhaar"].values: return False, "User already registered."
    # Basic validation for Aadhaar length (12 digits)
    if len(str(aadhaar).strip()) != 12: return False, "Aadhaar must be exactly 12 digits."
    if not dob: return False, "DOB cannot be empty."

    new_row = pd.DataFrame([[aadhaar, dob]], columns=["aadhaar", "dob"], dtype=str)
    pd.concat([df, new_row], ignore_index=True).to_csv(DB_FILE, index=False)
    return True, "Registration successful! You can now log in."

def validate_login(aadhaar, dob):
    df = load_users()
    return ((df["aadhaar"] == aadhaar) & (df["dob"] == dob)).any()

# --- Custom CSS for Attractive Auth Page ---
st.markdown("""
    <style>
    /* General App Background */
    .stApp { background-color: #020617; color: #e2e8f0; }
    
    /* Blue Gradient Button Style (Used for Login/Register) */
    div.stButton > button { 
        background: linear-gradient(90deg, #0891b2, #2563eb); 
        color: white; 
        border: none; 
        transition: transform 0.2s; 
    }
    div.stButton > button:hover {
        transform: scale(1.02);
    }

    /* Container for the Auth Form to center it and give it a clean look */
    .auth-container {
        max-width: 400px;
        margin: 50px auto;
        padding: 30px;
        border-radius: 10px;
        background-color: #0f172a; /* Slightly lighter than app background */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    }
    
    /* Text input styling */
    .stTextInput label {
        color: #94a3b8;
    }

    /* Header styling */
    .stApp header h1 {
        text-align: center;
        color: #38bdf8;
    }
    </style>
    """, unsafe_allow_html=True)
# --- End of Custom CSS ---

# State for Authentication Mode
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "page_mode" not in st.session_state: st.session_state["page_mode"] = "Login"

if not st.session_state["logged_in"]:
    # Use a custom container for better visual grouping
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    st.title(f"🛡 {st.session_state['page_mode']} Access")
    
    if st.session_state["page_mode"] == "Login":
        with st.form("login_form"):
            u = st.text_input("Aadhaar Number", max_chars=12)
            p = st.text_input("DOB (DD-MM-YYYY)", type="password", help="Example: 01-01-1990")
            
            if st.form_submit_button("🔑 Login", use_container_width=True):
                if validate_login(u, p):
                    st.session_state["logged_in"] = True
                    st.rerun()
                else: 
                    st.error("Invalid Aadhaar or DOB. Please try again.")

        if st.button("Need to Register?", key="switch_to_register", use_container_width=True):
            st.session_state["page_mode"] = "Register"
            st.rerun()

    elif st.session_state["page_mode"] == "Register":
        with st.form("register_form"):
            new_u = st.text_input("Aadhaar Number (12 Digits)", max_chars=12)
            new_p = st.text_input("DOB (DD-MM-YYYY)", type="password", help="Use DD-MM-YYYY format")
            
            if st.form_submit_button("📝 Register", use_container_width=True):
                success, message = save_user(new_u, new_p)
                if success:
                    st.success(message)
                    # Automatically switch back to login after successful registration
                    st.session_state["page_mode"] = "Login"
                    # Rerun to clear the form and display the Login page
                    st.rerun() 
                else:
                    st.error(message)

        if st.button("Go to Login", key="switch_to_login", use_container_width=True):
            st.session_state["page_mode"] = "Login"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ==============================================================================
#       DASHBOARD LOGIC
# ==============================================================================

# --- Model Download and Loading (Unchanged) ---
def ensure_files_exist():
    """Downloads the OpenCV Age/Gender models if missing."""
    files = {
        "age_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
        "age_net.caffemodel": "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
        "gender_deploy.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
        "gender_net.caffemodel": "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"
    }
    
    if not os.path.exists("models"): os.makedirs("models")
    
    for fname, url in files.items():
        path = os.path.join("models", fname)
        if not os.path.exists(path):
            with st.spinner(f"Downloading AI Resource: {fname}..."):
                try:
                    r = requests.get(url)
                    with open(path, "wb") as f: f.write(r.content)
                except Exception as e: st.error(f"Failed to download {fname}: {e}")

# Run the check immediately
ensure_files_exist()

# --- LOAD MODELS ---
@st.cache_resource
def load_yolo(): return YOLO("yolov8n.pt")

@st.cache_resource
def load_ocr(): return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def load_age_gender_models():
    # Load standard OpenCV models
    age_net = cv2.dnn.readNet(os.path.join("models", "age_net.caffemodel"), 
                              os.path.join("models", "age_deploy.prototxt"))
    gender_net = cv2.dnn.readNet(os.path.join("models", "gender_net.caffemodel"), 
                                 os.path.join("models", "gender_deploy.prototxt"))
    return age_net, gender_net

YOLO_MODEL = load_yolo()
OCR_READER = load_ocr()
AGE_NET, GENDER_NET = load_age_gender_models()

# --- ANALYSIS FUNCTIONS (Unchanged) ---

def analyze_demographics_opencv(pil_image):
    """
    Detects faces and predicts Age/Gender using OpenCV DNN (No TensorFlow).
    """
    img = np.array(pil_image)
    # OpenCV models expect BGR
    if img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Detect Faces using Haar Cascades (Built-in, fast)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']
    
    results = []
    
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Predict Gender
        GENDER_NET.setInput(blob)
        gender_preds = GENDER_NET.forward()
        gender = gender_list[gender_preds[0].argmax()]
        gender_conf = gender_preds[0].max()
        
        # Predict Age
        AGE_NET.setInput(blob)
        age_preds = AGE_NET.forward()
        age = age_list[age_preds[0].argmax()]
        
        results.append({
            "Gender": gender,
            "Confidence": f"{int(gender_conf * 100)}%",
            "Age Group": age
        })
        
    return pd.DataFrame(results)

def run_scan(pil_image):
    # 1. Object Detection (YOLO)
    img_arr = np.array(pil_image)
    y_res = YOLO_MODEL.predict(img_arr, conf=0.01, verbose=False)[0]
    annotated = Image.fromarray(cv2.cvtColor(y_res.plot(), cv2.COLOR_BGR2RGB))
    
    # 2. Text Extraction (OCR)
    ocr_text = " ".join(OCR_READER.readtext(img_arr, detail=0))
    
    # 3. Demographics (OpenCV)
    demographics = analyze_demographics_opencv(pil_image)
    
    # 4. Parse Objects
    objects = []
    for box in y_res.boxes:
        cls_name = YOLO_MODEL.names[int(box.cls)]
        objects.append({"Type": cls_name, "Risk": "High" if cls_name in ['knife','scissors', 'gun', 'pistol', 'handgun', 'rifle', 'shotgun', 
        'firearm', 'sword', 'bomb', 'grenade', 'explosive', 'fire', 'smoke','boat'] else "Neutral"})
        
    return annotated, ocr_text, demographics, pd.DataFrame(objects)

# --- UI (Unchanged) ---
st.markdown("##  Threat_Detection AI")
st.divider()

c1, c2 = st.columns([5, 7])

with c1:
    src = st.radio("Source", ["Camera", "Upload"], horizontal=True)
    img_buffer = st.camera_input("Cam") if src == "Camera" else st.file_uploader("File")
    
    btn = st.button("INITIATE SCAN", use_container_width=True)
    if img_buffer and src == "Upload": st.image(img_buffer, caption="Preview")

with c2:
    if btn and img_buffer:
        pil_img = Image.open(img_buffer).convert("RGB")
        
        with st.spinner("Processing..."):
            viz_img, text, demo_df, obj_df = run_scan(pil_img)
            
            # Metrics
            r_lvl = "HIGH" if not obj_df.empty and "High" in obj_df["Risk"].values else "LOW"
            m1, m2 = st.columns(2)
            m1.metric("Threat Level", r_lvl)
            m2.metric("Humans Detected", len(demo_df))
            
            # Results
            st.subheader("👤 Demographics")
            if not demo_df.empty: st.dataframe(demo_df, use_container_width=True)
            else: st.info("No faces detected.")
            
            st.subheader("📍 Location (OCR)")
            if text:
                st.success(f"Found: {text}")
                # Note: The maps link uses the Google Maps base URL format
                st.link_button("Open Maps", f"https://www.google.com/maps/search/{text.replace(' ', '+')}")
            else: st.warning("No text found.")
            
            st.image(viz_img, caption="YOLO Scan")
