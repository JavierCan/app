import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import cv2
import mediapipe as mp
# --- PARCHE DE COMPATIBILIDAD ---
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
# --------------------------------
import math
import av
import threading
import time
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from datetime import datetime

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Love Intelligence Pro", layout="wide", page_icon="🫰")

st.markdown("""
    <style>
    .main { background: #050505; color: #ffffff; }
    .stMetric { 
        background: linear-gradient(135deg, rgba(255,0,150,0.1), rgba(0,255,255,0.1)); 
        border: 2px solid #ff0096; border-radius: 15px; padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

URL = "https://byzsjfizbtmzvstkvcuu.supabase.co"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ5enNqZml6YnRtenZzdGt2Y3V1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAxNjc4NzMsImV4cCI6MjA4NTc0Mzg3M30.Pv2CpCJsCfKJlJydHuzF7H0WFg3u5f_xRHkB21_YGEo"

@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)

supabase = init_supabase()

# --- PROCESADOR DE VIDEO OPTIMIZADO PARA MÓVIL ---
class VideoProcessor:
    def __init__(self):
        # Bajamos la complejidad para que el móvil no se trabe
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5, # Menos exigente para ganar velocidad
            min_tracking_confidence=0.5
        )
        self.ultimo_envio = 0
        self.alpha = 0.3 
        self.dist_k_filtrada = 1.0 

    def enviar_datos(self, tipo, frame_img):
        try:
            nombre = f"love_{int(time.time())}.jpg"
            _, buffer = cv2.imencode('.jpg', frame_img)
            supabase.storage.from_("capturas").upload(path=nombre, file=buffer.tobytes(), file_options={"content-type": "image/jpeg"})
            url = supabase.storage.from_("capturas").get_public_url(nombre)
            
            res = supabase.table("registros").select("conteo_acumulado").order("id", desc=True).limit(1).execute()
            count = (res.data[0]['conteo_acumulado'] if res.data else 0) + 1
            
            supabase.table("registros").insert({
                "tipo_gesto": tipo, "foto_url": url, "conteo_acumulado": count, 
                "created_at": pd.Timestamp.now(tz='UTC').isoformat()
            }).execute()
        except: pass

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        # Reducimos resolución de procesamiento para móviles
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        gesto_detectado = None
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                p = hand_lms.landmark
                dist_raw = math.hypot(p[4].x - p[8].x, p[4].y - p[8].y)
                self.dist_k_filtrada = (self.alpha * dist_raw) + ((1 - self.alpha) * self.dist_k_filtrada)
                if 0.005 < self.dist_k_filtrada < 0.04:
                    gesto_detectado = "korean_heart"

        if gesto_detectado:
            cv2.putText(img, "DETECTION ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            if time.time() - self.ultimo_envio > 6:
                self.ultimo_envio = time.time()
                threading.Thread(target=self.enviar_datos, args=(gesto_detectado, img.copy())).start()
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- DASHBOARD ---
st.title("🌌 Command Center v8.0")
st.sidebar.header("Cámara")
cam_mode = st.sidebar.radio("Cámara:", ("Frontal", "Trasera"))
facing_mode = "user" if cam_mode == "Frontal" else "environment"

kpi_area = st.empty()
c1, c2 = st.columns([1.5, 1])

with c1:
    webrtc_streamer(
        key="v8-deploy", 
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"facingMode": facing_mode}, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_html_attrs={"playsInline": True, "autoPlay": True, "muted": True}
    )

with c2:
    gallery_area = st.empty()

# --- DATA LOOP ---
while True:
    try:
        res = supabase.table("registros").select("*").order("id", desc=True).limit(6).execute()
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
            with kpi_area.container():
                st.metric("🔥 TOTAL GESTOS", df['conteo_acumulado'].max())
            with gallery_area.container():
                for i, row in df.iterrows():
                    st.image(row['foto_url'], caption=row['created_at'].strftime('%H:%M:%S'))
    except: pass
    time.sleep(3)
