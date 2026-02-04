import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import math
import av
import threading
import time
import pandas as pd
from supabase import create_client, Client

# --- IMPORTACIÓN ESTÁNDAR ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Love Intelligence Pro", layout="wide", page_icon="🫰")

URL = "https://byzsjfizbtmzvstkvcuu.supabase.co"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ5enNqZml6YnRtenZzdGt2Y3V1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAxNjc4NzMsImV4cCI6MjA4NTc0Mzg3M30.Pv2CpCJsCfKJlJydHuzF7H0WFg3u5f_xRHkB21_YGEo"

@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)

supabase = init_supabase()

class VideoProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        gesto_detectado = None
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                p = hand_lms.landmark
                # Lógica K-Heart
                dist_raw = math.hypot(p[4].x - p[8].x, p[4].y - p[8].y)
                self.dist_k_filtrada = (self.alpha * dist_raw) + ((1 - self.alpha) * self.dist_k_filtrada)
                
                if 0.005 < self.dist_k_filtrada < 0.035:
                    gesto_detectado = "korean_heart"

        if gesto_detectado:
            cv2.putText(img, "🫰 K-HEART DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            if time.time() - self.ultimo_envio > 6:
                self.ultimo_envio = time.time()
                threading.Thread(target=self.enviar_datos, args=(gesto_detectado, img.copy())).start()
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ---
st.title("🌌 Love Intelligence Deploy")
cam_mode = st.sidebar.radio("Cámara:", ("Frontal", "Trasera"))
facing_mode = "user" if cam_mode == "Frontal" else "environment"

webrtc_streamer(
    key="deploy-final", 
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": {"facingMode": facing_mode}, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_html_attrs={"playsInline": True, "autoPlay": True, "muted": True}
)

# Galería rápida
try:
    res = supabase.table("registros").select("*").order("id", desc=True).limit(4).execute()
    if res.data:
        st.subheader("Últimas capturas")
        cols = st.columns(4)
        for i, row in enumerate(res.data):
            cols[i].image(row['foto_url'])
except: pass
