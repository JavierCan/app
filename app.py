import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import cv2
import mediapipe as mp
import math
import av
import threading
import time
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from datetime import datetime

# --- 1. CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Love Intelligence Pro", layout="wide", page_icon="🫰")

st.markdown("""
    <style>
    .main { background: #050505; color: #ffffff; }
    .stMetric { 
        background: linear-gradient(135deg, rgba(255,0,150,0.1), rgba(0,255,255,0.1)); 
        border: 2px solid #ff0096; border-radius: 20px; padding: 20px; 
        box-shadow: 0 0 15px rgba(255,0,150,0.3);
    }
    .combo-box {
        background: #ff0096; color: white; padding: 10px; 
        border-radius: 10px; text-align: center; font-weight: bold;
        animation: pulse 1s infinite;
    }
    @keyframes pulse { 0% {transform: scale(1);} 50% {transform: scale(1.05);} 100% {transform: scale(1);} }
    </style>
    """, unsafe_allow_html=True)

# Credenciales de Supabase
URL = "https://byzsjfizbtmzvstkvcuu.supabase.co"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ5enNqZml6YnRtenZzdGt2Y3V1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAxNjc4NzMsImV4cCI6MjA4NTc0Mzg3M30.Pv2CpCJsCfKJlJydHuzF7H0WFg3u5f_xRHkB21_YGEo"

@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)

supabase = init_supabase()

# --- 2. PROCESADOR DE VIDEO CON FILTRO DE ESTABILIDAD ---
class VideoProcessor:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.ultimo_envio = 0
        self.alpha = 0.2  # Filtro Kalman-light
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
        except Exception as e: print(f"Error subida: {e}")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        clean_img = img.copy()
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        gesto_detectado = None
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)
                p = hand_lms.landmark
                
                # --- Filtro de Distancia ---
                dist_raw = math.hypot(p[4].x - p[8].x, p[4].y - p[8].y)
                self.dist_k_filtrada = (self.alpha * dist_raw) + ((1 - self.alpha) * self.dist_k_filtrada)
                
                if 0.005 < self.dist_k_filtrada < 0.035:
                    gesto_detectado = "korean_heart"

            if len(results.multi_hand_landmarks) == 2:
                h1, h2 = results.multi_hand_landmarks[0].landmark, results.multi_hand_landmarks[1].landmark
                dn = (math.hypot(h1[8].x-h2[8].x, h1[8].y-h2[8].y) + math.hypot(h1[4].x-h2[4].x, h1[4].y-h2[4].y))/2
                if dn < 0.06: gesto_detectado = "normal_heart"

        if gesto_detectado:
            color = (255, 0, 255) if gesto_detectado == "korean_heart" else (0, 255, 0)
            cv2.rectangle(img, (0, 0), (640, 60), (0,0,0), -1)
            cv2.putText(img, f"🫰 {gesto_detectado.upper()} ACTIVO", (120, 40), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            
            ahora = time.time()
            if ahora - self.ultimo_envio > 6:
                self.ultimo_envio = ahora
                threading.Thread(target=self.enviar_datos, args=(gesto_detectado, clean_img)).start()
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. DASHBOARD UI ---
st.title("🌌 Love Intelligence Command Center")

# Selector de cámara en la barra lateral
st.sidebar.header("📸 Opciones de Cámara")
cam_mode = st.sidebar.radio("Modo:", ("Frontal", "Trasera"))
facing_mode = "user" if cam_mode == "Frontal" else "environment"

kpi_area = st.empty()
st.divider()

c1, c2 = st.columns([1.3, 1.7])

with c1:
    webrtc_streamer(
        key="love-v8", 
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"facingMode": facing_mode}, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_html_attrs={"playsInline": True, "autoPlay": True, "muted": True, "style": {"width": "100%"}}
    )
    chart_area = st.empty()

with c2:
    combo_placeholder = st.empty()
    gallery_area = st.empty()

# --- 4. BUCLE DE DATOS EN TIEMPO REAL ---
while True:
    try:
        res = supabase.table("registros").select("*").order("id", desc=True).limit(20).execute()
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame()

        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
            
            with kpi_area.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🔥 TOTAL", df['conteo_acumulado'].max())
                m2.metric("🫰 K-STYLE", len(df[df['tipo_gesto'] == 'korean_heart']))
                m3.metric("❤️ NORMAL", len(df[df['tipo_gesto'] == 'normal_heart']))
                m4.metric("⏱️ STATUS", "ONLINE", delta="LIVE")

            ahora_utc = pd.Timestamp.now(tz='UTC')
            recientes = df[df['created_at'] > (ahora_utc - pd.Timedelta(seconds=60))]
            if len(recientes) >= 3:
                combo_placeholder.markdown(f"<div class='combo-box'>🔥 COMBO X{len(recientes)}: ¡ESTÁS ON FIRE! ❤️‍F</div>", unsafe_allow_html=True)
            else: combo_placeholder.empty()

            with chart_area.container():
                df_l = df.groupby(df['created_at'].dt.floor('s')).size().reset_index(name='n')
                fig = px.line(df_l, x='created_at', y='n', template="plotly_dark")
                fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, width='stretch', key=f"c_{int(time.time())}")

            with gallery_area.container():
                cols = st.columns(3)
                for i, row in df.head(6).iterrows():
                    with cols[i % 3]:
                        st.image(row['foto_url'], width='stretch')
                        st.caption(f"{row['tipo_gesto']} | {row['created_at'].strftime('%H:%M:%S')}")
    except: pass
    time.sleep(2)
