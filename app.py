import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --- SAYT SOZLAMALARI ---
st.set_page_config(
    page_title="InsectID - Hashorotlarni aniqlash", 
    page_icon="🐞", 
    layout="wide"
)

# --- ZAMONAVIY DIZAYN (CSS) ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background: linear-gradient(to right, #ffffff, #e8f5e9);
    }
    h1 {
        color: #1b5e20;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2e7d32;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODELNI YUKLASH FUNKSIYASI ---
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_model()

# --- ASOSIY QISM ---
st.markdown("<h1 style='text-align: center;'>🐞 Hashorotlar Turini Aniqlash Tizimi</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Sun'iy intellekt yordamida hashorotlarni lahzalarda aniqlang</p>", unsafe_allow_html=True)
st.divider()

# Sidebar (Yon menyu)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/528/528398.png", width=100)
    st.title("Loyiha haqida")
    st.info("""
    **Maqsad:** 5 xil turdagi hashorotlarni aniqlash.
    **Texnologiya:** YOLOv8 (Ultralytics).
    **Muallif:** Individual loyiha ishchisi.
    """)
    
    if model is None:
        st.warning("⚠️ Model (best.pt) hali yuklanmagan. Hozircha faqat dizayn ko'rinishida.")
    else:
        st.success("✅ Model muvaffaqiyatli ulangan!")

# Markaziy qism (Rasm yuklash)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Rasmni yuklang")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Tanlangan rasm", use_container_width=True)

with col2:
    st.subheader("🔍 Tahlil natijasi")
    
    if uploaded_file:
        if st.button("Hashorotni aniqlash"):
            if model is not None:
                with st.spinner('AI tahlil qilmoqda...'):
                    results = model.predict(img)
                    res_img = results[0].plot()
                    
                    # Natijani ko'rsatish
                    st.image(res_img, caption="Aniqlangan ob'ektlar", use_container_width=True)
                    
                    if len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            label = results[0].names[int(box.cls[0])]
                            conf = float(box.conf[0])
                            st.metric(label=f"Aniqlangan hashorot: {label}", value=f"{conf*100:.1f}%")
                    else:
                        st.error("Kechirasiz, rasmda hashorot topilmadi.")
            else:
                st.error("Xatolik: 'best.pt' fayli topilmadi! Iltimos, model o'qib bo'lgach faylni papkaga joylang.")
    else:
        st.info("Natijani ko'rish uchun chap tomondan rasm yuklang.")

# Sayt pastki qismi
st.markdown("---")
st.caption("Individual loyiha ishi © 2026. Barcha huquqlar himoyalangan.")