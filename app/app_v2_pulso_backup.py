import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from PIL import Image

# Configuración de página con viewport completo
st.set_page_config(
    page_title="Pulso · Tamizaje Metabólico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS: Minimalista, Modern HealthTech (One Medical / Apple Health / Linear style)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Ocultar barra superior por defecto de Streamlit */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }

    /* Reset global */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #FAFAFC !important;
        color: #0F172A !important;
    }

    .main .block-container {
        max-width: 860px !important;
        padding-top: 0 !important;
        padding-bottom: 4rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* Header de borde a borde (Full-bleed Edge-to-Edge Navbar) */
    .app-navbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        width: 100vw;
        height: 58px;
        background: #FFFFFF;
        border-bottom: 1px solid #EDEDF2;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 32px;
        z-index: 999999;
        box-sizing: border-box;
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-mark {
        width: 22px;
        height: 22px;
        background: #0F172A;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .brand-text {
        font-size: 0.92rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #0F172A;
    }
    .brand-text span {
        font-weight: 400;
        color: #64748B;
        margin-left: 6px;
        font-size: 0.85rem;
    }
    .nav-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        color: #64748B;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
    }
    .nav-tag {
        background: #F1F5F9;
        color: #334155;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #E2E8F0;
    }

    /* Espaciador para la barra fija */
    .navbar-spacer {
        height: 78px;
    }

    /* Título principal minimalista */
    .hero-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #0F172A;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #64748B;
        margin: 0 0 28px 0;
        line-height: 1.5;
        font-weight: 400;
    }

    /* Tarjetas minimalistas */
    .surface-card {
        background: #FFFFFF;
        border: 1px solid #EBECEF;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .card-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .card-heading {
        font-size: 1.02rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: #0F172A;
        margin-bottom: 14px;
    }

    /* Chips de datos reconocidos */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }
    .metric-chip {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.8rem;
        color: #334155;
    }
    .metric-chip strong {
        color: #0F172A;
        font-weight: 600;
    }

    /* Botón minimalista moderno */
    .stButton > button {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 0.90rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
        border: 1px solid #0F172A !important;
        width: 100% !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: #1E293B !important;
        border-color: #1E293B !important;
    }

    /* Panel de resultados estilo Apple Health / One Medical */
    .result-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 28px;
        margin-top: 24px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
    }
    .result-score-block {
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin-bottom: 4px;
    }
    .result-percentage {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: #0F172A;
        line-height: 1;
    }
    .result-percentage-sub {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
    }
    
    /* Barra sutil de progreso */
    .result-bar-bg {
        background: #F1F5F9;
        border-radius: 999px;
        height: 6px;
        width: 100%;
        overflow: hidden;
        margin: 12px 0 20px 0;
    }
    .result-bar-amber {
        background: #D97706;
        height: 100%;
        border-radius: 999px;
    }
    .result-bar-teal {
        background: #0D9488;
        height: 100%;
        border-radius: 999px;
    }

    /* Badges de estado sobrios (Teoría del color médica: cero rojo/verde estridentes) */
    .badge-amber {
        display: inline-block;
        background: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        margin-bottom: 12px;
    }
    .badge-teal {
        display: inline-block;
        background: #F0FDFA;
        color: #0F766E;
        border: 1px solid #99F6E4;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        margin-bottom: 12px;
    }

    /* Pautas clínicas limpias */
    .guideline-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 0.84rem;
        line-height: 1.6;
        color: #334155;
    }
    .guideline-box strong {
        color: #0F172A;
    }

    /* Estilo para los inputs de Streamlit */
    div[data-baseweb="input"] {
        border-radius: 6px !important;
        background-color: #FFFFFF !important;
        border-color: #E2E8F0 !important;
    }
    div[data-baseweb="select"] {
        border-radius: 6px !important;
    }

    /* Pie de página mínimo */
    .app-footer {
        text-align: center;
        font-size: 0.74rem;
        color: #94A3B8;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #EDEDF2;
    }
</style>

<!-- Full-bleed Edge-to-Edge Navigation Bar -->
<div class="app-navbar">
    <div class="brand-group">
        <div class="brand-mark">P</div>
        <div class="brand-text">Pulso<span>Salud Digital</span></div>
    </div>
    <div class="nav-status">
        <span class="status-dot"></span>
        <span>Protocolo MinSalud Colombia</span>
        <span class="nav-tag">Naïve Bayes v2.4</span>
    </div>
</div>
<div class="navbar-spacer"></div>
""", unsafe_allow_html=True)

# Carga de modelo
@st.cache_resource
def load_trained_model():
    ruta = os.path.join(os.path.dirname(__file__), '..', 'models', 'modelo.pkl')
    if not os.path.exists(ruta):
        ruta = os.path.join(os.path.dirname(__file__), 'models', 'modelo.pkl')
    return joblib.load(ruta)

model = load_trained_model()

# Motor OCR ligero y de alta precisión
def extract_data_from_image(img_pil):
    extracted = {}
    ocr_text = ""
    lines = []
    
    try:
        import winocr
        res = winocr.recognize_pil_sync(img_pil, 'es')
        
        words = []
        for l in res['lines']:
            for w in l['words']:
                rect = w['bounding_rect']
                words.append({'text': w['text'], 'x': rect['x'], 'y': rect['y']})
                
        rows = []
        for w in sorted(words, key=lambda item: (item['y'], item['x'])):
            placed = False
            for r in rows:
                if abs(r['avg_y'] - w['y']) <= 14:
                    r['words'].append(w)
                    r['avg_y'] = sum(x['y'] for x in r['words']) / len(r['words'])
                    placed = True
                    break
            if not placed:
                rows.append({'avg_y': w['y'], 'words': [w]})
                
        sorted_rows = sorted(rows, key=lambda r: r['avg_y'])
        lines = [' '.join(w['text'] for w in sorted(r['words'], key=lambda item: item['x'])) for r in sorted_rows]
        ocr_text = '\n'.join(lines)
    except Exception:
        ocr_text = ""
        lines = []

    if not lines and ocr_text:
        lines = ocr_text.split('\n')

    # Identificación de magnitudes de laboratorios colombianos
    for line in lines:
        l_low = line.lower()
        
        # Glucemia
        if any(k in l_low for k in ['gluc', 'glic']):
            nums = re.findall(r'(\d+[\.,]?\d*)', l_low)
            for n in nums:
                val = float(n.replace(',', '.'))
                if 45 <= val <= 400:
                    extracted['Glucose'] = val
                    break
                elif 400 < val <= 4000:
                    extracted['Glucose'] = round(val / 10.0, 1)
                    break
                    
        # Insulina
        if 'insulin' in l_low:
            nums = re.findall(r'(\d+[\.,]?\d*)', l_low)
            for n in nums:
                val = float(n.replace(',', '.'))
                if 2 <= val <= 250:
                    extracted['Insulin'] = val
                    break
                elif 250 < val <= 2500:
                    extracted['Insulin'] = round(val / 10.0, 1)
                    break
                    
        # Presión diastólica
        if any(k in l_low for k in ['tensi', 'presi', 't.a']):
            ta = re.search(r'(\d{2,3})\s*[\/\-y]\s*(\d{2,3})', l_low)
            if ta:
                extracted['BloodPressure'] = float(ta.group(2))
            else:
                nums = re.findall(r'(\d{2,3})', l_low)
                for n in nums:
                    val = float(n)
                    if 45 <= val <= 130:
                        extracted['BloodPressure'] = val
                        break
                        
        # IMC
        if any(k in l_low for k in ['imc', 'masa corporal']):
            nums = re.findall(r'(\d+[\.,]?\d*)', l_low)
            for n in nums:
                val = float(n.replace(',', '.'))
                if 14 <= val <= 65:
                    extracted['BMI'] = val
                    break
                elif 140 < val <= 650:
                    extracted['BMI'] = round(val / 10.0, 1)
                    break
                    
        # Gestaciones
        if any(k in l_low for k in ['gestaci', 'embaraz', 'parto']):
            emb_m = re.search(r'(?:gestaci\w*|embaraz\w*|parto\w*|g\s*[:=]?\s*)(\d+)', l_low)
            if emb_m:
                extracted['Pregnancies'] = int(emb_m.group(1))

        # Antecedente familiar
        if any(k in l_low for k in ['madre', 'mama', 'padre', 'papa', 'herman']) and any(d in l_low for d in ['diabet', 'dm2']):
            extracted['DiabetesPedigreeFunction'] = 0.65
        elif any(k in l_low for k in ['abuel', 'tio', 'tia']) and any(d in l_low for d in ['diabet', 'dm2']):
            extracted['DiabetesPedigreeFunction'] = 0.38

    if 'Age' not in extracted and ocr_text:
        age_m = re.search(r'(?:edad|a[ñn]os)[^\d\n]*?(\d{1,3})', ocr_text.lower())
        if age_m:
            val = int(age_m.group(1))
            if 1 <= val <= 110:
                extracted['Age'] = val

    return extracted, ocr_text

# Estado en sesión
if 'glucose' not in st.session_state: st.session_state.glucose = 100.0
if 'insulin' not in st.session_state: st.session_state.insulin = 65.0
if 'blood_pressure' not in st.session_state: st.session_state.blood_pressure = 76.0
if 'skin_thickness' not in st.session_state: st.session_state.skin_thickness = 22.0
if 'bmi' not in st.session_state: st.session_state.bmi = 25.5
if 'age' not in st.session_state: st.session_state.age = 35
if 'pregnancies' not in st.session_state: st.session_state.pregnancies = 1
if 'detected_summary' not in st.session_state: st.session_state.detected_summary = None

# Hero Title
st.markdown("""
<div style="margin-top: 10px;">
    <h1 class="hero-title">Evaluación de Riesgo Metabólico</h1>
    <p class="hero-subtitle">
        Tamizaje temprano de diabetes con modelo Naïve Bayes poblacional y extracción óptica de exámenes de laboratorio.
    </p>
</div>
""", unsafe_allow_html=True)

# SECCIÓN 1: INGESTIÓN DEL EXAMEN (CÁMARA / ARCHIVO)
st.markdown("""
<div class="surface-card">
    <div class="card-label">01 / Ingestión Documental</div>
    <div class="card-heading">Captura de Examen de Laboratorio</div>
""", unsafe_allow_html=True)

metodo = st.segmented_control(
    "Método de captura",
    options=["Cámara del dispositivo", "Cargar archivo de imagen", "Examen de prueba (Bogotá)"],
    default="Cámara del dispositivo",
    label_visibility="collapsed"
)

foto_archivo = None

if metodo == "Cámara del dispositivo":
    cam_in = st.camera_input("Enfoque el reporte impreso con buena iluminación", label_visibility="collapsed")
    if cam_in is not None:
        foto_archivo = cam_in

elif metodo == "Cargar archivo de imagen":
    up_in = st.file_uploader("Seleccione una fotografía o escaneo del examen (JPG, PNG)", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    if up_in is not None:
        foto_archivo = up_in

elif metodo == "Examen de prueba (Bogotá)":
    demo_img_path = os.path.join(os.path.dirname(__file__), 'ejemplo_examen_colombia.png')
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.write("")
        if st.button("Cargar datos de examen de prueba"):
            if os.path.exists(demo_img_path):
                img_demo = Image.open(demo_img_path)
                data_d, _ = extract_data_from_image(img_demo)
                if 'Glucose' in data_d: st.session_state.glucose = float(data_d['Glucose'])
                if 'Insulin' in data_d: st.session_state.insulin = float(data_d['Insulin'])
                if 'BloodPressure' in data_d: st.session_state.blood_pressure = float(data_d['BloodPressure'])
                if 'BMI' in data_d: st.session_state.bmi = float(data_d['BMI'])
                if 'Age' in data_d: st.session_state.age = int(data_d['Age'])
                if 'Pregnancies' in data_d: st.session_state.pregnancies = int(data_d['Pregnancies'])
                st.session_state.detected_summary = data_d
                st.rerun()
    with col_d2:
        if os.path.exists(demo_img_path):
            st.image(demo_img_path, caption="Examen de referencia: Laboratorio Metropolitano", width=220)

# Procesar imagen cargada
if foto_archivo is not None:
    try:
        pil_im = Image.open(foto_archivo)
        with st.spinner("Procesando imagen..."):
            extracted, _ = extract_data_from_image(pil_im)
            if extracted:
                if 'Glucose' in extracted: st.session_state.glucose = float(extracted['Glucose'])
                if 'Insulin' in extracted: st.session_state.insulin = float(extracted['Insulin'])
                if 'BloodPressure' in extracted: st.session_state.blood_pressure = float(extracted['BloodPressure'])
                if 'BMI' in extracted: st.session_state.bmi = float(extracted['BMI'])
                if 'Age' in extracted: st.session_state.age = int(extracted['Age'])
                if 'Pregnancies' in extracted: st.session_state.pregnancies = int(extracted['Pregnancies'])
                st.session_state.detected_summary = extracted
    except Exception as e:
        st.caption(f"Aviso de captura: {e}")

# Chips de confirmación sutil
if st.session_state.detected_summary:
    chips_html = "".join([f'<div class="metric-chip"><span>{k}:</span> <strong>{v}</strong></div>' for k, v in st.session_state.detected_summary.items()])
    st.markdown(f"""
    <div style="margin-top: 14px;">
        <div style="font-size:0.75rem; color:#64748B; margin-bottom:4px;">Valores detectados en la imagen:</div>
        <div class="chip-container">{chips_html}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# SECCIÓN 2: FORMULARIO DE PARÁMETROS CLÍNICOS
st.markdown("""
<div class="surface-card">
    <div class="card-label">02 / Parámetros Clínicos</div>
    <div class="card-heading">Registro y Verificación</div>
""", unsafe_allow_html=True)

col_f1, col_f2 = st.columns(2)

with col_f1:
    glucosa = st.number_input(
        "Glucemia en ayunas (mg/dL)",
        min_value=40.0, max_value=350.0,
        value=float(st.session_state.glucose),
        step=1.0,
        help="Normal: 70 - 99 mg/dL. Prediabetes: 100 - 125 mg/dL."
    )
    insulina = st.number_input(
        "Insulina basal (µUI/mL)",
        min_value=1.0, max_value=400.0,
        value=float(st.session_state.insulin),
        step=1.0,
        help="Referencia en suero: 2.6 - 24.9 µUI/mL."
    )
    presion = st.number_input(
        "Presión diastólica (mm Hg)",
        min_value=40.0, max_value=140.0,
        value=float(st.session_state.blood_pressure),
        step=1.0,
        help="Referencia habitual: menor a 80 mm Hg."
    )
    edad = st.number_input(
        "Edad (años)",
        min_value=18, max_value=100,
        value=int(st.session_state.age),
        step=1
    )

with col_f2:
    imc = st.number_input(
        "Índice de Masa Corporal (kg/m²)",
        min_value=12.0, max_value=60.0,
        value=float(st.session_state.bmi),
        step=0.1,
        help="Normopeso: 18.5 - 24.9 | Sobrepeso: 25.0 - 29.9"
    )
    embarazos = st.number_input(
        "Número de embarazos",
        min_value=0, max_value=20,
        value=int(st.session_state.pregnancies),
        step=1,
        help="0 en hombres o sin antecedentes gestacionales."
    )
    
    opciones_pedigree = {
        "Sin antecedentes familiares directos": 0.18,
        "Familiar de segundo grado (abuelos, tíos)": 0.38,
        "Familiar de primer grado (padre, madre o hermano/a)": 0.65,
        "Ambos padres con antecedentes de diabetes": 1.15
    }
    pedigree_label = st.selectbox(
        "Antecedentes familiares de diabetes",
        options=list(opciones_pedigree.keys()),
        index=1
    )
    pedigree_val = opciones_pedigree[pedigree_label]
    pliegue_cutaneo = float(st.session_state.skin_thickness)

st.markdown("</div>", unsafe_allow_html=True)

# BOTÓN DE EVALUACIÓN
calcular = st.button("Evaluar Riesgo Metabólico", use_container_width=True)

# SECCIÓN 3: RESULTADOS (DISEÑO HEALTH-APP MODERNO)
if calcular:
    df_eval = pd.DataFrame({
        'Pregnancies': [embarazos],
        'Glucose': [glucosa],
        'BloodPressure': [presion],
        'SkinThickness': [pliegue_cutaneo],
        'Insulin': [insulina],
        'BMI': [imc],
        'DiabetesPedigreeFunction': [pedigree_val],
        'Age': [edad]
    })
    
    prediccion = model.predict(df_eval)[0]
    probabilidades = model.predict_proba(df_eval)[0]
    prob_atencion = float(probabilidades[1] * 100.0)
    prob_estabilidad = float(probabilidades[0] * 100.0)

    st.markdown("""<div class="result-container">""", unsafe_allow_html=True)

    if prediccion == 1 or prob_atencion >= 50.0:
        # ATENCIÓN PREVENTIVA (Ámbar suave / Ocre - Cero rojo)
        st.markdown(f"""
        <div class="badge-amber">Atención Preventiva Sugerida</div>
        <div class="result-score-block">
            <span class="result-percentage">{prob_atencion:.0f}%</span>
            <span class="result-percentage-sub">probabilidad posterior estimada por Naïve Bayes</span>
        </div>
        <div class="result-bar-bg">
            <div class="result-bar-amber" style="width: {min(prob_atencion, 100.0):.1f}%;"></div>
        </div>
        <div class="guideline-box">
            <strong>Orientación de cuidado clínico (MinSalud Colombia):</strong><br>
            • <strong>Consulta de control:</strong> Solicitar cita de valoración preventiva en medicina general a través de su EPS / IPS.<br>
            • <strong>Prueba complementaria:</strong> Se sugiere correlacionar con prueba de <em>Hemoglobina Glicosilada (HbA1c)</em> o curva de tolerancia a la glucosa.<br>
            • <strong>Intervención oportuna:</strong> Los desbalances metabólicos tempranos son altamente reversibles mediante pautas de nutrición balanceada y ejercicio regular.
        </div>
        """, unsafe_allow_html=True)
    else:
        # ESTABILIDAD (Teal suave - Cero verde chillón)
        st.markdown(f"""
        <div class="badge-teal">Rango Fisiológico Favorable</div>
        <div class="result-score-block">
            <span class="result-percentage">{prob_estabilidad:.0f}%</span>
            <span class="result-percentage-sub">índice de estabilidad metabólica</span>
        </div>
        <div class="result-bar-bg">
            <div class="result-bar-teal" style="width: {min(prob_estabilidad, 100.0):.1f}%;"></div>
        </div>
        <div class="guideline-box">
            <strong>Pautas de bienestar preventivo:</strong><br>
            • <strong>Mantenimiento:</strong> Preservar hábitos alimenticios conscientes con consumo moderado de carbohidratos refinados.<br>
            • <strong>Actividad física:</strong> Cumplir con al menos 150 minutos semanales de actividad física moderada recomendada por la OMS y MinSalud.<br>
            • <strong>Seguimiento:</strong> Mantener controles médicos y tamizajes de laboratorio periódicos de rutina.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""</div>""", unsafe_allow_html=True)

# Footer minimalista
st.markdown("""
<div class="app-footer">
    Pulso Salud Digital · Algoritmo de clasificación supervisada Naïve Bayes · Cumple con directrices del Ministerio de Salud y Protección Social de Colombia
</div>
""", unsafe_allow_html=True)
