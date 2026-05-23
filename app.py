"""
╔══════════════════════════════════════════════════════════════════╗
║            BMI ANALYZER PRO — Streamlit Application             ║
║      Universitas AMIKOM Yogyakarta — Data Mining Project         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import io
import base64
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Resolve paths relative to this file — works on Streamlit Cloud and local
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "bmi.csv"

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMI Analyzer Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
LABEL_MAP = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity",
}

LABEL_COLORS = {
    "Extremely Weak": "#e74c3c",
    "Weak": "#e67e22",
    "Normal": "#27ae60",
    "Overweight": "#f39c12",
    "Obesity": "#e74c3c",
    "Extreme Obesity": "#c0392b",
}

LABEL_GRADIENT = {
    "Extremely Weak": "linear-gradient(135deg, #e74c3c, #c0392b)",
    "Weak": "linear-gradient(135deg, #e67e22, #d35400)",
    "Normal": "linear-gradient(135deg, #27ae60, #1e8449)",
    "Overweight": "linear-gradient(135deg, #f39c12, #d68910)",
    "Obesity": "linear-gradient(135deg, #e74c3c, #a93226)",
    "Extreme Obesity": "linear-gradient(135deg, #922b21, #641e16)",
}

BMI_THRESHOLDS = {
    "Extremely Weak": (0, 16),
    "Weak": (16, 18.5),
    "Normal": (18.5, 25),
    "Overweight": (25, 30),
    "Obesity": (30, 35),
    "Extreme Obesity": (35, 60),
}

HEALTH_TIPS = {
    "Extremely Weak": {
        "icon": "🚨",
        "summary": "Berat badan sangat kurang. Perlu perhatian medis segera.",
        "diet": [
            "Tingkatkan asupan kalori 500–1000 kkal/hari di atas kebutuhan basal",
            "Konsumsi makanan padat gizi: kacang-kacangan, alpukat, telur, ikan",
            "Makan 5–6 kali sehari dengan porsi kecil tapi sering",
            "Tambahkan protein 1.5–2 g/kg BB per hari untuk pemulihan massa otot",
        ],
        "exercise": [
            "Hindari kardio intensitas tinggi — hemat energi",
            "Fokus pada latihan kekuatan ringan: squat, push-up, resistance band",
            "3x seminggu, 20–30 menit, intensitas rendah-sedang",
            "Yoga atau pilates untuk fleksibilitas dan keseimbangan hormon",
        ],
        "hydration": "Minum minimal 1.5–2 liter air per hari. Tambahkan elektrolit jika perlu.",
        "sleep": "Tidur 8–9 jam per malam. Kekurangan tidur mengganggu nafsu makan dan hormon pertumbuhan.",
        "lifestyle": [
            "Konsultasi dengan dokter gizi segera",
            "Monitor berat badan setiap minggu",
            "Kelola stres — stres kronis mengurangi nafsu makan",
        ],
    },
    "Weak": {
        "icon": "⚠️",
        "summary": "Berat badan di bawah ideal. Perlu peningkatan asupan kalori.",
        "diet": [
            "Tingkatkan 300–500 kkal/hari dari makanan bergizi",
            "Perbanyak karbohidrat kompleks: nasi merah, oat, ubi",
            "Konsumsi protein berkualitas: ayam, ikan, tahu, tempe, kacang",
            "Tambahkan healthy fat: minyak zaitun, kacang almond, alpukat",
        ],
        "exercise": [
            "Latihan kekuatan 3x seminggu untuk membangun massa otot",
            "Kardio ringan 2x seminggu, 20–30 menit",
            "Push-up, dumbbell, resistance training",
            "Hindari olahraga berlebihan yang membakar terlalu banyak kalori",
        ],
        "hydration": "2 liter air per hari. Minum susu atau protein shake setelah latihan.",
        "sleep": "7–8 jam tidur berkualitas untuk mendukung recovery otot.",
        "lifestyle": [
            "Track asupan kalori harian dengan aplikasi",
            "Konsultasi ahli gizi untuk meal plan personal",
            "Hindari melewatkan sarapan",
        ],
    },
    "Normal": {
        "icon": "✅",
        "summary": "BMI ideal! Pertahankan gaya hidup sehat Anda.",
        "diet": [
            "Pertahankan pola makan seimbang: 50% karbohidrat, 30% protein, 20% lemak",
            "Konsumsi 5 porsi buah dan sayur per hari",
            "Batasi gula tambahan dan makanan ultra-proses",
            "Meal prep mingguan untuk konsistensi pola makan",
        ],
        "exercise": [
            "Minimal 150 menit kardio moderat per minggu (WHO guideline)",
            "Strength training 2x seminggu",
            "Variasikan olahraga: lari, renang, bersepeda, HIIT",
            "Tetap aktif di luar jam gym: tangga vs lift, jalan kaki",
        ],
        "hydration": "2–2.5 liter air per hari. Tambah saat berolahraga.",
        "sleep": "7–8 jam per malam. Konsistensikan jadwal tidur dan bangun.",
        "lifestyle": [
            "Lakukan medical check-up tahunan",
            "Kelola stres dengan meditasi atau journaling",
            "Monitor BMI setiap 3 bulan",
        ],
    },
    "Overweight": {
        "icon": "⚡",
        "summary": "Berat badan sedikit berlebih. Perubahan gaya hidup dapat membawa ke kondisi ideal.",
        "diet": [
            "Defisit kalori 300–500 kkal/hari — bertahap dan berkelanjutan",
            "Prioritaskan protein tinggi untuk mempertahankan massa otot",
            "Kurangi karbohidrat olahan dan gula; pilih whole grain",
            "Intermittent fasting 16:8 bisa menjadi strategi efektif",
        ],
        "exercise": [
            "Kardio 30–45 menit, 4–5x seminggu",
            "HIIT 2–3x seminggu untuk pembakaran lemak optimal",
            "Strength training untuk meningkatkan metabolisme basal",
            "Aktif minimal 8.000–10.000 langkah per hari",
        ],
        "hydration": "Minum 2.5–3 liter per hari. Air sebelum makan membantu kontrol porsi.",
        "sleep": "7–8 jam. Kurang tidur meningkatkan ghrelin (hormon lapar).",
        "lifestyle": [
            "Hindari makan larut malam (setelah pukul 20.00)",
            "Masak sendiri — kontrol bahan dan kalori",
            "Join komunitas fitness untuk accountability",
        ],
    },
    "Obesity": {
        "icon": "🔴",
        "summary": "Obesitas memerlukan perubahan gaya hidup serius dan konsultasi medis.",
        "diet": [
            "Defisit kalori 500–750 kkal/hari dengan supervisi ahli gizi",
            "Prioritaskan protein, serat tinggi, kalori rendah",
            "Eliminasi minuman manis, junk food, dan makanan goreng",
            "Gunakan piring lebih kecil dan makan perlahan (mindful eating)",
        ],
        "exercise": [
            "Mulai dari low-impact: jalan kaki, renang, sepeda statis",
            "Tingkatkan durasi secara bertahap: 10 → 20 → 30 → 45 menit",
            "Hindari aktivitas high-impact yang membebani sendi",
            "Konsultasi fisioterapis sebelum memulai program latihan",
        ],
        "hydration": "3 liter air per hari. Ganti minuman manis dengan air putih atau infused water.",
        "sleep": "7–9 jam. Obesitas berkorelasi dengan sleep apnea — periksa ke dokter.",
        "lifestyle": [
            "Konsultasi dokter — cek gula darah, kolesterol, tekanan darah",
            "Pertimbangkan program penurunan berat badan berbasis medis",
            "Set target realistis: turun 0.5–1 kg per minggu",
        ],
    },
    "Extreme Obesity": {
        "icon": "🚑",
        "summary": "Obesitas ekstrem membutuhkan penanganan medis profesional segera.",
        "diet": [
            "WAJIB di bawah supervisi dokter gizi atau bariatric specialist",
            "Program diet medis terstruktur — jangan diet ekstrem sendiri",
            "Kurangi kalori secara bertahap dengan panduan profesional",
            "Monitor nutrisi mikro dengan ketat — risiko defisiensi tinggi",
        ],
        "exercise": [
            "Mulai dengan gerakan minimal: duduk-berdiri, jalan di tempat",
            "Hydrotherapy (renang) — paling aman untuk sendi",
            "Fisioterapi terpandu sangat dianjurkan",
            "Jangan memaksakan diri — risiko cedera sangat tinggi",
        ],
        "hydration": "3–4 liter air per hari di bawah pengawasan medis.",
        "sleep": "Wajib periksa sleep apnea. Kualitas tidur sangat mempengaruhi berat badan.",
        "lifestyle": [
            "Konsultasi segera dengan dokter spesialis obesitas",
            "Evaluasi opsi medis: program bariatric, terapi perilaku",
            "Dukungan psikologis sangat penting — obesitas sering berkaitan dengan kondisi mental",
        ],
    },
}

# ─────────────────────────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────────────────────────
def get_theme_css(dark_mode: bool) -> str:
    if dark_mode:
        bg = "#0f0f1a"
        bg2 = "#1a1a2e"
        card_bg = "#16213e"
        card_border = "#0f3460"
        text = "#e8e8f0"
        text_muted = "#8888aa"
        accent = "#7c3aed"
        accent2 = "#06b6d4"
        input_bg = "#1e1e3a"
    else:
        bg = "#f0f2f8"
        bg2 = "#ffffff"
        card_bg = "#ffffff"
        card_border = "#e2e8f0"
        text = "#1a1a2e"
        text_muted = "#64748b"
        accent = "#7c3aed"
        accent2 = "#06b6d4"
        input_bg = "#f8fafc"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {{
    font-family: 'Inter', sans-serif !important;
}}

.stApp {{
    background: {bg} !important;
}}

section[data-testid="stSidebar"] {{
    background: {bg2} !important;
    border-right: 1px solid {card_border} !important;
}}

.main-header {{
    background: linear-gradient(135deg, {accent} 0%, {accent2} 100%);
    padding: 2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(124,58,237,0.3);
    position: relative;
    overflow: hidden;
}}

.main-header::before {{
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}}

.main-header h1 {{
    color: #ffffff !important;
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
}}

.main-header p {{
    color: rgba(255,255,255,0.8) !important;
    font-size: 1rem !important;
    margin: 0.4rem 0 0 0 !important;
}}

.metric-card {{
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}}

.metric-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(124,58,237,0.15);
    border-color: {accent};
}}

.metric-card .value {{
    font-size: 2.2rem;
    font-weight: 800;
    color: {accent};
    line-height: 1;
}}

.metric-card .label {{
    font-size: 0.8rem;
    color: {text_muted};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.4rem;
}}

.bmi-result-card {{
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    margin: 1rem 0;
}}

.bmi-result-card .bmi-number {{
    font-size: 4rem;
    font-weight: 900;
    color: white;
    line-height: 1;
}}

.bmi-result-card .bmi-label {{
    font-size: 1.3rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    margin-top: 0.5rem;
}}

.section-card {{
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
}}

.section-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {text};
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.tip-item {{
    background: {input_bg};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    border-left: 3px solid {accent};
    color: {text};
    font-size: 0.88rem;
}}

.model-badge {{
    display: inline-block;
    background: linear-gradient(135deg, {accent}, {accent2});
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 0.2rem;
}}

.insight-box {{
    background: linear-gradient(135deg, {accent}15, {accent2}15);
    border: 1px solid {accent}40;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: {text};
    font-size: 0.9rem;
}}

.stTab {{
    background: {card_bg} !important;
}}

/* Sidebar styling */
.sidebar-input-label {{
    color: {text};
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {bg}; }}
::-webkit-scrollbar-thumb {{ background: {accent}; border-radius: 3px; }}

/* Tab active */
.stTabs [aria-selected="true"] {{
    color: {accent} !important;
    border-bottom-color: {accent} !important;
    font-weight: 700 !important;
}}

/* Hide streamlit branding */
#MainMenu, footer {{ visibility: hidden; }}

/* Button */
.stButton > button {{
    background: linear-gradient(135deg, {accent}, {accent2}) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}}

.progress-bar-custom {{
    height: 8px;
    border-radius: 4px;
    background: {card_border};
    overflow: hidden;
    margin: 4px 0;
}}

.progress-bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 1s ease;
}}
</style>
"""

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "training_results" not in st.session_state:
    st.session_state.training_results = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None

# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["Label"] = df["Index"].map(LABEL_MAP)
    df["Height_m"] = df["Height"] / 100
    df["BMI"] = df["Weight"] / (df["Height_m"] ** 2)
    return df


# ─────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
def build_feature_vector(gender: str, height: float, weight: float) -> list:
    h_m = height / 100
    bmi = weight / (h_m ** 2)
    gender_num = 1 if gender == "Male" else 0
    h_cat = 0 if height <= 155 else 1 if height <= 170 else 2 if height <= 185 else 3
    w_cat = 0 if weight <= 60 else 1 if weight <= 80 else 2 if weight <= 100 else 3
    return [
        gender_num, height, weight, h_m, bmi, bmi ** 2,
        weight / height, weight ** 2, height ** 2, h_cat, w_cat,
    ]


FEATURE_COLS = [
    "Gender_num", "Height", "Weight", "Height_m", "BMI", "BMI_squared",
    "Weight_Height_ratio", "Weight_squared", "Height_squared",
    "Height_category", "Weight_category",
]


@st.cache_data
def prepare_features(df):
    df2 = df.copy()
    df2["Gender_num"] = (df2["Gender"] == "Male").astype(int)
    df2["Height_m"] = df2["Height"] / 100
    df2["BMI"] = df2["Weight"] / (df2["Height_m"] ** 2)
    df2["BMI_squared"] = df2["BMI"] ** 2
    df2["Weight_Height_ratio"] = df2["Weight"] / df2["Height"]
    df2["Weight_squared"] = df2["Weight"] ** 2
    df2["Height_squared"] = df2["Height"] ** 2
    df2["Height_category"] = pd.cut(df2["Height"], bins=[0, 155, 170, 185, 300], labels=[0, 1, 2, 3]).astype(int)
    df2["Weight_category"] = pd.cut(df2["Weight"], bins=[0, 60, 80, 100, 300], labels=[0, 1, 2, 3]).astype(int)
    X = df2[FEATURE_COLS]
    y = df2["Index"]
    return X, y


# ─────────────────────────────────────────────────────────────────
# BMI CATEGORY FROM FORMULA
# ─────────────────────────────────────────────────────────────────
def bmi_category_from_value(bmi: float) -> str:
    if bmi < 16:
        return "Extremely Weak"
    elif bmi < 18.5:
        return "Weak"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obesity"
    else:
        return "Extreme Obesity"


def ideal_weight_range(height_cm: float, gender: str) -> tuple:
    h_m = height_cm / 100
    low = 18.5 * h_m ** 2
    high = 24.9 * h_m ** 2
    return round(low, 1), round(high, 1)


# ─────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def train_all_models(X_arr, y_arr, feature_cols):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False

    X = pd.DataFrame(X_arr, columns=feature_cols)
    y = pd.Series(y_arr)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    configs = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", random_state=42, probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    }
    if has_xgb:
        configs["XGBoost"] = XGBClassifier(n_estimators=150, random_state=42, eval_metric="mlogloss", verbosity=0)

    results = {}
    for name, model in configs.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_test": y_test.values,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    return results, scaler, best_name


# ─────────────────────────────────────────────────────────────────
# BMI GAUGE (SVG-based via plotly)
# ─────────────────────────────────────────────────────────────────
def make_bmi_gauge(bmi_val: float, category: str):
    color = LABEL_COLORS.get(category, "#888888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=bmi_val,
        number={"suffix": " kg/m²", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [10, 45], "tickwidth": 1, "tickcolor": "#888"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [10, 16], "color": "#e74c3c30"},
                {"range": [16, 18.5], "color": "#e67e2230"},
                {"range": [18.5, 25], "color": "#27ae6030"},
                {"range": [25, 30], "color": "#f39c1230"},
                {"range": [30, 35], "color": "#e74c3c30"},
                {"range": [35, 45], "color": "#c0392b30"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": bmi_val,
            },
        },
        title={"text": f"<b>{category}</b>", "font": {"size": 16, "color": color}},
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cccccc"},
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# BODY SILHOUETTE (SVG)
# ─────────────────────────────────────────────────────────────────
def make_body_svg(gender: str, bmi: float, category: str) -> str:
    color = LABEL_COLORS.get(category, "#7c3aed")

    # Body proportions based on BMI
    bmi_clamped = max(12, min(50, bmi))
    fatness = (bmi_clamped - 12) / (50 - 12)  # 0=lean, 1=obese

    waist_w = 28 + fatness * 52    # 28px lean → 80px obese
    hip_w = 32 + fatness * 48
    chest_w = 30 + fatness * 36
    shoulder_w = 44 + fatness * 20 if gender == "Male" else 38 + fatness * 14
    neck_w = 9 + fatness * 8
    head_r = 20
    belly_extra = fatness * 18

    cx = 90
    head_cy = 32
    neck_top = head_cy + head_r
    neck_bot = neck_top + 18
    chest_top = neck_bot
    chest_bot = chest_top + 55
    waist_top = chest_bot
    waist_bot = waist_top + 35
    hip_top = waist_bot
    hip_bot = hip_top + 40
    leg_bot = hip_bot + 90
    leg_w = 13 + fatness * 10

    # Arm
    arm_top_x = cx - shoulder_w * 0.5
    arm_bot_x = arm_top_x - 6
    arm_w = 10 + fatness * 6

    glow = f"drop-shadow(0 0 12px {color}88)"

    # Build SVG path for torso (mirrored)
    torso_path = (
        f"M {cx - neck_w/2:.1f} {neck_top:.1f} "
        f"C {cx - shoulder_w/2:.1f} {neck_top:.1f}, {cx - shoulder_w/2:.1f} {chest_top:.1f}, {cx - chest_w/2:.1f} {chest_bot:.1f} "
        f"C {cx - waist_w/2 - belly_extra:.1f} {waist_top + 15:.1f}, {cx - hip_w/2:.1f} {waist_bot:.1f}, {cx - hip_w/2:.1f} {hip_bot:.1f} "
        f"L {cx + hip_w/2:.1f} {hip_bot:.1f} "
        f"C {cx + hip_w/2:.1f} {waist_bot:.1f}, {cx + waist_w/2 + belly_extra:.1f} {waist_top + 15:.1f}, {cx + chest_w/2:.1f} {chest_bot:.1f} "
        f"C {cx + shoulder_w/2:.1f} {chest_top:.1f}, {cx + shoulder_w/2:.1f} {neck_top:.1f}, {cx + neck_w/2:.1f} {neck_top:.1f} "
        f"Z"
    )

    # Breast indicator for female
    breast_svg = ""
    if gender == "Female":
        bx1 = cx - chest_w * 0.28
        bx2 = cx + chest_w * 0.28
        by = chest_top + 30
        br = 7 + fatness * 5
        breast_svg = f"""
        <ellipse cx="{bx1:.1f}" cy="{by:.1f}" rx="{br:.1f}" ry="{br*0.7:.1f}"
                 fill="{color}55" stroke="{color}88" stroke-width="1"/>
        <ellipse cx="{bx2:.1f}" cy="{by:.1f}" rx="{br:.1f}" ry="{br*0.7:.1f}"
                 fill="{color}55" stroke="{color}88" stroke-width="1"/>
        """

    svg = f"""
    <svg viewBox="0 0 180 330" xmlns="http://www.w3.org/2000/svg"
         style="filter:{glow}; width:100%; height:320px;">
      <defs>
        <radialGradient id="bg_grad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="{color}18"/>
          <stop offset="100%" stop-color="transparent"/>
        </radialGradient>
        <linearGradient id="body_grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="{color}"/>
          <stop offset="100%" stop-color="{color}cc"/>
        </linearGradient>
      </defs>

      <!-- BG glow -->
      <ellipse cx="{cx}" cy="165" rx="80" ry="150" fill="url(#bg_grad)"/>

      <!-- Left arm -->
      <rect x="{cx - shoulder_w/2 - arm_w:.1f}" y="{chest_top:.1f}"
            width="{arm_w:.1f}" height="{waist_bot - chest_top + 20:.1f}"
            rx="{arm_w/2:.1f}" fill="url(#body_grad)" opacity="0.85"/>

      <!-- Right arm -->
      <rect x="{cx + shoulder_w/2:.1f}" y="{chest_top:.1f}"
            width="{arm_w:.1f}" height="{waist_bot - chest_top + 20:.1f}"
            rx="{arm_w/2:.1f}" fill="url(#body_grad)" opacity="0.85"/>

      <!-- Torso -->
      <path d="{torso_path}" fill="url(#body_grad)" opacity="0.95"/>

      <!-- Female breasts -->
      {breast_svg}

      <!-- Left leg -->
      <rect x="{cx - hip_w*0.35 - leg_w:.1f}" y="{hip_bot:.1f}"
            width="{leg_w:.1f}" height="{leg_bot - hip_bot:.1f}"
            rx="{leg_w/2:.1f}" fill="url(#body_grad)" opacity="0.9"/>

      <!-- Right leg -->
      <rect x="{cx + hip_w*0.35:.1f}" y="{hip_bot:.1f}"
            width="{leg_w:.1f}" height="{leg_bot - hip_bot:.1f}"
            rx="{leg_w/2:.1f}" fill="url(#body_grad)" opacity="0.9"/>

      <!-- Neck -->
      <rect x="{cx - neck_w/2:.1f}" y="{neck_top:.1f}"
            width="{neck_w:.1f}" height="18"
            rx="{neck_w/2:.1f}" fill="url(#body_grad)"/>

      <!-- Head -->
      <circle cx="{cx}" cy="{head_cy}" r="{head_r}" fill="url(#body_grad)"/>

      <!-- Eyes -->
      <circle cx="{cx - 6}" cy="{head_cy - 3}" r="2.5" fill="white" opacity="0.9"/>
      <circle cx="{cx + 6}" cy="{head_cy - 3}" r="2.5" fill="white" opacity="0.9"/>
      <circle cx="{cx - 6}" cy="{head_cy - 3}" r="1.2" fill="rgba(0,0,0,0.6)"/>
      <circle cx="{cx + 6}" cy="{head_cy - 3}" r="1.2" fill="rgba(0,0,0,0.6)"/>

      <!-- Hair (female) -->
      {"" if gender == "Male" else f'<ellipse cx="{cx}" cy="{head_cy - head_r*0.6:.1f}" rx="{head_r*1.15:.1f}" ry="{head_r*0.55:.1f}" fill="{color}dd"/>'}

      <!-- BMI label ring -->
      <circle cx="{cx}" cy="310" r="12" fill="none" stroke="{color}" stroke-width="2" opacity="0.5"/>
      <text x="{cx}" y="314" text-anchor="middle" font-size="8"
            fill="{color}" font-weight="bold">{bmi:.1f}</text>
    </svg>
    """
    return svg


# ─────────────────────────────────────────────────────────────────
# PDF EXPORT
# ─────────────────────────────────────────────────────────────────
def generate_pdf_report(gender, height, weight, bmi, category, ideal_low, ideal_high):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=22, spaceAfter=6,
                                     textColor=colors.HexColor("#7c3aed"))
        story.append(Paragraph("BMI Analyzer Pro — Laporan Hasil", title_style))
        story.append(Spacer(1, 0.4*cm))

        data = [
            ["Parameter", "Nilai"],
            ["Jenis Kelamin", gender],
            ["Tinggi Badan", f"{height} cm"],
            ["Berat Badan", f"{weight} kg"],
            ["BMI", f"{bmi:.2f} kg/m²"],
            ["Kategori", category],
            ["Berat Ideal", f"{ideal_low} – {ideal_high} kg"],
        ]
        t = Table(data, colWidths=[8*cm, 8*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f8ff")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f0ff")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        tips = HEALTH_TIPS.get(category, {})
        if tips:
            story.append(Paragraph(f"Rekomendasi: {tips.get('summary', '')}", styles["Normal"]))

        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────
def main():
    df = load_data()

    # Inject CSS
    st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────────────────
    with st.sidebar:
        # Dark/Light toggle
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.markdown("### ⚙️ Pengaturan")
        with col_t2:
            if st.button("🌙" if not st.session_state.dark_mode else "☀️", key="theme_btn"):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()

        st.divider()
        st.markdown("### 📋 Input Data")

        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"],
                              format_func=lambda x: "👨 Laki-laki" if x == "Male" else "👩 Perempuan")
        height = st.slider("Tinggi Badan (cm)", 140, 210, 170, step=1)
        weight = st.slider("Berat Badan (kg)", 30, 200, 70, step=1)
        age = st.slider("Usia (tahun)", 10, 80, 25, step=1)

        st.divider()

        # Live BMI preview in sidebar
        h_m = height / 100
        bmi_live = weight / (h_m ** 2)
        cat_live = bmi_category_from_value(bmi_live)
        color_live = LABEL_COLORS.get(cat_live, "#888")
        ideal_low, ideal_high = ideal_weight_range(height, gender)

        st.markdown(f"""
        <div style="background:{color_live}22; border:1px solid {color_live}55;
                    border-radius:12px; padding:1rem; text-align:center;">
            <div style="font-size:2rem; font-weight:900; color:{color_live};">{bmi_live:.1f}</div>
            <div style="font-size:0.8rem; color:{color_live}; font-weight:600;">{cat_live}</div>
            <div style="font-size:0.72rem; color:#888; margin-top:4px;">Ideal: {ideal_low}–{ideal_high} kg</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # PDF download
        pdf_bytes = generate_pdf_report(gender, height, weight, bmi_live, cat_live, ideal_low, ideal_high)
        if pdf_bytes:
            st.download_button(
                "📄 Download Laporan PDF",
                data=pdf_bytes,
                file_name=f"bmi_report_{gender.lower()}_{height}cm_{weight}kg.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("Install `reportlab` untuk fitur PDF.")

    # ── HEADER ───────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>⚡ BMI Analyzer Pro</h1>
        <p>Machine Learning · Data Mining · Health Intelligence — Universitas AMIKOM Yogyakarta</p>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Kalkulator", "📊 Analisis Data", "🤖 Model ML", "💡 Rekomendasi", "🎯 Batch Prediksi"
    ])

    # ═══════════════════════════════════════════════════════════
    # TAB 1 — KALKULATOR BMI
    # ═══════════════════════════════════════════════════════════
    with tab1:
        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{bmi_live:.1f}</div>
                <div class="label">BMI Score</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{height}</div>
                <div class="label">Tinggi (cm)</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{weight}</div>
                <div class="label">Berat (kg)</div></div>""", unsafe_allow_html=True)
        with m4:
            diff = round(weight - (ideal_low + ideal_high) / 2, 1)
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            st.markdown(f"""<div class="metric-card">
                <div class="value" style="font-size:1.6rem;">{diff_str} kg</div>
                <div class="label">dari Ideal</div></div>""", unsafe_allow_html=True)

        st.markdown("")

        col_sil, col_gauge, col_info = st.columns([1, 1.4, 1.6])

        with col_sil:
            st.markdown("##### 🧍 Body Silhouette")
            svg_body = make_body_svg(gender, bmi_live, cat_live)
            st.markdown(svg_body, unsafe_allow_html=True)

        with col_gauge:
            st.markdown("##### 📈 BMI Gauge")
            fig_gauge = make_bmi_gauge(bmi_live, cat_live)
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            # Category range bars
            st.markdown("**Rentang Kategori BMI:**")
            cats_display = [
                ("Extremely Weak", "< 16", "#e74c3c"),
                ("Weak", "16 – 18.5", "#e67e22"),
                ("Normal ✓", "18.5 – 25", "#27ae60"),
                ("Overweight", "25 – 30", "#f39c12"),
                ("Obesity", "30 – 35", "#e74c3c"),
                ("Extreme Obesity", "> 35", "#c0392b"),
            ]
            for cname, crange, ccolor in cats_display:
                active = cat_live.lower() in cname.lower()
                weight_str = "800" if active else "400"
                bg = f"{ccolor}22" if active else "transparent"
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px; padding:4px 8px;
                            border-radius:6px; background:{bg}; margin:2px 0;">
                    <div style="width:10px; height:10px; border-radius:50%;
                                background:{ccolor}; flex-shrink:0;"></div>
                    <span style="font-size:0.8rem; font-weight:{weight_str};
                                 color:{ccolor if active else '#888'};">{cname}</span>
                    <span style="margin-left:auto; font-size:0.75rem; color:#888;">{crange}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_info:
            st.markdown("##### 📋 Detail Analisis")

            color_card = LABEL_COLORS.get(cat_live, "#7c3aed")
            grad = LABEL_GRADIENT.get(cat_live, "linear-gradient(135deg, #7c3aed, #06b6d4)")
            st.markdown(f"""
            <div class="bmi-result-card" style="background:{grad};">
                <div class="bmi-number">{bmi_live:.2f}</div>
                <div class="bmi-label">{cat_live}</div>
                <div style="color:rgba(255,255,255,0.75); font-size:0.85rem; margin-top:8px;">
                    {gender} · {height} cm · {weight} kg · {age} tahun
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Info grid
            info_items = [
                ("⚖️ Berat Ideal", f"{ideal_low} – {ideal_high} kg"),
                ("📏 Tinggi (m)", f"{h_m:.2f} m"),
                ("📐 BMI Formula", f"{weight} ÷ {h_m:.2f}² = {bmi_live:.2f}"),
                ("🎯 Target Ideal", f"BMI 18.5 – 24.9"),
            ]
            for icon_label, val in info_items:
                st.markdown(f"""
                <div class="tip-item" style="display:flex; justify-content:space-between;">
                    <span>{icon_label}</span>
                    <b>{val}</b>
                </div>
                """, unsafe_allow_html=True)

            # ML Prediction (if trained)
            if st.session_state.models_trained and st.session_state.best_model and st.session_state.scaler:
                st.markdown("##### 🤖 Prediksi ML")
                fv = build_feature_vector(gender, height, weight)
                fv_scaled = st.session_state.scaler.transform([fv])
                pred_idx = st.session_state.best_model.predict(fv_scaled)[0]
                pred_proba = st.session_state.best_model.predict_proba(fv_scaled)[0]
                pred_label = LABEL_MAP[pred_idx]
                confidence = pred_proba[pred_idx] * 100

                st.markdown(f"""
                <div class="insight-box">
                    🤖 Model prediksi: <b>{pred_label}</b><br>
                    🎯 Confidence: <b>{confidence:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)

                # Mini prob bars
                for i, (lbl, prob) in enumerate(zip(LABEL_MAP.values(), pred_proba)):
                    bar_color = LABEL_COLORS.get(lbl, "#888")
                    pct = prob * 100
                    st.markdown(f"""
                    <div style="margin:3px 0;">
                        <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#888;">
                            <span>{lbl}</span><span>{pct:.1f}%</span>
                        </div>
                        <div class="progress-bar-custom">
                            <div class="progress-bar-fill" style="width:{pct:.1f}%; background:{bar_color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("💡 Latih model ML di tab **🤖 Model ML** untuk prediksi berbasis ML.")

    # ═══════════════════════════════════════════════════════════
    # TAB 2 — ANALISIS DATA
    # ═══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📊 Exploratory Data Analysis")
        st.caption(f"Dataset: {len(df)} sampel · {df['Gender'].nunique()} gender · {df['Index'].nunique()} kategori BMI")

        # Descriptive stats
        with st.expander("📋 Statistik Deskriptif", expanded=False):
            desc = df[["Height", "Weight", "BMI"]].describe().round(2)
            st.dataframe(desc, use_container_width=True)

        # Row 1: Distribution + Gender
        c1, c2 = st.columns(2)
        with c1:
            label_counts = df["Label"].value_counts().reset_index()
            label_counts.columns = ["Kategori", "Jumlah"]
            label_counts["Color"] = label_counts["Kategori"].map(LABEL_COLORS)
            fig_dist = px.bar(
                label_counts, x="Kategori", y="Jumlah",
                color="Kategori",
                color_discrete_map=LABEL_COLORS,
                title="📊 Distribusi Kategori BMI",
                text="Jumlah",
            )
            fig_dist.update_traces(textposition="outside", marker_line_width=0)
            fig_dist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False, font={"color": "#cccccc"},
                xaxis={"tickangle": -20}, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with c2:
            gender_counts = df["Gender"].value_counts().reset_index()
            gender_counts.columns = ["Gender", "Jumlah"]
            fig_gender = px.pie(
                gender_counts, names="Gender", values="Jumlah",
                title="👥 Distribusi Gender",
                color_discrete_sequence=["#7c3aed", "#06b6d4"],
                hole=0.45,
            )
            fig_gender.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#cccccc"}, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        # Row 2: Scatter + BMI Distribution
        c3, c4 = st.columns(2)
        with c3:
            fig_scatter = px.scatter(
                df, x="Height", y="Weight", color="Label",
                color_discrete_map=LABEL_COLORS,
                title="🔵 Scatter: Tinggi vs Berat",
                opacity=0.7, size_max=8,
                hover_data={"BMI": ":.1f", "Gender": True},
                labels={"Height": "Tinggi (cm)", "Weight": "Berat (kg)"},
            )
            fig_scatter.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#cccccc"}, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c4:
            fig_bmi_dist = px.histogram(
                df, x="BMI", color="Gender",
                color_discrete_sequence=["#7c3aed", "#06b6d4"],
                title="📈 Distribusi BMI",
                nbins=30, barmode="overlay", opacity=0.75,
                labels={"BMI": "BMI (kg/m²)"},
            )
            fig_bmi_dist.add_vline(x=18.5, line_dash="dash", line_color="#27ae60", annotation_text="18.5")
            fig_bmi_dist.add_vline(x=25, line_dash="dash", line_color="#f39c12", annotation_text="25")
            fig_bmi_dist.add_vline(x=30, line_dash="dash", line_color="#e74c3c", annotation_text="30")
            fig_bmi_dist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#cccccc"}, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_bmi_dist, use_container_width=True)

        # Row 3: Box plots + Heatmap
        c5, c6 = st.columns(2)
        with c5:
            fig_box = px.box(
                df, x="Label", y="BMI", color="Label",
                color_discrete_map=LABEL_COLORS,
                title="📦 Box Plot BMI per Kategori",
                labels={"Label": "Kategori", "BMI": "BMI"},
            )
            fig_box.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#cccccc"}, showlegend=False,
                xaxis={"tickangle": -20}, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with c6:
            corr_cols = ["Height", "Weight", "BMI", "Index"]
            corr_data = df[corr_cols].corr().round(3)
            fig_heat = px.imshow(
                corr_data,
                title="🌡️ Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=True,
            )
            fig_heat.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#cccccc"}, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # Auto-insights
        st.markdown("### 💡 Auto-Insights")
        bmi_mean = df["BMI"].mean()
        bmi_std = df["BMI"].std()
        top_cat = df["Label"].value_counts().index[0]
        pct_top = df["Label"].value_counts().iloc[0] / len(df) * 100
        male_pct = (df["Gender"] == "Male").mean() * 100

        insights = [
            f"📌 Rata-rata BMI dataset adalah **{bmi_mean:.1f}** (std: {bmi_std:.1f}) — termasuk kategori **{bmi_category_from_value(bmi_mean)}**.",
            f"📌 Kategori paling umum: **{top_cat}** ({pct_top:.1f}% dari total {len(df)} sampel).",
            f"📌 Komposisi gender: **{male_pct:.1f}% Male** dan **{100-male_pct:.1f}% Female**.",
            f"📌 Korelasi tertinggi: BMI ↔ Index ({corr_data.loc['BMI','Index']:.2f}) — sangat kuat, seperti yang diharapkan.",
            f"📌 Dataset menunjukkan skew ke kategori berat berlebih ({df[df['Index']>=3].shape[0]} dari {len(df)} sampel ≥ Overweight).",
        ]
        for ins in insights:
            st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 3 — MODEL ML
    # ═══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🤖 Machine Learning — Model Comparison")

        if not st.session_state.models_trained:
            st.info("Klik tombol di bawah untuk melatih semua model ML pada dataset.")
            if st.button("🚀 Latih Semua Model ML", use_container_width=False):
                with st.spinner("Melatih model... ini mungkin memerlukan beberapa detik..."):
                    X, y = prepare_features(df)
                    results, scaler_obj, best_name = train_all_models(
                        X.values, y.values, list(X.columns)
                    )
                    st.session_state.training_results = results
                    st.session_state.best_model = results[best_name]["model"]
                    st.session_state.scaler = scaler_obj
                    st.session_state.best_name = best_name
                    st.session_state.models_trained = True
                st.rerun()
        else:
            results = st.session_state.training_results
            best_name = st.session_state.best_name

            st.success(f"✅ Model terbaik: **{best_name}** dengan akurasi **{results[best_name]['accuracy']*100:.2f}%**")

            # Metrics table
            metrics_data = []
            for name, r in results.items():
                metrics_data.append({
                    "Model": name,
                    "Accuracy": f"{r['accuracy']*100:.2f}%",
                    "F1 Score": f"{r['f1']*100:.2f}%",
                    "Precision": f"{r['precision']*100:.2f}%",
                    "Recall": f"{r['recall']*100:.2f}%",
                    "Best": "⭐" if name == best_name else "",
                })
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

            # Accuracy comparison chart
            model_names = list(results.keys())
            accs = [results[m]["accuracy"] * 100 for m in model_names]
            f1s = [results[m]["f1"] * 100 for m in model_names]

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                name="Accuracy", x=model_names, y=accs,
                marker_color="#7c3aed", text=[f"{v:.1f}%" for v in accs],
                textposition="outside",
            ))
            fig_compare.add_trace(go.Bar(
                name="F1 Score", x=model_names, y=f1s,
                marker_color="#06b6d4", text=[f"{v:.1f}%" for v in f1s],
                textposition="outside",
            ))
            fig_compare.update_layout(
                title="📊 Perbandingan Performa Model",
                barmode="group",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#cccccc"},
                yaxis={"range": [0, 110]},
                legend={"orientation": "h", "y": -0.2},
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # Confusion matrices
            st.markdown("#### 🔢 Confusion Matrices")
            target_names = ["Ext.Weak", "Weak", "Normal", "Overweight", "Obesity", "Ext.Obesity"]
            model_list = list(results.items())

            for i in range(0, len(model_list), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(model_list):
                        m_name, r = model_list[i + j]
                        cm = r["confusion_matrix"]
                        with col:
                            fig_cm = px.imshow(
                                cm,
                                x=target_names[:cm.shape[1]],
                                y=target_names[:cm.shape[0]],
                                title=f"{m_name} ({r['accuracy']*100:.1f}%)",
                                color_continuous_scale="Purples",
                                text_auto=True,
                            )
                            fig_cm.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                font={"color": "#cccccc"},
                                margin=dict(t=60, b=20),
                                xaxis={"tickangle": -25},
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)

            # Feature importance (Random Forest)
            if "Random Forest" in results:
                rf_model = results["Random Forest"]["model"]
                if hasattr(rf_model, "feature_importances_"):
                    st.markdown("#### 🎯 Feature Importance (Random Forest)")
                    fi = pd.DataFrame({
                        "Feature": FEATURE_COLS,
                        "Importance": rf_model.feature_importances_,
                    }).sort_values("Importance", ascending=True).tail(10)
                    fig_fi = px.bar(
                        fi, x="Importance", y="Feature",
                        orientation="h",
                        title="Top 10 Feature Importance",
                        color="Importance",
                        color_continuous_scale="Purples",
                    )
                    fig_fi.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#cccccc"}, showlegend=False,
                        margin=dict(t=60, b=20),
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 4 — REKOMENDASI
    # ═══════════════════════════════════════════════════════════
    with tab4:
        tips = HEALTH_TIPS.get(cat_live, {})
        color_r = LABEL_COLORS.get(cat_live, "#7c3aed")
        grad_r = LABEL_GRADIENT.get(cat_live, "linear-gradient(135deg, #7c3aed, #06b6d4)")

        st.markdown(f"""
        <div style="background:{grad_r}; border-radius:16px; padding:1.5rem 2rem; margin-bottom:1.5rem;">
            <div style="font-size:2.5rem;">{tips.get('icon', '💡')}</div>
            <div style="color:white; font-size:1.1rem; font-weight:700; margin-top:0.3rem;">
                {cat_live} — BMI {bmi_live:.1f}
            </div>
            <div style="color:rgba(255,255,255,0.85); font-size:0.95rem; margin-top:0.4rem;">
                {tips.get('summary', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        c_diet, c_ex = st.columns(2)
        with c_diet:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("##### 🥗 Pola Makan")
            for tip in tips.get("diet", []):
                st.markdown(f'<div class="tip-item">✓ {tip}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_ex:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("##### 🏃 Olahraga")
            for tip in tips.get("exercise", []):
                st.markdown(f'<div class="tip-item">✓ {tip}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c_hyd, c_sleep, c_life = st.columns(3)
        with c_hyd:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("##### 💧 Hidrasi")
            st.markdown(f'<div class="tip-item">{tips.get("hydration", "")}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_sleep:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("##### 😴 Tidur")
            st.markdown(f'<div class="tip-item">{tips.get("sleep", "")}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_life:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("##### 🌟 Lifestyle")
            for tip in tips.get("lifestyle", []):
                st.markdown(f'<div class="tip-item">✓ {tip}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # BMI Journey tracker
        st.markdown("### 📈 BMI Journey Tracker")
        st.markdown("Masukkan data historis BMI kamu (simulasi):")
        col_hist = st.columns(5)
        history_bmis = []
        for i, col in enumerate(col_hist):
            with col:
                val = st.number_input(f"Bulan -{5-i}", value=round(bmi_live + (5-i)*0.8, 1),
                                      min_value=10.0, max_value=60.0, step=0.1, key=f"hist_{i}")
                history_bmis.append(val)
        history_bmis.append(bmi_live)

        months = [f"Bulan -{5-i}" for i in range(5)] + ["Sekarang"]
        colors_journey = [LABEL_COLORS.get(bmi_category_from_value(b), "#888") for b in history_bmis]

        fig_journey = go.Figure()
        fig_journey.add_trace(go.Scatter(
            x=months, y=history_bmis,
            mode="lines+markers+text",
            text=[f"{b:.1f}" for b in history_bmis],
            textposition="top center",
            line=dict(color="#7c3aed", width=3),
            marker=dict(size=12, color=colors_journey, line=dict(color="white", width=2)),
        ))
        fig_journey.add_hrect(y0=18.5, y1=25, fillcolor="#27ae6015", line_width=0,
                               annotation_text="Normal Zone", annotation_position="top right")
        fig_journey.update_layout(
            title="📈 Tren BMI (6 bulan terakhir)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#cccccc"},
            yaxis={"title": "BMI"},
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig_journey, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 5 — BATCH PREDIKSI
    # ═══════════════════════════════════════════════════════════
    with tab5:
        st.markdown("### 🎯 Batch Prediksi BMI")
        st.caption("Upload CSV dengan kolom: Gender, Height (cm), Weight (kg)")

        sample_csv = pd.DataFrame({
            "Gender": ["Male", "Female", "Male"],
            "Height": [175, 160, 180],
            "Weight": [70, 55, 95],
        })
        st.markdown("**Contoh format CSV:**")
        st.dataframe(sample_csv, use_container_width=False)

        csv_sample = sample_csv.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Template CSV", csv_sample, "template_bmi.csv", "text/csv")

        uploaded_file = st.file_uploader("Upload CSV kamu", type=["csv"])
        if uploaded_file:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"✅ {len(df_batch)} baris ditemukan.")

            # Compute BMI
            df_batch["Height_m"] = df_batch["Height"] / 100
            df_batch["BMI"] = df_batch["Weight"] / (df_batch["Height_m"] ** 2)
            df_batch["Kategori_Formula"] = df_batch["BMI"].apply(bmi_category_from_value)

            # ML prediction if trained
            if st.session_state.models_trained and st.session_state.best_model:
                feats = []
                for _, row in df_batch.iterrows():
                    feats.append(build_feature_vector(row["Gender"], row["Height"], row["Weight"]))
                feats_sc = st.session_state.scaler.transform(feats)
                preds = st.session_state.best_model.predict(feats_sc)
                df_batch["Kategori_ML"] = [LABEL_MAP[p] for p in preds]

            df_batch = df_batch.drop(columns=["Height_m"])
            st.dataframe(df_batch, use_container_width=True)

            # Download result
            csv_out = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Hasil Prediksi", csv_out,
                               "hasil_prediksi_bmi.csv", "text/csv")

            # Summary chart
            fig_batch = px.histogram(
                df_batch, x="Kategori_Formula",
                color="Kategori_Formula",
                color_discrete_map=LABEL_COLORS,
                title="Distribusi Kategori BMI (Batch)",
            )
            fig_batch.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#cccccc"}, showlegend=False,
            )
            st.plotly_chart(fig_batch, use_container_width=True)


if __name__ == "__main__":
    main()
