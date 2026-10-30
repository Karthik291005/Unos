# =========================================================
# KIDNEY GRAFT FAILURE RISK MONITOR
# Clean input → predict → show results
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import io
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Kidney Graft Risk Monitor",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    section[data-testid="stSidebar"] { background-color: #1a1f2e; }
    .main-title { font-size:2rem; font-weight:900; color:#63b3ed; }
    .main-sub   { font-size:0.85rem; color:#718096; margin-bottom:1rem; }
    .sec-header {
        font-size:0.95rem; font-weight:700; color:#63b3ed;
        border-bottom:1px solid #2d3748; padding-bottom:0.3rem;
        margin:1rem 0 0.6rem 0; text-transform:uppercase; letter-spacing:0.5px;
    }
    .kpi-card {
        background:linear-gradient(135deg,#1a202c,#2d3748);
        border:1px solid #2d3748; border-radius:12px;
        padding:1rem; text-align:center; margin-bottom:0.5rem;
    }
    .kpi-val   { font-size:1.8rem; font-weight:800; line-height:1.1; }
    .kpi-red   { color:#fc8181; }
    .kpi-green { color:#68d391; }
    .kpi-yellow{ color:#f6e05e; }
    .kpi-blue  { color:#63b3ed; }
    .kpi-label { font-size:0.72rem; color:#718096; text-transform:uppercase; letter-spacing:0.8px; margin-top:0.2rem; }
    .alert-high     { background:linear-gradient(135deg,#742a2a,#9b2c2c); border-left:5px solid #fc8181; border-radius:10px; padding:1rem 1.4rem; margin:0.5rem 0; color:#fff5f5; }
    .alert-moderate { background:linear-gradient(135deg,#744210,#975a16); border-left:5px solid #f6e05e; border-radius:10px; padding:1rem 1.4rem; margin:0.5rem 0; color:#fffff0; }
    .alert-low      { background:linear-gradient(135deg,#1c4532,#276749); border-left:5px solid #68d391; border-radius:10px; padding:1rem 1.4rem; margin:0.5rem 0; color:#f0fff4; }
    .badge-red    { background:#742a2a; color:#fc8181; border-radius:5px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
    .badge-green  { background:#1c4532; color:#68d391; border-radius:5px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
    .badge-yellow { background:#744210; color:#f6e05e; border-radius:5px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
    .result-box {
        background:linear-gradient(135deg,#1a202c,#2d3748);
        border:1px solid #2d3748; border-radius:14px;
        padding:1.5rem; margin-top:0.5rem;
    }
    .footer { text-align:center; color:#4a5568; font-size:0.75rem; margin-top:2rem; padding-top:0.8rem; border-top:1px solid #2d3748; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MATPLOTLIB DARK THEME
# =========================================================
plt.rcParams.update({
    'figure.facecolor': '#1a202c', 'axes.facecolor':  '#1a202c',
    'axes.edgecolor':   '#2d3748', 'axes.labelcolor': '#a0aec0',
    'xtick.color':      '#718096', 'ytick.color':     '#718096',
    'text.color':       '#e2e8f0', 'grid.color':      '#2d3748',
    'grid.alpha': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10, 'axes.titlesize': 11,
    'figure.dpi': 120, 'savefig.dpi': 300,
})

COLORS = {
    'high':     '#fc8181', 'moderate': '#f6e05e',
    'low':      '#68d391', 'blue':     '#63b3ed',
    'baseline': '#718096', 'trigger':  '#ed8936',
    'thresh':   '#48bb78', 'orange':   '#f6ad55',
    'purple':   '#b794f4', 'teal':     '#4fd1c7',
}

BASELINE    = 13.42
TRIGGER     = 18.42
HIGH_THRESH = 30.0
HISTORY_FILE = "processed_outputs/patient_history.json"

# =========================================================
# MODEL LOAD
# =========================================================
@st.cache_resource
def load_model():
    mp = "processed_outputs/gradient_boosting_model.pkl"
    pp = "processed_outputs/preprocessor.pkl"
    try:
        if os.path.exists(mp) and os.path.exists(pp):
            with open(mp, "rb") as f:
                model = pickle.load(f)
            with open(pp, "rb") as f:
                prep = pickle.load(f)
            return model, prep, True
    except Exception:
        pass
    return None, None, False

@st.cache_resource
def load_explainer(_model):
    try:
        import shap
        return shap.TreeExplainer(_model), True
    except Exception:
        return None, False

# =========================================================
# CORE FUNCTIONS
# =========================================================
def predict(inputs, model, prep):
    """
    Takes patient inputs dict.
    Returns (risk_pct, method, X_proc).
    """
    if model is not None and prep is not None:
        try:
            pra1 = inputs.get("PRA1")
            flag = 1 if (pra1 is None or
                         (isinstance(pra1, float) and np.isnan(pra1))) else 0
            row = {
                "CREAT":              float(inputs["CREAT"]),
                "FUNC_STAT":          float(inputs["FUNC_STAT"]),
                "PRA1":               float(pra1) if pra1 is not None else np.nan,
                "ACUTE_REJ_EPI":      str(inputs["ACUTE_REJ_EPI"]),
                "DIABETES_DURING":    str(inputs["DIABETES_DURING"]),
                "HLA_TY_METH_CLASS1": str(inputs["HLA_TY_METH_CLASS1"]),
                "PRA1_missing_flag":  flag,
            }
            df_in  = pd.DataFrame([row])
            X_proc = prep.transform(df_in)
            prob   = model.predict_proba(X_proc)[0][1]
            return round(float(prob * 100), 2), "model", X_proc
        except Exception as e:
            st.warning(f"Model prediction failed: {e}. Using evidence-based fallback.")

    # Evidence fallback
    risk  = BASELINE
    creat = float(inputs.get("CREAT", 1.2))
    func  = float(inputs.get("FUNC_STAT", 2070))
    pra1  = float(inputs.get("PRA1") or 0)
    rej   = str(inputs.get("ACUTE_REJ_EPI", "3"))
    diab  = str(inputs.get("DIABETES_DURING", "N"))
    hla   = str(inputs.get("HLA_TY_METH_CLASS1", "DNA"))

    rej_rates  = {"3": 11.9, "998": 29.8, "1": 59.1, "2": 55.4}
    diab_rates = {"N": 13.3, "Y": 10.1,   "U": 24.5}
    hla_rates  = {"DNA": 11.5, "Serology": 15.4,
                   "Other,Specify": 16.2, "Serology; DNA": 17.2}

    if creat < 1.095:
        risk += 35.0
    elif creat < 1.5:
        risk += 3.0
    elif creat < 2.0:
        risk += 1.0
    else:
        risk -= 2.0

    risk += 4.0 if func < 1845.55 else -1.0
    risk += (6.0 + pra1 * 0.08) if pra1 >= 1.0 else 0.0
    risk += rej_rates.get(rej,  11.9) - 11.9
    risk += diab_rates.get(diab, 13.3) - 13.3
    risk += hla_rates.get(hla,  11.5) - 11.5

    return round(float(np.clip(risk, 1.0, 99.0)), 2), "evidence", None


def get_risk_level(r):
    if r >= HIGH_THRESH:
        return "HIGH",     "#fc8181", "🔴"
    elif r >= TRIGGER:
        return "MODERATE", "#f6e05e", "🟡"
    return "LOW",          "#68d391", "🟢"


def shap_contributions(inputs):
    """Evidence-based feature contributions (percentage points from baseline)."""
    creat = float(inputs.get("CREAT", 1.2))
    func  = float(inputs.get("FUNC_STAT", 2070))
    pra1  = float(inputs.get("PRA1") or 0)
    rej   = str(inputs.get("ACUTE_REJ_EPI", "3"))
    diab  = str(inputs.get("DIABETES_DURING", "N"))
    hla   = str(inputs.get("HLA_TY_METH_CLASS1", "DNA"))

    rej_rates  = {"3": 11.9, "998": 29.8, "1": 59.1, "2": 55.4}
    diab_rates = {"N": 13.3, "Y": 10.1,   "U": 24.5}
    hla_rates  = {"DNA": 11.5, "Serology": 15.4,
                   "Other,Specify": 16.2, "Serology; DNA": 17.2}

    if creat < 1.095:
        c_creat = 35.0
    elif creat < 1.5:
        c_creat = 3.0
    elif creat < 2.0:
        c_creat = 1.0
    else:
        c_creat = -2.0

    return {
        "CREAT":              c_creat,
        "FUNC_STAT":          4.0 if func < 1845.55 else -1.0,
        "PRA1":               (6.0 + pra1 * 0.08) if pra1 >= 1.0 else 0.0,
        "ACUTE_REJ_EPI":      rej_rates.get(rej,  11.9) - 11.9,
        "DIABETES_DURING":    diab_rates.get(diab, 13.3) - 13.3,
        "HLA_TY_METH_CLASS1": hla_rates.get(hla,  11.5) - 11.5,
    }

# =========================================================
# HISTORY
# =========================================================
def load_history(pid):
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f).get(str(pid), [])
    except Exception:
        return []

def save_history(pid, record):
    data = {}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                data = json.load(f)
        except Exception:
            pass
    data.setdefault(str(pid), []).append(record)
    data[str(pid)] = data[str(pid)][-30:]
    os.makedirs("processed_outputs", exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def clear_history(pid):
    if not os.path.exists(HISTORY_FILE):
        return
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        data.pop(str(pid), None)
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# =========================================================
# CHARTS
# =========================================================
def chart_gauge(risk_pct):
    level, color, _ = get_risk_level(risk_pct)
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.barh(0.5, TRIGGER,              left=0,        height=0.35, color=COLORS["low"],      alpha=0.3)
    ax.barh(0.5, HIGH_THRESH - TRIGGER,left=TRIGGER,  height=0.35, color=COLORS["moderate"], alpha=0.3)
    ax.barh(0.5, 100 - HIGH_THRESH,    left=HIGH_THRESH, height=0.35, color=COLORS["high"],  alpha=0.3)
    ax.barh(0.5, min(risk_pct, 100),   height=0.35, color=color, alpha=0.8)
    ax.axvline(x=risk_pct, color=color, linewidth=3, ymin=0.05, ymax=0.95)
    for xp, lbl, c in [(9, "LOW", COLORS["low"]),
                        (24, "MODERATE", COLORS["moderate"]),
                        (65, "HIGH", COLORS["high"])]:
        ax.text(xp, 0.88, lbl, ha="center", fontsize=8, color=c, fontweight="bold")
    ax.text(risk_pct, 0.08, f"{risk_pct:.1f}%",
            ha="center", fontsize=14, color=color, fontweight="bold")
    ax.axvline(x=TRIGGER,     color="#4a5568", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(x=HIGH_THRESH, color="#4a5568", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.xaxis.set_visible(True)
    ax.set_xticks([0, TRIGGER, HIGH_THRESH, 100])
    ax.set_xticklabels(["0%", f"{TRIGGER:.0f}%",
                         f"{HIGH_THRESH:.0f}%", "100%"],
                        fontsize=8, color="#718096")
    ax.xaxis.set_tick_params(length=0)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig


def chart_contributions(contrib, risk_pct):
    pairs  = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = [COLORS["high"] if v > 0 else COLORS["low"] for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=colors, alpha=0.85,
                   edgecolor="#0f1117", height=0.6)
    ax.axvline(x=0, color="#4a5568", linewidth=1.5)
    for bar, val in zip(bars, values):
        sign = "+" if val >= 0 else ""
        ax.text(
            val + (0.3 if val >= 0 else -0.3),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.1f}pp",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9, fontweight="bold",
            color=COLORS["high"] if val > 0 else COLORS["low"]
        )
    ax.set_xlabel("Contribution (percentage points from baseline 13.42%)",
                  color="#a0aec0")
    ax.set_title(
        f"Feature Contributions  —  Final risk: {risk_pct:.1f}%\n"
        f"Red = raises risk above baseline  |  Green = lowers risk",
        fontsize=10, fontweight="bold", color="#e2e8f0"
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig


def chart_risk_curves(inputs):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [
        ("CREAT",     0,    6,    1.095,   "INVERSE"),
        ("FUNC_STAT", 1000, 4100, 1845.55, "INVERSE*"),
        ("PRA1",      0,    100,  1.0,     "DIRECT"),
    ]
    for ax, (col, vmin, vmax, tau, direction) in zip(axes, specs):
        x = np.linspace(vmin, vmax, 300)
        if col == "CREAT":
            y = np.where(x < tau, 16 + (tau - x) * 15, 13.42 + (x - tau) * 0.5)
        elif col == "FUNC_STAT":
            y = np.where(x < tau, 16 + (tau - x) / 90, 13.42 - (x - tau) / 600)
        else:
            y = np.where(x < tau, 18.84, 20 + x * 0.06)
        y = np.clip(y, 0, 99)

        ax.fill_between(x, BASELINE, y, where=(y > BASELINE),
                        alpha=0.12, color=COLORS["high"])
        ax.fill_between(x, y, BASELINE, where=(y <= BASELINE),
                        alpha=0.12, color=COLORS["low"])
        ax.plot(x, y, color=COLORS["blue"], linewidth=2.2)
        ax.axhline(y=BASELINE, color=COLORS["baseline"],
                   linestyle="--", linewidth=1.2,
                   label=f"Baseline ({BASELINE:.1f}%)")
        ax.axhline(y=TRIGGER, color=COLORS["trigger"],
                   linestyle="-.", linewidth=1,
                   label=f"Trigger ({TRIGGER:.1f}%)")
        ax.axvline(x=tau, color=COLORS["thresh"],
                   linestyle=":", linewidth=2,
                   label=f"Threshold={tau:.2f}")

        raw_val = inputs.get(col)
        if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
            pt_val = tau
        else:
            pt_val = float(raw_val)

        y_pt = float(np.interp(pt_val, x, y))
        _, pt_color, _ = get_risk_level(y_pt)
        ax.axvline(x=pt_val, color=pt_color, linewidth=2.5, alpha=0.9)
        ax.scatter([pt_val], [y_pt], color=pt_color, s=120, zorder=6,
                   label=f"Patient: {pt_val:.2f}")
        ax.annotate(f"{y_pt:.1f}%", xy=(pt_val, y_pt),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=9, color=pt_color, fontweight="bold")

        ax.set_xlabel(col, fontweight="bold")
        ax.set_ylabel("Failure Risk (%)")
        ax.set_title(f"{col}  |  {direction}\nThreshold={tau:.3f}",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(0, min(70, y.max() * 1.25))

    plt.suptitle("Patient Position on Risk Curves",
                 fontsize=11, fontweight="bold",
                 color="#e2e8f0", y=1.02)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig


def chart_history(history):
    if len(history) < 2:
        return None
    dates  = [h["date"]        for h in history]
    risks  = [h["risk_pct"]    for h in history]
    creats = [h["inputs"].get("CREAT")    for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    pt_colors = [get_risk_level(r)[1] for r in risks]
    ax1.fill_between(range(len(dates)), risks, BASELINE,
                     where=[r > BASELINE for r in risks],
                     alpha=0.12, color=COLORS["high"])
    ax1.plot(range(len(dates)), risks, color=COLORS["blue"],
             linewidth=2.5, marker="o", markersize=7, zorder=3)
    for i, (r, c) in enumerate(zip(risks, pt_colors)):
        ax1.scatter(i, r, color=c, s=80, zorder=5)
        ax1.annotate(f"{r:.1f}%", xy=(i, r),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=8,
                     color=c, fontweight="bold")
    ax1.axhline(y=BASELINE,    color=COLORS["baseline"],
                linestyle=":",  linewidth=1,
                label=f"Baseline ({BASELINE:.1f}%)")
    ax1.axhline(y=TRIGGER,     color=COLORS["trigger"],
                linestyle="--", linewidth=1.2,
                label=f"Trigger ({TRIGGER:.1f}%)")
    ax1.axhline(y=HIGH_THRESH, color=COLORS["high"],
                linestyle="--", linewidth=1.2,
                label=f"High ({HIGH_THRESH:.0f}%)")
    ax1.set_ylabel("Predicted Risk (%)")
    ax1.set_title("Graft Failure Risk Trajectory", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle=":")

    valid_c = [(i, c) for i, c in enumerate(creats) if c is not None]
    if valid_c:
        xi, ci = zip(*valid_c)
        ax2.plot(xi, ci, color=COLORS["purple"],
                 linewidth=2, marker="s", markersize=6)
        ax2.axhline(y=1.095, color=COLORS["thresh"],
                    linestyle=":", linewidth=1.5,
                    label="Threshold=1.095")
        ax2.set_ylabel("Creatinine (mg/dL)")
        ax2.set_title("Creatinine Trend", fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, linestyle=":")

    ax2.set_xticks(range(len(dates)))
    ax2.set_xticklabels(dates, rotation=30, ha="right", fontsize=8)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig


def chart_forecast(history, n_fc=4):
    if len(history) < 3:
        return None
    risks  = np.array([h["risk_pct"] for h in history])
    dates  = [h["date"] for h in history]
    n      = len(risks)
    x      = np.arange(n)
    coeffs = np.polyfit(x, risks, 1)
    slope  = coeffs[0]
    trend  = np.polyval(coeffs, x)
    std    = np.std(risks - trend)

    x_fc   = np.arange(n, n + n_fc)
    y_fc   = np.clip(np.polyval(coeffs, x_fc), 0, 99)

    try:
        last_dt = datetime.strptime(dates[-1], "%Y-%m-%d")
        fc_dates = [(last_dt + timedelta(weeks=4*(i+1))).strftime("%Y-%m-%d")
                    for i in range(n_fc)]
    except Exception:
        fc_dates = [f"+{i+1}mo" for i in range(n_fc)]

    all_dates = dates + fc_dates
    all_y     = np.concatenate([risks, y_fc])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(range(n), risks, color=COLORS["blue"],
            linewidth=2.5, marker="o", markersize=7,
            label="Observed", zorder=4)
    ax.plot(x, trend, color="#4a5568", linewidth=1.2,
            linestyle="--", alpha=0.6,
            label=f"Trend ({slope:+.2f}%/visit)")
    ax.plot(range(n-1, n+n_fc),
            np.concatenate([[risks[-1]], y_fc]),
            color=COLORS["orange"], linewidth=2.5,
            marker="D", markersize=7, linestyle="--",
            label="Forecast", zorder=4)
    upper = np.clip(y_fc + 2*std, 0, 99)
    lower = np.clip(y_fc - 2*std, 0, 99)
    ax.fill_between(range(n, n+n_fc), lower, upper,
                    alpha=0.15, color=COLORS["orange"],
                    label="95% PI")
    ax.axvline(x=n-1, color="#4a5568",
               linewidth=1.5, linestyle=":", alpha=0.7)
    ax.axhline(y=TRIGGER,     color=COLORS["trigger"],
               linestyle="--", linewidth=1.2,
               label=f"Trigger ({TRIGGER:.1f}%)")
    ax.axhline(y=HIGH_THRESH, color=COLORS["high"],
               linestyle="--", linewidth=1.2,
               label=f"High ({HIGH_THRESH:.0f}%)")
    for i, r in enumerate(y_fc):
        _, c, _ = get_risk_level(r)
        ax.annotate(f"{r:.1f}%", xy=(n+i, r),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5,
                    color=c, fontweight="bold")
    ax.set_xticks(range(len(all_dates)))
    ax.set_xticklabels(all_dates, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Predicted Risk (%)")
    ax.set_title(f"Risk Forecast — {n_fc} visits ahead\n"
                 f"Trend: {slope:+.2f}% per visit  |  95% Prediction Interval shown",
                 fontweight="bold", color="#e2e8f0")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.3)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_ylim(0, min(100, max(all_y)*1.3))
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

# =========================================================
# PDF GENERATOR
# =========================================================
def make_pdf(pid, pname, visit, inputs, risk_pct, contrib, history):
    buf = io.BytesIO()
    level, color_str, emoji = get_risk_level(risk_pct)
    rgb_map = {
        COLORS["high"]:     (0.99, 0.51, 0.51),
        COLORS["moderate"]: (0.96, 0.88, 0.37),
        COLORS["low"]:      (0.41, 0.83, 0.57),
    }
    rgb = rgb_map.get(color_str, (0.39, 0.70, 0.93))

    with PdfPages(buf) as pdf:
        # Page 1 — Summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("#1a202c")
        fig.text(0.5, 0.96,
                 "KIDNEY GRAFT FAILURE RISK ASSESSMENT REPORT",
                 ha="center", fontsize=14,
                 fontweight="bold", color="#63b3ed")
        fig.text(0.5, 0.93,
                 "UNOS STAR Registry  |  Gradient Boosting  |  ROC-AUC 0.953",
                 ha="center", fontsize=9, color="#718096")

        for y_pos, label, val in [
            (0.89, "Patient ID :", pid),
            (0.86, "Name       :", pname or "Not provided"),
            (0.83, "Visit Date :", str(visit)),
            (0.80, "Generated  :", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ]:
            fig.text(0.08, y_pos, f"{label}  {val}",
                     fontsize=10, color="#e2e8f0")

        ax_box = fig.add_axes([0.08, 0.64, 0.84, 0.12])
        ax_box.set_facecolor("#2d3748")
        ax_box.text(0.5, 0.65,
                    f"{emoji}  {level} RISK  —  "
                    f"{risk_pct:.1f}% Predicted Graft Failure Risk",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color=rgb, transform=ax_box.transAxes)
        ax_box.text(0.5, 0.2,
                    f"Baseline: {BASELINE:.1f}%  |  "
                    f"Trigger: {TRIGGER:.1f}%  |  "
                    f"High: {HIGH_THRESH:.0f}%",
                    ha="center", va="center",
                    fontsize=9, color="#718096",
                    transform=ax_box.transAxes)
        ax_box.axis("off")

        # Parameters table
        fig.text(0.08, 0.61, "CLINICAL PARAMETERS",
                 fontsize=10, fontweight="bold", color="#63b3ed")
        col_x = [0.08, 0.30, 0.55, 0.75]
        headers = ["Parameter", "Value", "Threshold", "Status"]
        y = 0.585
        for h, cx in zip(headers, col_x):
            fig.text(cx, y, h, fontsize=8.5,
                     fontweight="bold", color="#63b3ed")
        y -= 0.025

        creat_v = inputs.get("CREAT", 2.0)
        func_v  = inputs.get("FUNC_STAT", 2070)
        pra1_v  = inputs.get("PRA1") or 0

        rows = [
            ("CREAT (mg/dL)", f"{float(creat_v):.2f}", "1.095",
             "HIGH RISK" if float(creat_v) < 1.095 else "OK"),
            ("FUNC_STAT",    f"{float(func_v):.0f}",  "1845.55",
             "BELOW" if float(func_v) < 1845.55 else "OK"),
            ("PRA1 (%)",     f"{float(pra1_v):.1f}",  "1.0",
             "ELEVATED" if float(pra1_v) >= 1.0 else "OK"),
            ("Acute Rejection", str(inputs.get("ACUTE_REJ_EPI","3")),
             "Cat 1,2,998", ""),
            ("Diabetes",     str(inputs.get("DIABETES_DURING","N")),
             "Cat U",        ""),
            ("HLA Method",   str(inputs.get("HLA_TY_METH_CLASS1","DNA")),
             "Sero+DNA highest",""),
        ]
        for row in rows:
            for val, cx in zip(row, col_x):
                if any(w in str(val) for w in ["HIGH", "BELOW", "ELEVATED"]):
                    clr = "#fc8181"
                elif "OK" in str(val):
                    clr = "#68d391"
                else:
                    clr = "#e2e8f0"
                fig.text(cx, y, str(val), fontsize=8, color=clr)
            y -= 0.03

        # Contributions
        fig.text(0.08, y - 0.01, "FEATURE CONTRIBUTIONS",
                 fontsize=10, fontweight="bold", color="#63b3ed")
        y -= 0.03
        for feat, val in sorted(contrib.items(),
                                  key=lambda x: abs(x[1]),
                                  reverse=True)[:6]:
            sign = "+" if val >= 0 else ""
            clr  = "#fc8181" if val > 0 else "#68d391"
            fig.text(0.08, y, f"{feat}:", fontsize=8, color="#a0aec0")
            fig.text(0.38, y, f"{sign}{val:.1f} pp",
                     fontsize=8, color=clr, fontweight="bold")
            bw = min(abs(val)/40, 0.35)
            if bw > 0:
                ax_b = fig.add_axes([0.48, y-0.004, bw, 0.014])
                ax_b.barh(0, [abs(val)], color=clr, alpha=0.7)
                ax_b.axis("off")
            y -= 0.028

        fig.text(0.5, 0.02,
                 "For clinical decision support only — "
                 "not a substitute for clinical judgement",
                 ha="center", fontsize=7.5, color="#4a5568")
        pdf.savefig(fig, bbox_inches="tight", facecolor="#1a202c")
        plt.close(fig)

        # Page 2 — History
        if len(history) >= 2:
            hf = chart_history(history)
            if hf:
                hf.patch.set_facecolor("#1a202c")
                hf.suptitle(f"Patient {pid} — Risk History",
                            fontsize=12, fontweight="bold",
                            color="#63b3ed")
                pdf.savefig(hf, bbox_inches="tight",
                            facecolor="#1a202c")
                plt.close(hf)

        # Page 3 — Forecast
        if len(history) >= 3:
            ff = chart_forecast(history)
            if ff:
                ff.patch.set_facecolor("#1a202c")
                pdf.savefig(ff, bbox_inches="tight",
                            facecolor="#1a202c")
                plt.close(ff)

        # Page 4 — Risk curves
        rc = chart_risk_curves(inputs)
        rc.patch.set_facecolor("#1a202c")
        pdf.savefig(rc, bbox_inches="tight", facecolor="#1a202c")
        plt.close(rc)

    buf.seek(0)
    return buf

# =========================================================
# MAIN APP
# =========================================================
def main():
    model, prep, model_loaded = load_model()
    explainer, shap_ok = (load_explainer(model)
                           if model_loaded else (None, False))

    # ── Header ───────────────────────────────────────────
    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown(
            '<span class="main-title">🫘 Kidney Graft Risk Monitor</span>',
            unsafe_allow_html=True)
        st.markdown(
            '<span class="main-sub">Post-Transplant Graft Failure Risk Assessment  |  '
            'UNOS STAR N=606,824  |  Gradient Boosting ROC-AUC=0.953</span>',
            unsafe_allow_html=True)
    with h2:
        badge = "🟢 Model Active" if model_loaded else "🟡 Evidence Mode"
        st.markdown(
            f'<div style="text-align:right;color:#68d391;'
            f'font-size:0.85rem;font-weight:700;padding-top:0.5rem;">'
            f'{badge}</div>',
            unsafe_allow_html=True)

    st.divider()

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("**👤 Patient**")
        pid   = st.text_input("Patient ID", value="PT-001")
        pname = st.text_input("Patient Name", value="")
        vdate = st.date_input("Visit Date", value=datetime.today())

        st.divider()
        st.markdown("**🔧 User Role**")
        role = st.selectbox("Role",
            ["👨‍⚕️ Doctor", "👤 Patient",
             "👩‍⚕️ Nurse", "🔬 Researcher"])

        st.divider()
        st.markdown("**📋 Model Performance**")
        for k, v in [("ROC-AUC","0.9533"),
                      ("Sensitivity","0.9969"),
                      ("Specificity","0.8105"),
                      ("PR-AUC","0.7410"),
                      ("Brier","0.0577"),
                      ("N patients","606,824")]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.8rem;padding:2px 0;">'
                f'<span style="color:#718096">{k}</span>'
                f'<span style="color:#63b3ed;font-weight:700">{v}</span>'
                f'</div>',
                unsafe_allow_html=True)

        st.divider()
        if st.button("🗑️ Clear History", use_container_width=True):
            clear_history(pid)
            st.success("History cleared")

    # ── Tabs ──────────────────────────────────────────────
    tabs = st.tabs([
        "📋 Input & Predict",
        "📊 Results",
        "📈 History & Forecast",
        "🧩 SHAP",
        "📄 PDF Report",
        "📚 Reference",
    ])

    # =====================================================
    # TAB 1 — INPUT FORM
    # =====================================================
    with tabs[0]:
        st.markdown(
            '<div class="sec-header">🔬 Enter Patient Clinical Values</div>',
            unsafe_allow_html=True)

        st.info(
            "Fill in the patient values from the follow-up visit "
            "and click **Predict Graft Failure Risk** at the bottom.")

        with st.form("patient_form"):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Numeric Features**")

                creat_miss = st.checkbox("Creatinine value unknown")
                if creat_miss:
                    creat_val = np.nan
                    st.caption("Will use median imputation in model")
                else:
                    creat_val = st.number_input(
                        "Serum Creatinine (mg/dL)",
                        min_value=0.1, max_value=20.0,
                        value=1.2, step=0.05,
                        help="Normal post-transplant: 0.7–1.3 mg/dL. "
                             "Threshold: 1.095 (below = high risk)")

                func_miss = st.checkbox("Functional Status unknown")
                if func_miss:
                    func_val = np.nan
                    st.caption("Will use median imputation in model")
                else:
                    func_val = st.number_input(
                        "Functional Status Code (FUNC_STAT)",
                        min_value=1.0, max_value=4100.0,
                        value=2070.0, step=10.0,
                        help="Karnofsky x10. Normal: 2070–2100. "
                             "Threshold: 1845.55")

                pra1_miss = st.checkbox("PRA1 not measured")
                if pra1_miss:
                    pra1_val = None
                    st.caption("PRA missingness is itself a risk signal (MNAR)")
                else:
                    pra1_val = st.slider(
                        "Panel Reactive Antibody % (PRA1)",
                        min_value=0.0, max_value=100.0,
                        value=0.0, step=1.0,
                        help="Any PRA >= 1% elevates failure risk to 20-26%")

            with c2:
                st.markdown("**Categorical Features**")

                rej_labels = {
                    "3":   "3 — No rejection episode",
                    "998": "998 — Unknown / Not reported",
                    "1":   "1 — One acute rejection episode",
                    "2":   "2 — Multiple rejection episodes",
                }
                acute_rej = st.selectbox(
                    "Acute Rejection Episodes",
                    options=["3", "998", "1", "2"],
                    format_func=lambda x: rej_labels[x],
                    help="Strongest predictor. Cat 1/2 = 55-59% historical failure")

                diab_labels = {
                    "N": "N — No diabetes",
                    "Y": "Y — Diabetes present",
                    "U": "U — Unknown status",
                }
                diabetes = st.selectbox(
                    "Diabetes During Follow-up",
                    options=["N", "Y", "U"],
                    format_func=lambda x: diab_labels[x],
                    help="Unknown (U) = 24.5% historical failure")

                hla_labels = {
                    "DNA":            "DNA typing",
                    "Serology":       "Serology",
                    "Other,Specify":  "Other / Specify",
                    "Serology; DNA":  "Serology + DNA (highest risk)",
                }
                hla = st.selectbox(
                    "HLA Typing Method (Class I)",
                    options=["DNA", "Serology",
                              "Other,Specify", "Serology; DNA"],
                    format_func=lambda x: hla_labels[x],
                    help="Serology+DNA = highest risk category (17.2%)")

                st.markdown("**Historical failure rates (reference)**")
                rej_rates = {"3": 11.9, "998": 29.8, "1": 59.1, "2": 55.4}
                chosen_rate = rej_rates[acute_rej]
                if chosen_rate > TRIGGER:
                    badge_color = "badge-red"
                elif chosen_rate > BASELINE:
                    badge_color = "badge-yellow"
                else:
                    badge_color = "badge-green"
                st.markdown(
                    f'<span class="{badge_color}">'
                    f'ACUTE_REJ_EPI = {acute_rej}: '
                    f'{chosen_rate:.1f}% historical failure</span>',
                    unsafe_allow_html=True)

            st.divider()
            submitted = st.form_submit_button(
                "🔍  Predict Graft Failure Risk",
                use_container_width=True,
                type="primary"
            )

        # Store in session state so other tabs can use it
        if submitted:
            inputs = {
                "CREAT":              creat_val,
                "FUNC_STAT":          func_val,
                "PRA1":               pra1_val,
                "ACUTE_REJ_EPI":      acute_rej,
                "DIABETES_DURING":    diabetes,
                "HLA_TY_METH_CLASS1": hla,
            }
            risk_pct, method, X_proc = predict(inputs, model, prep)
            contrib = shap_contributions(inputs)
            level, color_hex, emoji = get_risk_level(risk_pct)

            st.session_state["inputs"]   = inputs
            st.session_state["risk_pct"] = risk_pct
            st.session_state["method"]   = method
            st.session_state["X_proc"]   = X_proc
            st.session_state["contrib"]  = contrib
            st.session_state["level"]    = level
            st.session_state["pid"]      = pid
            st.session_state["pname"]    = pname
            st.session_state["vdate"]    = vdate
            st.session_state["role"]     = role

            # Save to history
            record = {
                "date":     str(vdate),
                "risk_pct": risk_pct,
                "category": level,
                "method":   method,
                "inputs": {
                    k: (float(v) if isinstance(v, (int, float))
                        and v is not None
                        and not (isinstance(v, float) and np.isnan(v))
                        else None)
                    for k, v in inputs.items()
                },
            }
            save_history(pid, record)
            st.success(
                f"Prediction complete for patient **{pid}**  |  "
                f"Risk = **{risk_pct:.1f}%** ({level})  |  "
                f"Method: {method}  |  "
                f"Go to **Results** tab to see full analysis.")

    # =====================================================
    # TAB 2 — RESULTS
    # =====================================================
    with tabs[1]:
        if "risk_pct" not in st.session_state:
            st.info(
                "No prediction yet. "
                "Go to **Input & Predict** tab, fill the form, "
                "and click Predict.")
        else:
            risk_pct = st.session_state["risk_pct"]
            inputs   = st.session_state["inputs"]
            contrib  = st.session_state["contrib"]
            level    = st.session_state["level"]
            method   = st.session_state["method"]
            role     = st.session_state.get("role", "Doctor")

            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            history_now = load_history(
                st.session_state.get("pid", "PT-001"))
            prev = (history_now[-2]["risk_pct"]
                    if len(history_now) >= 2
                    else risk_pct)
            diff = risk_pct - prev

            if level == "HIGH":
                rv_clr = "kpi-red"
            elif level == "MODERATE":
                rv_clr = "kpi-yellow"
            else:
                rv_clr = "kpi-green"

            if diff > 2:
                diff_clr = "kpi-red"
            elif diff < -2:
                diff_clr = "kpi-green"
            else:
                diff_clr = "kpi-blue"

            diff_sym = "UP" if diff > 0 else ("DOWN" if diff < 0 else "STABLE")

            for col_w, val, lbl, clr in [
                (k1, f"{risk_pct:.1f}%",        "Predicted Risk",  rv_clr),
                (k2, level,                      "Risk Category",   rv_clr),
                (k3, f"{diff_sym} {abs(diff):.1f}%", "vs Last Visit", diff_clr),
                (k4, f"{BASELINE:.1f}%",         "Cohort Baseline", "kpi-blue"),
            ]:
                with col_w:
                    st.markdown(
                        f'<div class="kpi-card">'
                        f'<div class="kpi-val {clr}">{val}</div>'
                        f'<div class="kpi-label">{lbl}</div>'
                        f'</div>',
                        unsafe_allow_html=True)

            # Gauge
            st.markdown(
                '<div class="sec-header">📊 Risk Gauge</div>',
                unsafe_allow_html=True)
            gf = chart_gauge(risk_pct)
            st.pyplot(gf, use_container_width=True)
            plt.close()

            # Alert
            if level == "HIGH":
                alert_cls = "alert-high"
            elif level == "MODERATE":
                alert_cls = "alert-moderate"
            else:
                alert_cls = "alert-low"

            if "Doctor" in role or "Nurse" in role:
                if level == "HIGH":
                    msg = (
                        f"<b>Immediate review required.</b> "
                        f"Predicted graft failure risk {risk_pct:.1f}% "
                        f"exceeds high-risk threshold ({HIGH_THRESH:.0f}%). "
                        f"Primary driver: CREAT = "
                        f"{inputs.get('CREAT', 'N/A')} mg/dL "
                        f"({'BELOW' if float(inputs.get('CREAT',2)) < 1.095 else 'above'}"
                        f" threshold 1.095)."
                    )
                elif level == "MODERATE":
                    msg = (
                        f"<b>Elevated risk.</b> "
                        f"Risk {risk_pct:.1f}% exceeds trigger ({TRIGGER:.1f}%). "
                        f"Schedule follow-up and monitor creatinine closely."
                    )
                else:
                    msg = (
                        f"<b>Within acceptable range.</b> "
                        f"Risk {risk_pct:.1f}% below trigger. "
                        f"Continue standard monitoring."
                    )
            else:
                if level == "HIGH":
                    msg = "🔴 <b>Your kidney needs urgent attention.</b> Please contact your transplant team today."
                elif level == "MODERATE":
                    msg = "🟡 <b>Some warning signs present.</b> Schedule a check-up soon."
                else:
                    msg = "🟢 <b>Your kidney appears stable.</b> Continue your medications and attend regular check-ups."

            st.markdown(
                f'<div class="{alert_cls}">{msg}</div>',
                unsafe_allow_html=True)

            # Threshold alerts
            st.markdown(
                '<div class="sec-header">⚠️ Threshold Alerts</div>',
                unsafe_allow_html=True)

            creat_v = float(inputs.get("CREAT", 2.0))
            func_v  = float(inputs.get("FUNC_STAT", 2070))
            pra1_v  = float(inputs.get("PRA1") or 0)

            alerts = [
                ("CREAT",     f"{creat_v:.2f} mg/dL", "1.095",
                 creat_v < 1.095, "BELOW threshold — HIGH RISK",
                 "OK: above threshold"),
                ("FUNC_STAT", f"{func_v:.0f}",         "1845.55",
                 func_v < 1845.55, "BELOW threshold",
                 "OK: normal range"),
                ("PRA1",      f"{pra1_v:.1f}%",        "1.0",
                 pra1_v >= 1.0, "ELEVATED — sensitization present",
                 "OK: no sensitization"),
            ]
            for feat, val, tau, flag, warn_msg, ok_msg in alerts:
                if flag:
                    badge = "badge-red"
                    status = warn_msg
                else:
                    badge = "badge-green"
                    status = ok_msg
                st.markdown(
                    f"**{feat}**: {val}  (threshold = {tau})  "
                    f'<span class="{badge}">{status}</span>',
                    unsafe_allow_html=True)

            # Risk curves
            st.markdown(
                '<div class="sec-header">📉 Patient on Risk Curves</div>',
                unsafe_allow_html=True)
            rc = chart_risk_curves(inputs)
            st.pyplot(rc, use_container_width=True)
            plt.close()

            # Patient summary table
            st.markdown(
                '<div class="sec-header">📋 Input Summary</div>',
                unsafe_allow_html=True)
            summary = pd.DataFrame([{
                "Feature":   k,
                "Value":     str(v) if v is not None else "Not measured",
                "Threshold": {
                    "CREAT":     "1.095 mg/dL",
                    "FUNC_STAT": "1845.55",
                    "PRA1":      "1.0%",
                }.get(k, "N/A"),
            } for k, v in inputs.items()])
            st.dataframe(summary, use_container_width=True,
                          hide_index=True)

            st.caption(
                f"Prediction method: **{method}**  |  "
                f"{'Real trained model (Gradient Boosting)' if method == 'model' else 'Evidence-based fallback (model not loaded)'}")

    # =====================================================
    # TAB 3 — HISTORY & FORECAST
    # =====================================================
    with tabs[2]:
        history = load_history(
            st.session_state.get("pid", pid))
        if len(history) == 0:
            st.info(
                "No history yet. Submit a prediction in "
                "Tab 1 to start tracking.")
        else:
            latest = history[-1]["risk_pct"]
            peak   = max(h["risk_pct"] for h in history)
            first  = history[0]["risk_pct"]
            n_high = sum(1 for h in history
                          if h["category"] == "HIGH")
            trend  = latest - first

            k1, k2, k3, k4, k5 = st.columns(5)
            for col_w, val, lbl, clr in [
                (k1, f"{latest:.1f}%",
                 "Latest Risk",
                 "kpi-red" if latest >= HIGH_THRESH
                 else "kpi-yellow" if latest >= TRIGGER
                 else "kpi-green"),
                (k2, f"{peak:.1f}%", "Peak Risk", "kpi-red"),
                (k3,
                 f"{'UP' if trend>0 else 'DOWN'} {abs(trend):.1f}%",
                 "Overall Trend",
                 "kpi-red" if trend > 3
                 else "kpi-green" if trend < -3
                 else "kpi-blue"),
                (k4, str(len(history)), "Total Visits", "kpi-blue"),
                (k5, str(n_high),       "High Risk Visits",
                 "kpi-red" if n_high > 0 else "kpi-green"),
            ]:
                with col_w:
                    st.markdown(
                        f'<div class="kpi-card">'
                        f'<div class="kpi-val {clr}">{val}</div>'
                        f'<div class="kpi-label">{lbl}</div>'
                        f'</div>',
                        unsafe_allow_html=True)

            st.markdown(
                '<div class="sec-header">📈 Risk History</div>',
                unsafe_allow_html=True)
            hf = chart_history(history)
            if hf:
                st.pyplot(hf, use_container_width=True)
                plt.close()

            st.markdown(
                '<div class="sec-header">🔮 Risk Forecast</div>',
                unsafe_allow_html=True)
            n_fc = st.slider("Visits to forecast", 2, 8, 4)
            if len(history) >= 3:
                ff = chart_forecast(history, n_fc)
                if ff:
                    st.pyplot(ff, use_container_width=True)
                    plt.close()
                risks_arr = np.array(
                    [h["risk_pct"] for h in history])
                slope = np.polyfit(
                    np.arange(len(risks_arr)), risks_arr, 1)[0]
                if slope > 1.5:
                    st.markdown(
                        f'<div class="alert-high">'
                        f'<b>Rising trend:</b> +{slope:.2f}% per visit. '
                        f'Consider clinical review.</div>',
                        unsafe_allow_html=True)
                elif slope < -1.5:
                    st.markdown(
                        f'<div class="alert-low">'
                        f'<b>Improving trend:</b> {slope:.2f}% per visit. '
                        f'Treatment appears effective.</div>',
                        unsafe_allow_html=True)
            else:
                st.info(
                    f"Need 3+ visits for forecast. "
                    f"Have {len(history)} visit(s).")

            st.markdown(
                '<div class="sec-header">📋 Visit History Table</div>',
                unsafe_allow_html=True)
            tbl = pd.DataFrame([{
                "Date":     h["date"],
                "Risk %":   f"{h['risk_pct']:.1f}%",
                "Category": h["category"],
                "CREAT":    h["inputs"].get("CREAT", "-"),
                "PRA1":     h["inputs"].get("PRA1", "-"),
                "Rejection":h["inputs"].get("ACUTE_REJ_EPI", "-"),
                "Method":   h.get("method", "-"),
            } for h in reversed(history)])
            st.dataframe(tbl, use_container_width=True,
                          hide_index=True)

    # =====================================================
    # TAB 4 — SHAP
    # =====================================================
    with tabs[3]:
        if "contrib" not in st.session_state:
            st.info("Run a prediction in Tab 1 first.")
        else:
            contrib  = st.session_state["contrib"]
            risk_pct = st.session_state["risk_pct"]
            X_proc   = st.session_state.get("X_proc")

            st.markdown(
                '<div class="sec-header">🧩 Feature Contributions</div>',
                unsafe_allow_html=True)

            # Try real SHAP first
            real_shap_vals = None
            if shap_ok and explainer and X_proc is not None:
                try:
                    sv = explainer.shap_values(X_proc)
                    real_shap_vals = (sv[1][0]
                                      if isinstance(sv, list)
                                      else sv[0])
                    st.success(
                        "Real SHAP values computed from "
                        "trained Gradient Boosting model.")
                except Exception:
                    pass

            if real_shap_vals is not None:
                # Use real SHAP feature names from model
                feat_names = [
                    "CREAT", "FUNC_STAT", "PRA1",
                    "ACUTE_REJ_EPI_2", "ACUTE_REJ_EPI_3",
                    "ACUTE_REJ_EPI_998",
                    "HLA_TY_METH_CLASS1_Other,Specify",
                    "HLA_TY_METH_CLASS1_Serology",
                    "HLA_TY_METH_CLASS1_Serology; DNA",
                    "DIABETES_DURING_U", "DIABETES_DURING_Y",
                    "PRA1_missing_flag",
                ]
                pairs  = sorted(
                    zip(feat_names[:len(real_shap_vals)],
                        real_shap_vals),
                    key=lambda x: abs(x[1]), reverse=True
                )[:10]
                labels = [p[0] for p in pairs]
                values = [float(p[1]) for p in pairs]
                contrib_to_plot = dict(zip(labels, values))
            else:
                contrib_to_plot = contrib
                st.info(
                    "Using evidence-based contributions "
                    "(model SHAP not available).")

            sw = chart_contributions(contrib_to_plot, risk_pct)
            st.pyplot(sw, use_container_width=True)
            plt.close()

            st.markdown(
                '<div class="sec-header">📊 Contribution Table</div>',
                unsafe_allow_html=True)
            exp_rows = []
            for feat, val in sorted(
                contrib_to_plot.items(),
                key=lambda x: abs(x[1]), reverse=True
            ):
                exp_rows.append({
                    "Feature":      feat,
                    "Contribution": f"{'+' if val>=0 else ''}{val:.2f} pp",
                    "Direction":    "Raises risk" if val > 0 else "Lowers risk",
                    "Magnitude":    "High" if abs(val)>5 else "Medium" if abs(val)>1 else "Low",
                })
            st.dataframe(pd.DataFrame(exp_rows),
                          use_container_width=True,
                          hide_index=True)

            st.markdown("""
            <div style="color:#718096;font-size:0.8rem;margin-top:0.8rem;">
            <b>How to read:</b> Each bar shows how much that feature
            pushes the prediction above (red) or below (green) the
            baseline rate of 13.42%.
            Sum of all contributions + 13.42% = final predicted risk.
            </div>
            """, unsafe_allow_html=True)

    # =====================================================
    # TAB 5 — PDF REPORT
    # =====================================================
    with tabs[4]:
        st.markdown(
            '<div class="sec-header">📄 Download Patient Report</div>',
            unsafe_allow_html=True)

        if "risk_pct" not in st.session_state:
            st.info("Run a prediction in Tab 1 first.")
        else:
            risk_pct = st.session_state["risk_pct"]
            inputs   = st.session_state["inputs"]
            contrib  = st.session_state["contrib"]
            pid_s    = st.session_state.get("pid", pid)
            pname_s  = st.session_state.get("pname", "")
            vdate_s  = st.session_state.get("vdate", datetime.today())
            history  = load_history(pid_s)

            st.markdown(
                f"**Patient:** {pid_s}  |  "
                f"**Visit:** {vdate_s}  |  "
                f"**Risk:** {risk_pct:.1f}%")
            st.markdown(
                "PDF contains: Summary page, "
                "History dashboard, Risk forecast, "
                "Threshold curves.")

            if st.button(
                "📥 Generate PDF Report",
                use_container_width=True,
                type="primary"
            ):
                with st.spinner("Building PDF..."):
                    pdf_buf = make_pdf(
                        pid_s, pname_s, str(vdate_s),
                        inputs, risk_pct, contrib, history
                    )
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_buf,
                    file_name=(
                        f"GraftRisk_{pid_s}_{vdate_s}.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("PDF ready.")

    # =====================================================
    # TAB 6 — REFERENCE
    # =====================================================
    with tabs[5]:
        c1r, c2r = st.columns(2)
        with c1r:
            st.markdown("#### Numeric Thresholds")
            st.dataframe(pd.DataFrame([
                {"Feature":"CREAT",    "Threshold":"1.095 mg/dL",
                 "Direction":"INVERSE", "Below":"0.02%",
                 "Above":"16.17%",     "Lift":"696x"},
                {"Feature":"FUNC_STAT","Threshold":"1845.55",
                 "Direction":"INVERSE*","Below":"9.21%",
                 "Above":"13.68%",     "Lift":"1.49x"},
                {"Feature":"PRA1",     "Threshold":"1.0%",
                 "Direction":"DIRECT",  "Below":"18.84%",
                 "Above":"20-26%",     "Lift":"Monotonic"},
            ]), use_container_width=True, hide_index=True)
            st.caption(
                "* FUNC_STAT INVERSE = registry carry-forward artefact")

            st.markdown("#### Risk Zones")
            st.dataframe(pd.DataFrame([
                {"Zone":"LOW",      "Range":f"< {TRIGGER:.1f}%",       "Action":"Standard monitoring"},
                {"Zone":"MODERATE", "Range":f"{TRIGGER:.1f}-{HIGH_THRESH:.0f}%","Action":"Increased surveillance"},
                {"Zone":"HIGH",     "Range":f"> {HIGH_THRESH:.0f}%",    "Action":"Immediate clinical review"},
            ]), use_container_width=True, hide_index=True)

        with c2r:
            st.markdown("#### Categorical Risk Rates")
            for feat, rates, labels in [
                ("ACUTE_REJ_EPI",
                 {"3":11.9,"998":29.8,"1":59.1,"2":55.4},
                 {"3":"No rejection","998":"Unknown","1":"One episode","2":"Multiple"}),
                ("DIABETES_DURING",
                 {"N":13.3,"Y":10.1,"U":24.5},
                 {"N":"No diabetes","Y":"Present","U":"Unknown"}),
                ("HLA_TY_METH_CLASS1",
                 {"DNA":11.5,"Serology":15.4,"Other,Specify":16.2,"Serology; DNA":17.2},
                 {"DNA":"DNA","Serology":"Serology","Other,Specify":"Other","Serology; DNA":"Sero+DNA"}),
            ]:
                st.markdown(f"**{feat}**")
                st.dataframe(pd.DataFrame([{
                    "Category":   labels.get(k, k),
                    "Failure %":  f"{v:.1f}%",
                    "vs Baseline":f"{v-BASELINE:+.1f}pp",
                    "Zone": "HIGH" if v >= HIGH_THRESH
                            else "MODERATE" if v >= TRIGGER
                            else "LOW",
                } for k, v in sorted(
                    rates.items(), key=lambda x: -x[1])
                ]), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="footer">
        🫘 Kidney Graft Risk Monitor  |
        UNOS STAR N=606,824  |  Gradient Boosting AUC=0.953  |
        For research and clinical decision support only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()