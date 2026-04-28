"""
shared.py
Shared styling, UI components, and inference utilities.

Design system
─────────────
Palette  : warm off-white bg, navy ink, amber accent
Type     : Fraunces (headings) · Inter (body / numbers)
Cards    : white surface, 1 px border, soft shadow — no blur
Non-interactive decorative elements: .tag (pointer-events: none)
Interactive elements: only real st.button / st.form controls
"""

from pathlib import Path
import sys
import json
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PREPROCESS_DIR = ROOT / "preprocess"
DATA_PATH = ROOT / "data" / "insurance.csv"
SAVED_DIR = ROOT / "saved_models"

for _p in (MODELS_DIR, PREPROCESS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from preprocess import get_classifier_data_logistic, get_regressor_data_linear


# ── Design tokens ─────────────────────────────────────────────────────────────

PALETTE = {
    "bg":         "#f7f5ef",
    "surface":    "#ffffff",
    "surface_2":  "#f0ede6",
    "border":     "#e2ddd5",
    "ink":        "#1c1f26",
    "ink_2":      "#5a6070",
    "ink_3":      "#9aa0ab",
    "accent":     "#c96c1a",
    "accent_bg":  "#fef4e8",
    "navy":       "#0f1a2e",
    "green":      "#1a7a40",
    "red":        "#b91c1c",
}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.stApp {
    background-color: {bg};
    color: {ink};
}
h1, h2, h3, h4 {{
    font-family: 'Fraunces', Georgia, serif !important;
    color: {ink};
    letter-spacing: -0.02em;
}}

/* ── Layout ── */
.block-container {{
    max-width: 1200px;
    padding: 1.5rem 2rem 4rem;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {navy} !important;
}}
[data-testid="stSidebar"] * {{
    color: #d8d4c8 !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {{
    color: #f0ece2 !important;
    font-family: 'Fraunces', Georgia, serif !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: #a8a49a !important;
    font-size: 0.82rem;
}}
/* Radio buttons in sidebar */
[data-testid="stSidebar"] .stRadio label {{
    color: #f0ece2 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{
    gap: 0.1rem;
}}

/* ── Page header ── */
.page-header {{
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid {border};
    margin-bottom: 1.75rem;
}}
.page-kicker {{
    display: block;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {accent};
    margin-bottom: 0.35rem;
}}
.page-header h1 {{
    font-size: 1.85rem;
    margin: 0 0 0.35rem;
    line-height: 1.15;
    font-weight: 700;
}}
.page-subtitle {{
    color: {ink_2};
    font-size: 0.95rem;
    max-width: 680px;
    margin: 0;
    line-height: 1.5;
}}

/* ── Generic card ── */
.card {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 1.25rem 1.375rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    margin-bottom: 1rem;
}}
.card-label {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {ink_3};
    margin-bottom: 0.5rem;
}}
.card-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1rem;
    font-weight: 700;
    margin: 0 0 0.5rem;
    color: {ink};
}}
.card-body {{
    color: {ink_2};
    font-size: 0.88rem;
    line-height: 1.6;
}}
.card-body strong {{ color: {ink}; }}

/* ── Metric tiles (row of summary numbers) ── */
.metric-tile {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 0.9rem 1rem;
    height: 100%;
}}
.metric-tile-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {ink_3};
    margin-bottom: 0.35rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.metric-tile-value {{
    font-size: clamp(1.2rem, 2.2vw, 1.65rem);
    font-weight: 700;
    color: {ink};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
}}
.metric-tile-note {{
    font-size: 0.76rem;
    color: {ink_3};
    margin-top: 0.2rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

/* ── Result panel (predictor primary output) ── */
.result-panel {{
    background: {navy};
    border-radius: 16px;
    padding: 1.75rem 2rem;
    color: white;
    margin-bottom: 0.875rem;
}}
.result-label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45);
    margin-bottom: 0.4rem;
}}
.result-amount {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: clamp(2.2rem, 4.5vw, 3.2rem);
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.0;
    white-space: nowrap;
}}
.result-per-year {{
    font-size: 0.82rem;
    color: rgba(255,255,255,0.45);
    margin-top: 0.25rem;
    margin-bottom: 1.25rem;
}}
.range-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.4rem;
}}
.range-bounds {{
    font-size: 0.82rem;
    color: rgba(255,255,255,0.55);
    font-variant-numeric: tabular-nums;
}}
.range-track {{
    position: relative;
    height: 5px;
    background: rgba(255,255,255,0.16);
    border-radius: 99px;
    margin-bottom: 0.5rem;
}}
.range-fill {{
    position: absolute;
    left: 0; top: 0; bottom: 0;
    border-radius: 99px;
    background: rgba(255,255,255,0.28);
}}
.range-pin {{
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 13px;
    height: 13px;
    background: #ffffff;
    border: 2.5px solid {navy};
    border-radius: 50%;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.5);
}}
.range-caption {{
    font-size: 0.75rem;
    color: rgba(255,255,255,0.38);
}}

/* ── Routing/info callout ── */
.routing-card {{
    background: {surface_2};
    border: 1px solid {border};
    border-left: 3px solid {accent};
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.875rem;
}}
.routing-label {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {accent};
    margin-bottom: 0.35rem;
}}
.routing-body {{
    font-size: 0.86rem;
    color: {ink_2};
    line-height: 1.55;
    margin: 0;
}}
.routing-body strong {{ color: {ink}; }}

/* ── Empty state ── */
.empty-state {{
    background: {surface_2};
    border: 1.5px dashed {border};
    border-radius: 14px;
    padding: 2.75rem 2rem;
    text-align: center;
}}
.empty-icon {{ font-size: 1.75rem; margin-bottom: 0.65rem; }}
.empty-title {{
    font-size: 0.95rem;
    font-weight: 600;
    color: {ink_2};
    margin-bottom: 0.3rem;
}}
.empty-body {{
    font-size: 0.85rem;
    color: {ink_3};
    line-height: 1.5;
}}

/* ── Non-interactive decorative tags ── */
.tag-row {{ display: flex; gap: 0.45rem; flex-wrap: wrap; margin-top: 0.5rem; }}
.tag {{
    display: inline-block;
    padding: 0.25rem 0.6rem;
    background: {accent_bg};
    color: {accent};
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    pointer-events: none;
    user-select: none;
    cursor: default;
}}

/* ── Section subheading ── */
.section-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ink_3};
    margin-bottom: 0.6rem;
    display: block;
}}

/* ── Form controls ── */
.stSlider [data-testid="stSliderThumb"] {{ background: {navy} !important; }}
.stRadio label {{ font-size: 0.875rem !important; font-weight: 500 !important; }}
.stSelectbox label, .stSlider > label {{
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: {ink_2} !important;
    margin-bottom: 0.1rem !important;
}}
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] label p,
[data-testid="stAppViewContainer"] .stSlider label,
[data-testid="stAppViewContainer"] .stSlider label p,
[data-testid="stAppViewContainer"] label[data-testid="stWidgetLabel"] {{
    background: transparent !important;
    background-color: transparent !important;
}}
[data-testid="stAppViewContainer"] .stNumberInput label,
[data-testid="stAppViewContainer"] .stNumberInput label p,
[data-testid="stAppViewContainer"] .stSelectbox label,
[data-testid="stAppViewContainer"] .stSelectbox label p,
[data-testid="stAppViewContainer"] .stRadio label,
[data-testid="stAppViewContainer"] .stRadio label p,
[data-testid="stAppViewContainer"] label[data-testid="stWidgetLabel"],
[data-testid="stAppViewContainer"] label[data-testid="stWidgetLabel"] p,
[data-testid="stAppViewContainer"] .stRadio div[role="radiogroup"] label,
[data-testid="stAppViewContainer"] .stRadio div[role="radiogroup"] label p {{
    color: {ink} !important;
    -webkit-text-fill-color: {ink} !important;
    background: transparent !important;
    background-color: transparent !important;
}}
/* Sidebar overrides — must come AFTER the broad label rule to win */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] label[data-testid="stWidgetLabel"],
[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio label p {{
    color: #f0ece2 !important;
    -webkit-text-fill-color: #f0ece2 !important;
    background: transparent !important;
    background-color: transparent !important;
}}
[data-testid="stAppViewContainer"] .stCaption {{
    color: {ink_2} !important;
}}
/* Dropdown — outline style, no dark fill */
.stSelectbox div[data-baseweb="select"] > div {{
    background-color: {surface} !important;
    border: 1.5px solid {border} !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}}
.stSelectbox div[data-baseweb="select"] *,
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] svg {{
    color: {ink} !important;
    background-color: transparent !important;
}}
/* Dropdown menu popup */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul {{
    background-color: {surface} !important;
    border: 1px solid {border} !important;
    border-radius: 8px !important;
}}
[data-baseweb="popover"] [role="option"] {{
    color: {ink} !important;
    background-color: {surface} !important;
}}
[data-baseweb="popover"] [role="option"]:hover {{
    background-color: {accent_bg} !important;
}}
.stSlider [data-testid="stTickBar"],
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {{
    color: {ink} !important;
}}

/* EDA overview metric row */
[data-testid="stMetric"],
[data-testid="stMetric"] * ,
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {{
    color: {ink} !important;
    -webkit-text-fill-color: {ink} !important;
}}

/* ── Primary CTA button ── */
.stFormSubmitButton > button,
.stButton > button {{
    background: {navy} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.25rem !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, opacity 0.15s ease !important;
    box-shadow: 0 10px 24px rgba(15,26,46,0.18) !important;
    opacity: 1 !important;
    cursor: pointer !important;
}}
.stFormSubmitButton > button:hover,
.stButton > button:hover {{
    opacity: 0.92 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 14px 28px rgba(15,26,46,0.22) !important;
}}
.stFormSubmitButton > button *,
.stButton > button *,
.stFormSubmitButton > button div,
.stButton > button div,
.stFormSubmitButton > button p,
.stButton > button p,
.stFormSubmitButton > button span,
.stButton > button span {{
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}}
.stFormSubmitButton > button:disabled,
.stButton > button:disabled {{
    background: {navy} !important;
    color: #ffffff !important;
    opacity: 0.72 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
}}

/* Tabs should read like real controls */
button[role="tab"] {{
    color: {ink} !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    border-bottom-color: transparent !important;
}}
button[role="tab"] p,
button[role="tab"] div,
button[role="tab"] span {{
    color: {ink} !important;
    -webkit-text-fill-color: {ink} !important;
}}
button[role="tab"][aria-selected="true"] {{
    color: {navy} !important;
    border-bottom: 2px solid {navy} !important;
    background: rgba(255,255,255,0.5) !important;
}}
button[role="tab"][aria-selected="true"] p,
button[role="tab"][aria-selected="true"] div,
button[role="tab"][aria-selected="true"] span {{
    color: {navy} !important;
    -webkit-text-fill-color: {navy} !important;
}}

/* ── Divider ── */
hr, [data-testid="stDivider"] {{
    border-color: {border} !important;
    margin: 1.75rem 0 !important;
}}

/* ── Sidebar nav labels — high-specificity override ── */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {{
    color: #f0ece2 !important;
    -webkit-text-fill-color: #f0ece2 !important;
}}

/* Sidebar multiselect labels */
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stMultiSelect label p,
[data-testid="stSidebar"] .stMultiSelect label span,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stMultiSelect label p {{
    color: #f0ece2 !important;
    -webkit-text-fill-color: #f0ece2 !important;
}}

/* Sidebar multiselect — dark text on white dropdown */
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stMultiSelect label p {{
    color: #f0ece2 !important;
    -webkit-text-fill-color: #f0ece2 !important;
}}
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div {{
    background-color: #ffffff !important;
    border: 1.5px solid {border} !important;
}}
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] span,
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] div,
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"] span,
[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"] div {{
    color: {ink} !important;
    -webkit-text-fill-color: {ink} !important;
}}
[data-testid="stSidebar"] .stMultiSelect svg {{
    fill: {ink} !important;
    color: {ink} !important;
}}
"""


def inject_global_styles():
    """Apply the design system across all pages."""
    css = _CSS
    for key, value in PALETTE.items():
        css = css.replace("{" + key + "}", value)
    css = css.replace("{{", "{").replace("}}", "}")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _html(block: str) -> str:
    """Normalize custom HTML so Streamlit does not interpret it as a code block."""
    return dedent(block).strip()


# ── UI components ─────────────────────────────────────────────────────────────

def page_header(kicker: str, title: str, subtitle: str = ""):
    """Compact non-card page header with kicker, h1, and optional subtitle."""
    sub_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="page-header">
            <span class="page-kicker">{kicker}</span>
            <h1>{title}</h1>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str, note: str = ""):
    """Single metric tile — value never wraps."""
    note_html = (
        f'<div style="font-size:0.76rem; color:{PALETTE["ink_3"]}; '
        f'margin-top:0.2rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{note}</div>'
        if note else ""
    )
    st.markdown(
        _html(f"""
        <div style="
            background:{PALETTE['surface']};
            border:1px solid {PALETTE['border']};
            border-radius:12px;
            padding:0.9rem 1rem;
            height:100%;
            box-shadow:0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
        ">
            <div style="
                font-size:0.68rem;
                font-weight:700;
                letter-spacing:0.1em;
                text-transform:uppercase;
                color:{PALETTE['ink_3']};
                margin-bottom:0.35rem;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
            ">{label}</div>
            <div style="
                font-size:clamp(1.2rem, 2.2vw, 1.65rem);
                font-weight:700;
                color:{PALETTE['ink']};
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
                font-variant-numeric:tabular-nums;
                line-height:1.15;
            ">{value}</div>
            {note_html}
        </div>
        """),
        unsafe_allow_html=True,
    )


def result_panel(estimate: float, q10: float, q90: float, q50: float):
    """Primary prediction output: big number + range bar."""
    span = max(q90 - q10, 1)
    pin_pct = round(max(5.0, min(95.0, (estimate - q10) / span * 100)), 1)
    st.markdown(
        _html(f"""
        <div style="
            background:{PALETTE['navy']};
            border-radius:16px;
            padding:1.75rem 2rem;
            color:#ffffff;
            margin-bottom:0.875rem;
            box-shadow:0 10px 28px rgba(15,26,46,0.18);
        ">
            <div style="
                font-size:0.68rem;
                font-weight:600;
                letter-spacing:0.12em;
                text-transform:uppercase;
                color:rgba(255,255,255,0.55);
                margin-bottom:0.4rem;
            ">Predicted annual cost</div>
            <div style="
                font-family:'Fraunces', Georgia, serif;
                font-size:clamp(2.2rem, 4.5vw, 3.2rem);
                font-weight:700;
                color:#ffffff;
                letter-spacing:-0.03em;
                line-height:1.0;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
            ">${estimate:,.0f}</div>
            <div style="
                font-size:0.82rem;
                color:rgba(255,255,255,0.58);
                margin-top:0.25rem;
                margin-bottom:1.25rem;
            ">per year · 80% uncertainty interval</div>
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:baseline;
                margin-bottom:0.4rem;
            ">
                <span style="font-size:0.82rem; color:rgba(255,255,255,0.7); font-variant-numeric:tabular-nums;">${q10:,.0f}</span>
                <span style="font-size:0.82rem; color:rgba(255,255,255,0.7); font-variant-numeric:tabular-nums;">${q90:,.0f}</span>
            </div>
            <div style="
                position:relative;
                height:5px;
                background:rgba(255,255,255,0.16);
                border-radius:99px;
                margin-bottom:0.5rem;
            ">
                <div style="
                    position:absolute;
                    left:0; top:0; bottom:0;
                    width:100%;
                    border-radius:99px;
                    background:rgba(255,255,255,0.28);
                "></div>
                <div style="
                    position:absolute;
                    left:{pin_pct}%;
                    top:50%;
                    transform:translate(-50%, -50%);
                    width:13px;
                    height:13px;
                    background:#ffffff;
                    border:2.5px solid {PALETTE['navy']};
                    border-radius:50%;
                    box-shadow:0 0 0 2px rgba(255,255,255,0.5);
                "></div>
            </div>
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.5);">
                Quantile regression interval &nbsp;·&nbsp; median estimate ${q50:,.0f}
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def routing_card(segment: str, explanation: str):
    """Routing decision callout with amber left-border."""
    outer = (
        f"background:{PALETTE['surface_2']}; border:1px solid {PALETTE['border']}; "
        f"border-left:3px solid {PALETTE['accent']}; border-radius:10px; "
        f"padding:0.9rem 1rem; margin-bottom:0.875rem;"
    )
    label_style = (
        f"font-size:0.68rem; font-weight:700; text-transform:uppercase; "
        f"letter-spacing:0.1em; color:{PALETTE['accent']}; margin-bottom:0.35rem;"
    )
    body_style = f"font-size:0.86rem; color:{PALETTE['ink']}; line-height:1.55; margin:0;"
    html = (
        f'<div style="{outer}">'
        f'<div style="{label_style}">Model routing &nbsp;·&nbsp; {segment}</div>'
        f'<p style="{body_style}">{explanation}</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def card(title: str, body_html: str):
    """Generic info card."""
    title_html = (
        f'<div style="font-family:\'Fraunces\', Georgia, serif; font-size:1rem; '
        f'font-weight:700; margin:0 0 0.5rem; color:{PALETTE["ink"]};">{title}</div>'
        if title else ""
    )
    body = dedent(body_html).strip().replace("\n", " ")
    outer_style = (
        f"background:{PALETTE['surface']}; border:1px solid {PALETTE['border']}; "
        f"border-radius:12px; padding:1.25rem 1.375rem; "
        f"box-shadow:0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03); "
        f"margin-bottom:1rem; color:{PALETTE['ink']};"
    )
    inner_style = f"color:{PALETTE['ink']}; font-size:0.88rem; line-height:1.6;"
    html = f'<div style="{outer_style}">{title_html}<div style="{inner_style}">{body}</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def tags(*items):
    """Plainly decorative tags — pointer-events: none, no hover state."""
    style = (
        f"display:inline-block; padding:0.25rem 0.6rem; background:{PALETTE['accent_bg']}; "
        f"color:{PALETTE['accent']}; border-radius:6px; font-size:0.72rem; font-weight:600; "
        f"pointer-events:none; user-select:none; cursor:default; "
        f"margin-right:0.45rem; margin-top:0.5rem;"
    )
    tag_html = "".join(f'<span style="{style}">{item}</span>' for item in items)
    st.markdown(f'<div>{tag_html}</div>', unsafe_allow_html=True)


def empty_state(icon: str, title: str, body: str):
    outer = (
        f"background:{PALETTE['surface_2']}; border:1.5px dashed {PALETTE['border']}; "
        f"border-radius:14px; padding:2.75rem 2rem; text-align:center;"
    )
    html = (
        f'<div style="{outer}">'
        f'<div style="font-size:1.75rem; margin-bottom:0.65rem;">{icon}</div>'
        f'<div style="font-size:0.95rem; font-weight:600; color:{PALETTE["ink"]}; margin-bottom:0.3rem;">{title}</div>'
        f'<div style="font-size:0.85rem; color:{PALETTE["ink_2"]}; line-height:1.5;">{body}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# Backward-compat aliases used by page_data_exploration.py
def hero(title, subtitle, chips=None):
    """Legacy wrapper — maps to page_header + tags."""
    page_header("Medical Insurance Cost Predictor", title, subtitle)
    if chips:
        tags(*chips)


# ── Feature encoding ──────────────────────────────────────────────────────────

REGION_LABELS = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3,
}

TREE_FEATURES   = ["age", "sex", "bmi", "children", "region", "bmi_smoker"]
LINEAR_FEATURES = [
    "age", "sex", "bmi", "children", "smoker", "bmi_smoker",
    "region_northwest", "region_southeast", "region_southwest",
]


def encode_tree_features(profile: dict, smoker_value: str) -> np.ndarray:
    sex_val    = 1 if profile["sex"] == "male" else 0
    smoker_bin = 1 if smoker_value == "yes" else 0
    return np.array([[
        profile["age"], sex_val, profile["bmi"], profile["children"],
        REGION_LABELS[profile["region"]], profile["bmi"] * smoker_bin,
    ]], dtype=float)


def encode_linear_features(profile: dict) -> np.ndarray:
    sex_val    = 1 if profile["sex"] == "male" else 0
    smoker_bin = 1 if profile["smoker_status"] == "yes" else 0
    return np.array([[
        profile["age"], sex_val, profile["bmi"], profile["children"],
        smoker_bin, profile["bmi"] * smoker_bin,
        1 if profile["region"] == "northwest" else 0,
        1 if profile["region"] == "southeast" else 0,
        1 if profile["region"] == "southwest" else 0,
    ]], dtype=float)


def encode_classifier_features(profile: dict) -> np.ndarray:
    sex_val = 1 if profile["sex"] == "male" else 0
    return np.array([[
        profile["age"], sex_val, profile["bmi"], profile["children"],
        1 if profile["region"] == "northwest" else 0,
        1 if profile["region"] == "southeast" else 0,
        1 if profile["region"] == "southwest" else 0,
    ]], dtype=float)


# ── Data + model loading ──────────────────────────────────────────────────────

@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_block2_assets():
    """
    Load Block 2 artifacts, training first if saved files are missing.

    block2_classifier.py saves the best classifier as either
    rf_smoker_classifier.pkl or lr_smoker_classifier.pkl depending on F1.
    We check both, and also always keep a logistic model for predict_proba
    in the unknown-smoker path.
    """
    regressor_files = [
        SAVED_DIR / "rf_regressor_smoker.pkl",
        SAVED_DIR / "rf_regressor_nonsmoker.pkl",
    ]
    classifier_files = [
        SAVED_DIR / "rf_smoker_classifier.pkl",
        SAVED_DIR / "lr_smoker_classifier.pkl",
    ]
    has_regressors = all(p.exists() for p in regressor_files)
    has_classifier = any(p.exists() for p in classifier_files)

    if not (has_regressors and has_classifier):
        from block2_classifier import run_block2
        run_block2()

    # Load the classifier that exists
    if (SAVED_DIR / "rf_smoker_classifier.pkl").exists():
        classifier = joblib.load(SAVED_DIR / "rf_smoker_classifier.pkl")
    else:
        classifier = joblib.load(SAVED_DIR / "lr_smoker_classifier.pkl")

    return {
        "classifier":          classifier,
        "smoker_regressor":    joblib.load(SAVED_DIR / "rf_regressor_smoker.pkl"),
        "nonsmoker_regressor": joblib.load(SAVED_DIR / "rf_regressor_nonsmoker.pkl"),
    }


@st.cache_resource
def load_classifier_scaler():
    """
    Fit a logistic classifier on the full dataset for smoker probability
    estimation in the unknown-smoker path. Uses preprocess.py for encoding.
    """
    df = get_classifier_data_logistic(path=str(DATA_PATH))
    X  = df.drop(columns=["smoker", "charges", "log_charges"])
    y  = df["smoker"].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X.values)
    model  = LogisticRegression(max_iter=1000, random_state=12138, class_weight="balanced")
    model.fit(X_sc, y)
    return {"model": model, "scaler": scaler, "columns": list(X.columns)}


@st.cache_resource
def load_quantile_assets():
    """
    Fit quantile models (q10, q50, q90) for uncertainty estimates.
    Uses preprocess.py for encoding.
    """
    df = get_regressor_data_linear(path=str(DATA_PATH))
    X = df[LINEAR_FEATURES].values
    y = np.log1p(df["charges"].values)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    models = {}
    for q in (0.1, 0.5, 0.9):
        qm = QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
        qm.fit(X_sc, y)
        models[q] = qm
    return {"models": models, "scaler": scaler}


@st.cache_data
def dataset_feature_baselines() -> dict:
    df = load_raw_data().copy()
    df["sex"]      = df["sex"].map({"female": 0, "male": 1})
    df["region"]   = df["region"].map(REGION_LABELS)
    df["smoker"]   = df["smoker"].map({"no": 0, "yes": 1})
    df["bmi_smoker"] = df["bmi"] * df["smoker"]
    return {col: float(df[col].median()) for col in TREE_FEATURES}


# ── Core prediction ───────────────────────────────────────────────────────────

def make_prediction(profile: dict) -> dict:
    """
    Produce a point estimate, 80% uncertainty interval, routing info,
    and feature-impact table for the given profile dict.
    """
    block2    = load_block2_assets()
    clf_assets = load_classifier_scaler()
    q_assets   = load_quantile_assets()

    smoker_cost    = float(block2["smoker_regressor"].predict(encode_tree_features(profile, "yes"))[0])
    nonsmoker_cost = float(block2["nonsmoker_regressor"].predict(encode_tree_features(profile, "no"))[0])

    if profile["smoker_status"] == "unknown":
        clf_vec    = encode_classifier_features(profile)
        clf_sc     = clf_assets["scaler"].transform(clf_vec)
        smoker_prob = float(clf_assets["model"].predict_proba(clf_sc)[0, 1])
        estimate    = smoker_prob * smoker_cost + (1.0 - smoker_prob) * nonsmoker_cost
        segment     = "weighted blend"
    elif profile["smoker_status"] == "yes":
        smoker_prob = 1.0
        estimate    = smoker_cost
        segment     = "smoker segment"
    else:
        smoker_prob = 0.0
        estimate    = nonsmoker_cost
        segment     = "non-smoker segment"

    # Quantile interval (use median-estimated smoker status for encoding)
    lin_profile = {**profile, "smoker_status": "yes" if smoker_prob >= 0.5 else "no"}
    q_vec = encode_linear_features(lin_profile)
    q_sc  = q_assets["scaler"].transform(q_vec)
    q10   = float(np.expm1(q_assets["models"][0.1].predict(q_sc)[0]))
    q50   = float(np.expm1(q_assets["models"][0.5].predict(q_sc)[0]))
    q90   = float(np.expm1(q_assets["models"][0.9].predict(q_sc)[0]))
    lower, median, upper = sorted([q10, q50, q90])

    # Feature impact heuristic
    if segment == "smoker segment":
        importances = block2["smoker_regressor"].feature_importances_
    elif segment == "non-smoker segment":
        importances = block2["nonsmoker_regressor"].feature_importances_
    else:
        importances = (
            smoker_prob * block2["smoker_regressor"].feature_importances_
            + (1.0 - smoker_prob) * block2["nonsmoker_regressor"].feature_importances_
        )

    baselines   = dataset_feature_baselines()
    feat_values = {
        "age":       profile["age"],
        "sex":       1 if profile["sex"] == "male" else 0,
        "bmi":       profile["bmi"],
        "children":  profile["children"],
        "region":    REGION_LABELS[profile["region"]],
        "bmi_smoker": profile["bmi"] * smoker_prob,
    }
    FEAT_LABELS = {"age": "Age", "sex": "Sex", "bmi": "BMI",
                   "children": "Children", "region": "Region",
                   "bmi_smoker": "BMI × smoker"}
    impact_rows = [
        {
            "feature":      name,
            "label":        FEAT_LABELS[name],
            "importance":   float(imp),
            "impact_score": float(abs(feat_values[name] - baselines[name]) * imp),
        }
        for name, imp in zip(TREE_FEATURES, importances)
    ]
    impacts = pd.DataFrame(impact_rows).sort_values("impact_score", ascending=False)

    return {
        "estimate":          estimate,
        "smoker_cost":       smoker_cost,
        "nonsmoker_cost":    nonsmoker_cost,
        "smoker_probability": smoker_prob,
        "segment":           segment,
        "q10":               lower,
        "q50":               median,
        "q90":               upper,
        "impacts":           impacts,
    }


def block2_summary_text(prediction: dict) -> str:
    p = prediction["smoker_probability"]
    if prediction["segment"] == "weighted blend":
        return (
            f"Smoker status was unknown, so the classifier estimated a <strong>{p:.0%} smoker probability</strong> "
            f"from the demographic profile. The final estimate blends both subgroup paths proportionally."
        )
    if prediction["segment"] == "smoker segment":
        return "Smoker status was provided as <strong>yes</strong>, so the profile routed directly to the smoker subgroup regressor."
    return "Smoker status was provided as <strong>no</strong>, so the profile routed directly to the non-smoker subgroup regressor."


# ── Comparison metrics ────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_comparison_metrics() -> dict:
    """
    Build the comparison bundle from saved metric JSON files.
    Trains any missing models lazily.

    Handles two JSON formats:
      - Allison's RF/XGB: {"metrics": {"MAE": ..., "RMSE": ..., "R2": ...}, ...}
      - Pipeline models:  {"mae": ..., "rmse": ..., "r2": ..., ...}
    """
    files = {
        "linear":        SAVED_DIR / "linear_metrics.json",
        "random_forest": SAVED_DIR / "random_forest_metrics.json",
        "rf_allison":    SAVED_DIR / "rf_metrics.json",
        "mlp":           SAVED_DIR / "mlp_metrics.json",
        "quantile":      SAVED_DIR / "quantile_metrics.json",
        "mdn":           SAVED_DIR / "mdn_metrics.json",
        "xgb":           SAVED_DIR / "xgb_metrics.json",
    }

    # Train missing models lazily
    if not files["linear"].exists():
        from linear_regression import fit_linear_regression
        fit_linear_regression()
    if not files["mlp"].exists():
        from mlp import fit_mlp
        fit_mlp()
    if not files["quantile"].exists():
        from quantile_regression import fit_quantile_models
        fit_quantile_models()
    if not files["mdn"].exists():
        from mdn import fit_mdn
        fit_mdn()
    # RF: check both formats
    if not files["random_forest"].exists() and not files["rf_allison"].exists():
        from random_forest import fit_random_forest
        fit_random_forest()

    # ── Load metrics ──────────────────────────────────────────────────────
    linear = _load_json(files["linear"])
    mlp    = _load_json(files["mlp"])
    qr     = _load_json(files["quantile"])
    mdn    = _load_json(files["mdn"])

    # ── Build leaderboard rows ────────────────────────────────────────────
    rows = [
        {
            "Model": "Linear Regression",
            "R2":    linear["r2_dollar"],
            "RMSE":  linear["rmse_dollar"],
            "MAE":   linear["mae_dollar"],
            "Notes": "Baseline · log-charges target",
        },
    ]

    # Random Forest — support both JSON formats
    if files["random_forest"].exists():
        rf = _load_json(files["random_forest"])
        rows.append({
            "Model": "Random Forest",
            "R2":    rf["r2"],
            "RMSE":  rf["rmse"],
            "MAE":   rf["mae"],
            "Notes": "Tree ensemble · used in predictor",
        })
    elif files["rf_allison"].exists():
        rf_raw = _load_json(files["rf_allison"])
        rf_m   = rf_raw["metrics"]
        rows.append({
            "Model": "Random Forest",
            "R2":    rf_m["R2"],
            "RMSE":  rf_m["RMSE"],
            "MAE":   rf_m["MAE"],
            "Notes": "Tree ensemble · GridSearchCV tuned",
        })

    rows.append({
        "Model": "MLP",
        "R2":    mlp["r2"],
        "RMSE":  mlp["rmse"],
        "MAE":   mlp["mae"],
        "Notes": "Feed-forward neural network",
    })
    rows.append({
        "Model": "MDN",
        "R2":    mdn["r2"],
        "RMSE":  mdn["rmse"],
        "MAE":   mdn["mae"],
        "Notes": "From-scratch mixture density network",
    })
    rows.append({
        "Model": "Quantile Reg.",
        "R2":    qr["r2_median_dollar"],
        "RMSE":  qr["rmse_median_dollar"],
        "MAE":   qr["mae_median_dollar"],
        "Notes": "Median + 80% interval",
    })

    # XGBoost — optional (Allison's format)
    if files["xgb"].exists():
        xgb_raw = _load_json(files["xgb"])
        xgb_m   = xgb_raw["metrics"]
        rows.append({
            "Model": "XGBoost",
            "R2":    xgb_m["R2"],
            "RMSE":  xgb_m["RMSE"],
            "MAE":   xgb_m["MAE"],
            "Notes": "Gradient boosting · GridSearchCV tuned",
        })

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values(["R2", "RMSE"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return {
        "leaderboard": leaderboard,
        "linear": linear,
        "mlp": mlp,
        "quantile": qr,
        "mdn": mdn,
    }


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_model_comparison(leaderboard: pd.DataFrame):
    """Horizontal bar chart comparing model R²."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    colors = [
        PALETTE["navy"] if i == 0 else PALETTE["ink_3"]
        for i in range(len(leaderboard))
    ]
    models = leaderboard["Model"].tolist()[::-1]
    r2s    = leaderboard["R2"].tolist()[::-1]
    bar_colors = colors[::-1]

    bars = ax.barh(models, r2s, color=bar_colors, height=0.55)
    ax.set_xlabel("R² on held-out test data", fontsize=9, color=PALETTE["ink_2"])
    ax.set_xlim(0, min(1.0, max(r2s) * 1.15))
    ax.tick_params(labelsize=9, colors=PALETTE["ink_2"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["border"])
    ax.xaxis.set_tick_params(color=PALETTE["border"])
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=PALETTE["border"], linewidth=0.6)

    for bar, val in zip(bars, r2s):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8.5,
            color=PALETTE["ink"], fontweight="600",
        )
    fig.tight_layout(pad=1.2)
    return fig


def plot_prediction_interval(df: pd.DataFrame):
    """Quantile interval vs actuals — first 50 test examples."""
    sample = df.head(50).copy().reset_index(drop=True)
    x = np.arange(len(sample))
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    ax.fill_between(x, sample["q10"], sample["q90"],
                    color=PALETTE["accent"], alpha=0.18, label="80% interval")
    ax.plot(x, sample["q50"], color=PALETTE["navy"], linewidth=1.8,
            label="Median prediction")
    ax.scatter(x, sample["actual_charges"], color=PALETTE["accent"],
               s=18, alpha=0.75, zorder=3, label="Actual charge")

    ax.set_xlabel("Test example", fontsize=9, color=PALETTE["ink_2"])
    ax.set_ylabel("Annual charge ($)", fontsize=9, color=PALETTE["ink_2"])
    ax.tick_params(labelsize=9, colors=PALETTE["ink_2"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(PALETTE["border"])
    ax.yaxis.grid(True, color=PALETTE["border"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(pad=1.2)
    return fig


def plot_feature_impacts(impacts: pd.DataFrame):
    """Horizontal bar chart of top local feature drivers."""
    top = impacts.head(5).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    fig.patch.set_facecolor(PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    ax.barh(top["label"], top["impact_score"], color=PALETTE["navy"], height=0.5)
    ax.set_xlabel("Relative influence score", fontsize=9, color=PALETTE["ink_2"])
    ax.tick_params(labelsize=9, colors=PALETTE["ink_2"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["border"])
    ax.xaxis.grid(True, color=PALETTE["border"], linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.2)
    return fig