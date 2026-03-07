"""
💧 Simulasi Sistem Distribusi Air - Pam Air (Water Tank)
Studi Kasus MODSIM 2026 - Praktikum 4: Continuous Simulation
NIM  : 11S25009
Nama : Wise Dwi Putra Napitupulu
"""

import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Simulasi Pam Air · MODSIM 2026",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Tema gelap ultramodern
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Latar utama dengan grain texture ── */
.stApp {
    background-color: #060a10;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,120,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(0,200,180,0.08) 0%, transparent 60%),
        url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(8, 14, 24, 0.95);
    border-right: 1px solid rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #8899aa !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #c8d8e8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}

/* slider track */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #00aaff !important;
    border-color: #00aaff !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBarMax"] {
    color: #556677 !important;
}

/* ── HERO HEADER ── */
.hero-wrap {
    position: relative;
    background: linear-gradient(135deg, #050e1c 0%, #091828 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 50% 80% at -5% 50%, rgba(0,130,255,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 30% 60% at 105% 50%, rgba(0,220,180,0.10) 0%, transparent 60%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '💧';
    position: absolute;
    right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.06;
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00aaff;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px; height: 1px;
    background: #00aaff;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #eef4ff;
    line-height: 1.1;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
}
.hero-title span {
    color: #00aaff;
}
.hero-desc {
    font-size: 0.9rem;
    color: #5a7a99;
    font-weight: 400;
    margin-bottom: 1.2rem;
    max-width: 600px;
    line-height: 1.6;
}
.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.chip {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    background: rgba(0,170,255,0.08);
    border: 1px solid rgba(0,170,255,0.2);
    color: #4499cc;
}
.chip.accent {
    background: rgba(0,220,180,0.08);
    border-color: rgba(0,220,180,0.2);
    color: #00ccaa;
}

/* ── METRIC CARDS ── */
.metrics-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.mcard {
    flex: 1;
    background: rgba(10,18,30,0.8);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.25s;
}
.mcard:hover {
    border-color: rgba(0,170,255,0.25);
    transform: translateY(-2px);
}
.mcard::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0,170,255,0.5), transparent);
}
.mcard.green::before { background: linear-gradient(90deg, transparent, rgba(0,220,130,0.5), transparent); }
.mcard.red::before   { background: linear-gradient(90deg, transparent, rgba(255,80,80,0.5), transparent); }
.mcard.gold::before  { background: linear-gradient(90deg, transparent, rgba(255,180,0,0.5), transparent); }
.mcard-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3a5570;
    margin-bottom: 0.5rem;
}
.mcard-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ddeeff;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.mcard-value .unit {
    font-size: 0.9rem;
    color: #3a5570;
    font-weight: 400;
}
.mcard-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #00aaff;
}
.mcard.green .mcard-sub { color: #00dd88; }
.mcard.red .mcard-sub   { color: #ff6688; }
.mcard.gold .mcard-sub  { color: #ffcc44; }

/* ── FORMULA CARD ── */
.fcard {
    background: rgba(0, 30, 60, 0.4);
    border: 1px solid rgba(0,170,255,0.12);
    border-left: 3px solid #00aaff;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #99bbdd;
    line-height: 2;
    margin: 0.8rem 0;
}
.fcard b { color: #00aaff; }
.fcard .result { color: #eef4ff; font-size: 0.95rem; }

/* ── INFO BOX ── */
.ibox {
    background: rgba(0,60,120,0.15);
    border: 1px solid rgba(0,150,255,0.15);
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    color: #7aaabb;
    font-size: 0.85rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    line-height: 1.5;
}
.ibox::before { content: '◈'; color: #00aaff; flex-shrink: 0; }

/* ── STATUS BADGE ── */
.sbadge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.9rem;
    border-radius: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
}
.sbadge.ok   { background: rgba(0,200,120,0.1); color: #00dd88; border: 1px solid rgba(0,200,120,0.2); }
.sbadge.warn { background: rgba(255,180,0,0.1);  color: #ffcc44; border: 1px solid rgba(255,180,0,0.2); }
.sbadge.bad  { background: rgba(255,60,60,0.1);  color: #ff6688; border: 1px solid rgba(255,60,60,0.2); }

/* ── SECTION TITLE ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #c8d8e8;
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,170,255,0.2), transparent);
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(8,14,24,0.8);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 5px;
    gap: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #445566 !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    font-family: 'DM Mono', monospace !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.5px;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,100,200,0.25) !important;
    color: #00aaff !important;
    border: 1px solid rgba(0,170,255,0.2) !important;
}

/* ── DATAFRAME ── */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stDataFrame"] table {
    background: rgba(8,14,24,0.9) !important;
}
[data-testid="stDataFrame"] th {
    background: rgba(0,40,80,0.6) !important;
    color: #4499cc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 10px 14px !important;
}
[data-testid="stDataFrame"] td {
    color: #8899aa !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 8px 14px !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
}

/* ── GENERAL TEXT ── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }
p, li, label, span { color: #8899aa; }
.stMarkdown p { color: #7a8a9a; line-height: 1.7; }
hr { border-color: rgba(255,255,255,0.04) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1a2a3a; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #224; }

/* sidebar section divider */
.sidebar-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,170,255,0.15), transparent);
    margin: 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER UTAMA
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">MODSIM 2026 · Praktikum 4</div>
    <div class="hero-title">Simulasi <span>Pam Air</span><br>Water Tank System</div>
    <div class="hero-desc">
        Pemodelan & Simulasi Continuous — pemecahan persamaan diferensial ketinggian
        air menggunakan metode Runge-Kutta 4/5 dengan scipy.integrate.solve_ivp
    </div>
    <div class="hero-chips">
        <span class="chip">NIM: 11S25009</span>
        <span class="chip">Wise Dwi Putra Napitupulu</span>
        <span class="chip accent">RK45 · scipy.solve_ivp</span>
        <span class="chip accent">Continuous Simulation</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WARNA & TEMPLATE PLOT
# ─────────────────────────────────────────────
PLOT_TEMPLATE = "plotly_dark"
PLOT_PAPER_BG = "rgba(0,0,0,0)"
PLOT_PLOT_BG  = "rgba(0,0,0,0)"
C_BLUE   = "#00aaff"
C_CYAN   = "#00ddcc"
C_GREEN  = "#00dd88"
C_RED    = "#ff5577"
C_GOLD   = "#ffcc44"
C_PURPLE = "#aa88ff"
C_MUTED  = "#334455"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameter Sistem")
    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)

    st.markdown("#### 📐 Dimensi Pam")
    radius      = st.slider("Jari-jari r [m]",   0.3, 2.0, 0.8, 0.05)
    height_tank = st.slider("Tinggi Pam H [m]",  0.5, 5.0, 2.0, 0.1)

    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)
    st.markdown("#### 🚿 Aliran Air")
    Q_in  = st.slider("Q_in [L/menit]",  0.0, 200.0, 80.0, 5.0)
    Q_out = st.slider("Q_out [L/menit]", 0.0, 200.0, 30.0, 5.0)

    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)
    st.markdown("#### 📊 Kondisi Awal & Durasi")
    h0_pct = st.slider("Ketinggian Awal [%]", 0, 100, 0)
    h0     = h0_pct / 100.0 * height_tank
    t_max  = st.slider("Durasi Simulasi [menit]", 10, 300, 120, 10)

    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#223344; line-height:2; font-family:"DM Mono",monospace;'>
    MODSIM 2026 · Praktikum 4<br>
    NIM 11S25009<br>
    Metode: RK45
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────
A_tank   = np.pi * radius ** 2
V_max    = A_tank * height_tank
Q_in_m3  = Q_in  / 1000.0 / 60.0
Q_out_m3 = Q_out / 1000.0 / 60.0

# ─────────────────────────────────────────────
# MODEL ODE
# ─────────────────────────────────────────────
def water_tank_ode(t, y, q_in, q_out, A, H):
    h = y[0]
    actual_qout = q_out if h > 1e-6 else 0.0
    actual_qin  = q_in  if h < H - 1e-6 else 0.0
    net = actual_qin - actual_qout
    dhdt = net / A
    dvdt = net
    if h >= H and net > 0: dhdt = 0.0; dvdt = 0.0
    if h <= 0 and net < 0: dhdt = 0.0; dvdt = 0.0
    return [dhdt, dvdt]

def run_simulation(q_in, q_out, h_init, t_end_min, A, H, n_pts=1200):
    t_span = (0, t_end_min * 60)
    t_eval = np.linspace(0, t_end_min * 60, n_pts)
    y0 = [np.clip(h_init, 0, H), np.clip(h_init, 0, H) * A]
    sol = solve_ivp(water_tank_ode, t_span, y0,
                    args=(q_in, q_out, A, H),
                    t_eval=t_eval, method='RK45',
                    dense_output=True, max_step=10.0)
    h_arr   = np.clip(sol.y[0], 0, H)
    v_arr   = h_arr * A
    pct_arr = h_arr / H * 100
    return pd.DataFrame({
        'time_s': sol.t, 'time_min': sol.t / 60,
        'height_m': h_arr, 'volume_m3': v_arr,
        'volume_liter': v_arr * 1000, 'pct_full': pct_arr
    })

def style_plot(fig, title="", h=440):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor=PLOT_PAPER_BG,
        plot_bgcolor=PLOT_PLOT_BG,
        title=dict(
            text=title,
            font=dict(size=13, color="#c8d8e8", family="Syne"),
            x=0, xanchor='left'
        ),
        font=dict(family="DM Sans", color="#556677", size=11),
        height=h,
        margin=dict(l=55, r=30, t=55, b=50),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(color="#8899aa", size=11)
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(0,170,255,0.15)",
            tickfont=dict(color="#445566"),
            title_font=dict(color="#556677")
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(0,170,255,0.15)",
            tickfont=dict(color="#445566"),
            title_font=dict(color="#556677")
        ),
    )
    return fig

# ─────────────────────────────────────────────
# SIMULASI UTAMA
# ─────────────────────────────────────────────
df = run_simulation(Q_in_m3, Q_out_m3, h0, t_max, A_tank, height_tank)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "00 · Ringkasan",
    "01 · Waktu Mengisi",
    "02 · Waktu Kosong",
    "03 · Profil h(t)",
    "04 · Simultan",
    "05 · Ukuran Optimal",
])

# ══════════════════════════════════════════════
# TAB 0 — RINGKASAN
# ══════════════════════════════════════════════
with tab0:
    net_lpm = Q_in - Q_out
    h_akhir = df['height_m'].iloc[-1]
    pct_akhir = df['pct_full'].iloc[-1]

    if net_lpm > 0:
        net_color = "green"; net_txt = f"↑ +{net_lpm:.1f} L/mnt"
    elif net_lpm < 0:
        net_color = "red";   net_txt = f"↓ {net_lpm:.1f} L/mnt"
    else:
        net_color = "gold";  net_txt = "= 0 L/mnt"

    if pct_akhir > 60:   lvl_cls = "green"
    elif pct_akhir > 20: lvl_cls = "gold"
    else:                lvl_cls = "red"

    st.markdown(f"""
    <div class="metrics-row">
        <div class="mcard">
            <div class="mcard-label">Volume Maksimum</div>
            <div class="mcard-value">{V_max*1000:.0f} <span class="unit">L</span></div>
            <div class="mcard-sub">{V_max:.3f} m³ · r={radius}m · H={height_tank}m</div>
        </div>
        <div class="mcard">
            <div class="mcard-label">Luas Penampang A</div>
            <div class="mcard-value">{A_tank:.3f} <span class="unit">m²</span></div>
            <div class="mcard-sub">π × {radius}² = {A_tank:.4f} m²</div>
        </div>
        <div class="mcard {net_color}">
            <div class="mcard-label">Net Flow Rate</div>
            <div class="mcard-value">{net_lpm:+.1f} <span class="unit">L/mnt</span></div>
            <div class="mcard-sub">{net_txt}</div>
        </div>
        <div class="mcard {lvl_cls}">
            <div class="mcard-label">Tinggi Akhir [{t_max}mnt]</div>
            <div class="mcard-value">{h_akhir:.2f} <span class="unit">m</span></div>
            <div class="mcard-sub">{pct_akhir:.1f}% penuh</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Grafik 4-panel
    fig0 = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Tinggi Air  h(t)", "Volume Air  V(t)",
                        "Persentase Pengisian", "Snapshot per Interval"),
        horizontal_spacing=0.1, vertical_spacing=0.18
    )

    # Panel 1 – h(t)
    fig0.add_trace(go.Scatter(
        x=df['time_min'], y=df['height_m'], mode='lines', name='h(t)',
        line=dict(color=C_BLUE, width=2.5),
        fill='tozeroy', fillcolor='rgba(0,170,255,0.05)'
    ), row=1, col=1)
    fig0.add_hline(y=height_tank, line_dash='dash', line_color=C_RED,
                   line_width=1, annotation_text='H_max',
                   annotation_font=dict(color=C_RED, size=10), row=1, col=1)

    # Panel 2 – V(t)
    fig0.add_trace(go.Scatter(
        x=df['time_min'], y=df['volume_liter'], mode='lines', name='V(t)',
        line=dict(color=C_CYAN, width=2.5),
        fill='tozeroy', fillcolor='rgba(0,220,200,0.05)'
    ), row=1, col=2)

    # Panel 3 – %
    fig0.add_trace(go.Scatter(
        x=df['time_min'], y=df['pct_full'], mode='lines', name='% Penuh',
        line=dict(color=C_GREEN, width=2.5),
        fill='tozeroy', fillcolor='rgba(0,220,130,0.05)'
    ), row=2, col=1)
    fig0.add_hline(y=100, line_dash='dot', line_color=C_RED, line_width=1,
                   annotation_text='Penuh',
                   annotation_font=dict(color=C_RED, size=9), row=2, col=1)

    # Panel 4 – Bar snapshot
    snaps = df.iloc[np.linspace(0, len(df)-1, 10, dtype=int)]
    alpha_vals = np.linspace(0.25, 0.85, len(snaps))
    colors_bar = [f'rgba(0,170,255,{a:.2f})' for a in alpha_vals]
    fig0.add_trace(go.Bar(
        x=[f"{t:.0f}m" for t in snaps['time_min']],
        y=snaps['pct_full'], name='Snapshot',
        marker_color=colors_bar,
        marker_line=dict(color='rgba(0,170,255,0.3)', width=1)
    ), row=2, col=2)

    style_plot(fig0, "", h=580)
    fig0.update_annotations(font=dict(size=11, color="#8899aa", family="Syne"))
    for i in range(1, 3):
        for j in range(1, 3):
            fig0.update_xaxes(title_text="menit", row=i, col=j,
                              gridcolor="rgba(255,255,255,0.03)",
                              tickfont=dict(color="#445566"))
            fig0.update_yaxes(gridcolor="rgba(255,255,255,0.03)",
                              tickfont=dict(color="#445566"), row=i, col=j)
    st.plotly_chart(fig0, use_container_width=True)

    # Model Matematika
    st.markdown('<div class="section-title">📐 Model Matematika</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fcard">
    <b>Persamaan Diferensial Utama:</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;dh/dt = ( Q_in − Q_out ) / A<br><br>
    <b>Volume Silinder:</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;V(t) = A · h(t) = π · r² · h(t)<br><br>
    <b>Constraint Fisik:</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;0 ≤ h(t) ≤ H &nbsp;—&nbsp; pam tidak meluap atau bernilai negatif<br><br>
    <b>Metode Numerik:</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;Runge-Kutta Orde 4/5 (RK45) via <span class="result">scipy.integrate.solve_ivp</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 1 — SOAL 1: WAKTU MENGISI
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">⏱ Soal 1 — Waktu Mengisi Pam Hingga Penuh</div>', unsafe_allow_html=True)
    st.markdown("""<div class="ibox">
    Kondisi: Pam <b>kosong (h=0)</b>, hanya Q_in aktif, Q_out = 0.
    Simulasi berjalan sampai h mencapai H (100% penuh).
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        q_fill = st.slider("Q_in [L/menit]", 10.0, 300.0, float(Q_in), 5.0, key="qf1")
        t_teori = V_max * 1000 / q_fill
        st.markdown(f"""
        <div class="fcard">
        <b>Analitik (laju konstan):</b><br>
        t = V_max / Q_in<br>
        t = {V_max*1000:.1f} L ÷ {q_fill:.1f} L/mnt<br>
        <span class="result">→ t = {t_teori:.2f} menit</span>
        </div>""", unsafe_allow_html=True)

    with col2:
        q_f_m3 = q_fill / 1000.0 / 60.0
        t_sim  = max(int(t_teori * 1.4) + 10, 30)
        df1    = run_simulation(q_f_m3, 0.0, 0.0, t_sim, A_tank, height_tank)
        idx_f  = df1[df1['height_m'] >= height_tank * 0.999].index
        t_full = df1.loc[idx_f[0], 'time_min'] if len(idx_f) > 0 else t_sim

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df1['time_min'], y=df1['height_m'],
            mode='lines', name='h(t)',
            line=dict(color=C_BLUE, width=3),
            fill='tozeroy', fillcolor='rgba(0,170,255,0.06)'
        ))
        fig1.add_vline(x=t_full, line_dash='dash', line_color=C_GREEN, line_width=1.5,
                       annotation_text=f'Penuh ✓  {t_full:.1f} mnt',
                       annotation_font=dict(color=C_GREEN, size=11))
        fig1.add_hline(y=height_tank, line_dash='dot', line_color=C_RED, line_width=1,
                       annotation_text=f'H = {height_tank} m',
                       annotation_font=dict(color=C_RED, size=10))
        style_plot(fig1, f"Pengisian Pam — selesai dalam {t_full:.2f} menit")
        fig1.update_xaxes(title_text="Waktu (menit)")
        fig1.update_yaxes(title_text="Tinggi Air h(t)  [m]")
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown(f"""
    <div class="metrics-row" style="margin-top:0.5rem;">
        <div class="mcard green">
            <div class="mcard-label">Waktu Pengisian (Simulasi)</div>
            <div class="mcard-value">{t_full:.2f} <span class="unit">menit</span></div>
            <div class="mcard-sub">{t_full*60:.0f} detik · analitik = {t_teori:.2f} mnt</div>
        </div>
        <div class="mcard">
            <div class="mcard-label">Volume yang Diisi</div>
            <div class="mcard-value">{V_max*1000:.0f} <span class="unit">L</span></div>
            <div class="mcard-sub">Q_in = {q_fill:.1f} L/mnt</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — SOAL 2: WAKTU MENGOSONGKAN
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🚿 Soal 2 — Waktu Mengosongkan Pam</div>', unsafe_allow_html=True)
    st.markdown("""<div class="ibox">
    Kondisi: Pam <b>penuh (h=H)</b>, hanya Q_out aktif, Q_in = 0.
    Simulasi berjalan sampai h mendekati 0 (kosong).
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        q_drain = st.slider("Q_out [L/menit]", 10.0, 300.0,
                            float(max(Q_out, 30.0)), 5.0, key="qd2")
        t_teori_d = V_max * 1000 / q_drain
        st.markdown(f"""
        <div class="fcard">
        <b>Analitik (laju konstan):</b><br>
        t = V_max / Q_out<br>
        t = {V_max*1000:.1f} L ÷ {q_drain:.1f} L/mnt<br>
        <span class="result">→ t = {t_teori_d:.2f} menit</span>
        </div>""", unsafe_allow_html=True)

    with col2:
        q_d_m3 = q_drain / 1000.0 / 60.0
        t_sim2 = max(int(t_teori_d * 1.4) + 10, 30)
        df2    = run_simulation(0.0, q_d_m3, height_tank, t_sim2, A_tank, height_tank)
        idx_e  = df2[df2['height_m'] <= 0.01].index
        t_empty= df2.loc[idx_e[0], 'time_min'] if len(idx_e) > 0 else t_sim2

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df2['time_min'], y=df2['height_m'],
            mode='lines', name='h(t)',
            line=dict(color=C_RED, width=3),
            fill='tozeroy', fillcolor='rgba(255,85,119,0.06)'
        ))
        fig2.add_vline(x=t_empty, line_dash='dash', line_color=C_GOLD, line_width=1.5,
                       annotation_text=f'Kosong ✓  {t_empty:.1f} mnt',
                       annotation_font=dict(color=C_GOLD, size=11))
        style_plot(fig2, f"Pengosongan Pam — selesai dalam {t_empty:.2f} menit")
        fig2.update_xaxes(title_text="Waktu (menit)")
        fig2.update_yaxes(title_text="Tinggi Air h(t)  [m]")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    <div class="metrics-row" style="margin-top:0.5rem;">
        <div class="mcard gold">
            <div class="mcard-label">Waktu Pengosongan (Simulasi)</div>
            <div class="mcard-value">{t_empty:.2f} <span class="unit">menit</span></div>
            <div class="mcard-sub">{t_empty*60:.0f} detik · analitik = {t_teori_d:.2f} mnt</div>
        </div>
        <div class="mcard">
            <div class="mcard-label">Volume Dikosongkan</div>
            <div class="mcard-value">{V_max*1000:.0f} <span class="unit">L</span></div>
            <div class="mcard-sub">Q_out = {q_drain:.1f} L/mnt</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — SOAL 3: PROFIL KETINGGIAN
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📈 Soal 3 — Profil Ketinggian h(t) terhadap Waktu</div>', unsafe_allow_html=True)
    st.markdown("""<div class="ibox">
    Visualisasi profil h(t) dari berbagai skenario Q_in / Q_out dan kondisi awal berbeda.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        scenarios = {
            "Pengisian Penuh (Q_out=0)": (Q_in_m3, 0, 0.0, C_BLUE),
            "Pengosongan (Q_in=0)":       (0, Q_out_m3, height_tank, C_RED),
            "Net Positif":                (Q_in_m3, Q_in_m3*0.4, h0, C_GREEN),
            "Net Negatif":                (Q_in_m3*0.4, Q_in_m3, h0, C_GOLD),
            "Setimbang (Q_in=Q_out)":     (Q_in_m3, Q_in_m3, height_tank*0.5, C_PURPLE),
        }
        selected = st.multiselect("Skenario yang ditampilkan:",
                                  list(scenarios.keys()),
                                  default=list(scenarios.keys())[:3])

    with col2:
        fig3 = go.Figure()
        for name in selected:
            qi, qo, hi, color = scenarios[name]
            dfs = run_simulation(qi, qo, hi, t_max, A_tank, height_tank)
            fig3.add_trace(go.Scatter(
                x=dfs['time_min'], y=dfs['height_m'],
                mode='lines', name=name,
                line=dict(color=color, width=2.5)
            ))
        fig3.add_hline(y=height_tank, line_dash='dot', line_color=C_RED, line_width=1,
                       annotation_text='H_max', annotation_font=dict(color=C_RED, size=10))
        fig3.add_hline(y=0, line_dash='dot', line_color=C_MUTED, line_width=1)
        style_plot(fig3, "Profil h(t) — Multi Skenario")
        fig3.update_xaxes(title_text="Waktu (menit)")
        fig3.update_yaxes(title_text="Tinggi h(t)  [m]")
        st.plotly_chart(fig3, use_container_width=True)

    # Cross-section visualisasi pam
    st.markdown('<div class="section-title">🛢 Penampang Silinder Pam (Kondisi Akhir)</div>', unsafe_allow_html=True)
    h_now  = df['height_m'].iloc[-1]
    pct_now = df['pct_full'].iloc[-1]

    fig3b = go.Figure()
    fig3b.add_shape(type="rect",
        x0=-radius, x1=radius, y0=0, y1=height_tank,
        fillcolor="rgba(10,20,40,0.5)",
        line=dict(color="rgba(0,170,255,0.2)", width=2))
    # Air
    fill_color = f"rgba(0,170,255,{0.15 + pct_now/100*0.30})"
    fig3b.add_shape(type="rect",
        x0=-radius*0.97, x1=radius*0.97, y0=0, y1=h_now,
        fillcolor=fill_color,
        line=dict(color=C_BLUE, width=1.5))
    # Permukaan air (glow line)
    if h_now > 0.05:
        fig3b.add_shape(type="line",
            x0=-radius*0.97, x1=radius*0.97,
            y0=h_now, y1=h_now,
            line=dict(color=C_CYAN, width=3))

    if h_now > 0.15:
        fig3b.add_annotation(x=0, y=h_now/2,
            text=f"<b>{h_now:.2f} m</b><br>{h_now/height_tank*100:.0f}%",
            showarrow=False, font=dict(color="#eef4ff", size=14, family="Syne"))
    if height_tank - h_now > 0.15:
        fig3b.add_annotation(x=0, y=(h_now + height_tank)/2,
            text=f"{height_tank-h_now:.2f} m kosong",
            showarrow=False, font=dict(color="#334455", size=11))

    # Dimensi annotations
    fig3b.add_annotation(x=radius*1.2, y=height_tank/2,
        text=f"H = {height_tank} m",
        showarrow=True, arrowhead=2, arrowcolor=C_MUTED,
        ax=30, ay=0, font=dict(color="#445566", size=10))
    fig3b.add_annotation(x=0, y=-0.25,
        text=f"⌀ = {radius*2:.1f} m",
        showarrow=False, font=dict(color="#445566", size=10))

    style_plot(fig3b, f"Penampang Pam — r={radius}m, H={height_tank}m — {pct_now:.1f}% penuh", h=400)
    fig3b.update_xaxes(range=[-radius*2, radius*2], title_text="Lebar (m)")
    fig3b.update_yaxes(range=[-0.4, height_tank*1.2], title_text="Tinggi (m)")
    st.plotly_chart(fig3b, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — SOAL 4: BERSAMAAN
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">⚖ Soal 4 — Pengisian & Pengosongan Bersamaan</div>', unsafe_allow_html=True)
    st.markdown("""<div class="ibox">
    Saat Q_in dan Q_out aktif bersamaan, perilaku pam ditentukan oleh
    <b>net flow = Q_in − Q_out</b>.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        qi4  = st.slider("Q_in [L/menit]",  0.0, 200.0, float(Q_in),  5.0, key="qi4")
        qo4  = st.slider("Q_out [L/menit]", 0.0, 200.0, float(Q_out), 5.0, key="qo4")
        h04  = st.slider("h₀ Awal [%]", 0, 100, 50, key="h04") / 100 * height_tank
        net4 = qi4 - qo4

        if net4 > 0:   badge = f'<span class="sbadge ok">↑ Mengisi &nbsp;net +{net4:.1f} L/mnt</span>'
        elif net4 < 0: badge = f'<span class="sbadge bad">↓ Kosong &nbsp;net {net4:.1f} L/mnt</span>'
        else:           badge = f'<span class="sbadge warn">= Setimbang &nbsp;net 0</span>'

        st.markdown(f"<br>{badge}", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="fcard" style="margin-top:0.8rem;">
        <b>Persamaan ODE:</b><br>
        dh/dt = ({qi4:.1f} − {qo4:.1f}) L/mnt ÷ A<br>
        A = {A_tank:.4f} m²<br>
        <span class="result">dh/dt = {net4/(A_tank*60000):.6f} m/s</span>
        </div>""", unsafe_allow_html=True)

    with col2:
        df4 = run_simulation(qi4/1000/60, qo4/1000/60, h04, t_max, A_tank, height_tank)

        fig4 = make_subplots(rows=1, cols=2,
            subplot_titles=("Profil h(t)", "Volume  V(t) [Liter]"),
            horizontal_spacing=0.12)

        fig4.add_trace(go.Scatter(
            x=df4['time_min'], y=df4['height_m'], mode='lines', name='h(t)',
            line=dict(color=C_BLUE, width=3),
            fill='tozeroy', fillcolor='rgba(0,170,255,0.06)'
        ), row=1, col=1)
        fig4.add_hline(y=height_tank, line_dash='dot', line_color=C_RED,
                       line_width=1, row=1, col=1)
        fig4.add_hline(y=h04, line_dash='dot', line_color=C_MUTED,
                       line_width=1, annotation_text='h₀',
                       annotation_font=dict(color=C_MUTED, size=9), row=1, col=1)

        fig4.add_trace(go.Scatter(
            x=df4['time_min'], y=df4['volume_liter'], mode='lines', name='V(t)',
            line=dict(color=C_PURPLE, width=2.5),
            fill='tozeroy', fillcolor='rgba(170,136,255,0.06)'
        ), row=1, col=2)

        style_plot(fig4, f"Q_in={qi4:.0f}  Q_out={qo4:.0f} L/mnt  →  Net={net4:+.0f} L/mnt")
        fig4.update_xaxes(title_text="Waktu (menit)",
                          gridcolor="rgba(255,255,255,0.03)", tickfont=dict(color="#445566"))
        fig4.update_yaxes(gridcolor="rgba(255,255,255,0.03)", tickfont=dict(color="#445566"))
        fig4.update_annotations(font=dict(size=11, color="#8899aa", family="Syne"))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-title">📋 Analisis Tiga Kondisi Net Flow</div>', unsafe_allow_html=True)
    cases_df = pd.DataFrame({
        'Kondisi':   ['Q_in > Q_out', 'Q_in = Q_out', 'Q_in < Q_out'],
        'Net Flow':  ['Positif (+)', 'Nol (0)', 'Negatif (−)'],
        'dh/dt':     ['> 0 → naik', '= 0 → tetap', '< 0 → turun'],
        'Perilaku':  ['Terisi → h → H', 'Ketinggian konstan', 'Dikosongkan → h → 0'],
        'Berhenti':  ['Saat h = H (overflow guard)', 'Tidak berhenti', 'Saat h = 0 (underflow guard)'],
    })
    st.dataframe(cases_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 5 — SOAL 5: UKURAN OPTIMAL
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🔧 Soal 5 — Menentukan Ukuran Pam Optimal</div>', unsafe_allow_html=True)
    st.markdown("""<div class="ibox">
    Tentukan dimensi pam (r, H) yang paling efisien untuk memenuhi kebutuhan air harian bangunan.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        daily_L  = st.number_input("Kebutuhan harian [Liter]", 500, 50000, 5000, 500)
        peak_hr  = st.slider("Jam puncak [jam/hari]", 1, 12, 4)
        sf       = st.slider("Safety Factor", 1.0, 2.0, 1.2, 0.05)
        q_refill = st.slider("Laju isi tersedia [L/menit]", 10.0, 200.0, float(Q_in), 5.0, key="qr5")

        target_L  = daily_L * sf
        peak_qout = daily_L / (peak_hr * 60)
        t_refill  = target_L / q_refill

        st.markdown(f"""
        <div class="fcard">
        <b>Target volume:</b> {target_L:,.0f} L<br>
        <b>Q_out puncak:</b> {peak_qout:.1f} L/mnt<br>
        <b>Safety factor:</b> {sf}×<br>
        <span class="result">Waktu pengisian: {t_refill:.1f} mnt</span>
        </div>""", unsafe_allow_html=True)

    with col2:
        radii   = np.linspace(0.3, 2.5, 80)
        heights = np.linspace(0.5, 5.0, 80)
        R_g, H_g = np.meshgrid(radii, heights)
        V_g = np.pi * R_g**2 * H_g * 1000  # liter

        fig5 = go.Figure(data=go.Contour(
            x=radii, y=heights, z=V_g,
            colorscale=[
                [0.0,  '#060a10'],
                [0.2,  '#071830'],
                [0.45, '#093060'],
                [0.7,  '#005599'],
                [0.88, '#0088dd'],
                [1.0,  '#00ccff'],
            ],
            contours=dict(start=500, end=40000, size=2500,
                          showlabels=True,
                          labelfont=dict(size=9, color='rgba(255,255,255,0.6)')),
            colorbar=dict(
                title=dict(text="Volume (L)",
                           font=dict(color='#445566', size=10, family='DM Mono')),
                tickfont=dict(color='#445566', size=9),
                thickness=12,
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(255,255,255,0.05)',
            )
        ))

        # Titik optimal
        mask = V_g >= target_L
        if mask.any():
            # Cari titik dengan tinggi minimum (paling efisien)
            opt_idx = np.argmin(H_g[mask])
            opt_r   = R_g[mask].flatten()[opt_idx]
            opt_h   = H_g[mask].flatten()[opt_idx]
            fig5.add_trace(go.Scatter(
                x=[opt_r], y=[opt_h], mode='markers',
                marker=dict(color=C_GOLD, size=18, symbol='star',
                            line=dict(color='white', width=1.5)),
                name='★ Optimal', showlegend=True
            ))
            fig5.add_annotation(x=opt_r, y=opt_h,
                text=f"  r={opt_r:.2f}m<br>  H={opt_h:.2f}m",
                showarrow=False,
                font=dict(color=C_GOLD, size=10, family='DM Mono'),
                xanchor='left')

        style_plot(fig5, f"Peta Volume Pam  [Liter]  —  target ≥ {target_L:,.0f} L", h=460)
        fig5.update_xaxes(title_text="Jari-jari r (m)")
        fig5.update_yaxes(title_text="Tinggi H (m)")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-title">📋 Tabel Rekomendasi Ukuran Pam</div>', unsafe_allow_html=True)
    recs = []
    for r in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        A_r    = np.pi * r**2
        H_need = (target_L / 1000) / A_r
        t_fill = target_L / q_refill
        if H_need <= 2.5:   status = "✅ Ideal"
        elif H_need <= 4.0: status = "⚠️ Tinggi"
        else:               status = "❌ Terlalu Tinggi"
        recs.append({
            'r (m)': r,
            'H dibutuhkan (m)': round(H_need, 2),
            'Volume (L)': f"{target_L:,.0f}",
            'Luas A (m²)': round(A_r, 3),
            'Waktu Isi (mnt)': round(t_fill, 1),
            'Status': status
        })
    st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="metrics-row" style="margin-top:1rem;">
        <div class="mcard green">
            <div class="mcard-label">Volume Target</div>
            <div class="mcard-value">{target_L:,.0f} <span class="unit">L</span></div>
            <div class="mcard-sub">{target_L/1000:.2f} m³ · safety factor {sf}×</div>
        </div>
        <div class="mcard">
            <div class="mcard-label">Kebutuhan Harian</div>
            <div class="mcard-value">{daily_L:,} <span class="unit">L</span></div>
            <div class="mcard-sub">Jam puncak {peak_hr} jam · Q_out ≈ {peak_qout:.1f} L/mnt</div>
        </div>
        <div class="mcard gold">
            <div class="mcard-label">Waktu Pengisian</div>
            <div class="mcard-value">{t_refill:.1f} <span class="unit">mnt</span></div>
            <div class="mcard-sub">dengan Q_in = {q_refill:.0f} L/mnt</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:0.8rem 0;'>
    <span style='font-family:"DM Mono",monospace; font-size:0.7rem; color:#1e3040; letter-spacing:2px;'>
    MODSIM 2026 · PRAKTIKUM 4 &nbsp;·&nbsp; NIM 11S25009 &nbsp;·&nbsp;
    WISE DWI PUTRA NAPITUPULU &nbsp;·&nbsp; RK45
    </span>
</div>
""", unsafe_allow_html=True)