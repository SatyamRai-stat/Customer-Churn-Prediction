import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnLens AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS + ANIMATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ─ RESET ─ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: #03000A !important;
    color: #E8E8FF !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stHeader"]          { display: none !important; }
[data-testid="stSidebar"]         { display: none !important; }
[data-testid="stToolbar"]         { display: none !important; }
[data-testid="stDecoration"]      { display: none !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ─ SCROLLBAR ─ */
::-webkit-scrollbar              { width: 4px; }
::-webkit-scrollbar-track        { background: #03000A; }
::-webkit-scrollbar-thumb        { background: #7C3AED; border-radius: 99px; }

/* ══════════════════════════════════
   HERO CANVAS SECTION
══════════════════════════════════ */
.hero-wrap {
    position: relative;
    width: 100%;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    background: radial-gradient(ellipse 80% 60% at 50% 10%, #1a0533 0%, #03000A 70%);
}
#neural-canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
}
.hero-content {
    position: relative;
    z-index: 2;
    text-align: center;
    padding: 0 24px;
    animation: heroFadeIn 1.2s cubic-bezier(.16,1,.3,1) both;
}
@keyframes heroFadeIn {
    from { opacity: 0; transform: translateY(40px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 99px;
    padding: 6px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: #C4B5FD;
    margin-bottom: 32px;
    text-transform: uppercase;
    animation: heroFadeIn 1.4s 0.2s cubic-bezier(.16,1,.3,1) both;
}
.hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #7C3AED;
    box-shadow: 0 0 8px #7C3AED;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: .5; transform: scale(1.4); }
}
.hero-title {
    font-size: clamp(52px, 9vw, 120px);
    font-weight: 900;
    line-height: 0.92;
    letter-spacing: -0.04em;
    color: #fff;
    margin-bottom: 24px;
    animation: heroFadeIn 1.4s 0.35s cubic-bezier(.16,1,.3,1) both;
}
.hero-title .grad {
    background: linear-gradient(135deg, #7C3AED 0%, #EC4899 50%, #F59E0B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
}
/* Remove glitch pseudo-elements - they cause doubled text in Streamlit's iframe */
.glitch::before, .glitch::after { display: none !important; }
.hero-sub {
    font-size: clamp(14px, 1.6vw, 18px);
    font-weight: 300;
    color: rgba(232,232,255,0.55);
    max-width: 540px;
    margin: 0 auto 48px;
    line-height: 1.7;
    animation: heroFadeIn 1.4s 0.5s cubic-bezier(.16,1,.3,1) both;
}
.hero-stats {
    display: flex;
    gap: 48px;
    justify-content: center;
    animation: heroFadeIn 1.4s 0.65s cubic-bezier(.16,1,.3,1) both;
    margin-bottom: 56px;
}
.hero-stat-val {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(135deg, #A78BFA, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 4px;
}
.hero-stat-lbl {
    font-size: 11px;
    font-weight: 500;
    color: rgba(232,232,255,0.4);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.scroll-cta {
    animation: heroFadeIn 1.4s 0.8s cubic-bezier(.16,1,.3,1) both, float 3s 2s ease-in-out infinite;
    cursor: pointer;
    font-size: 13px;
    color: rgba(232,232,255,0.35);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
@keyframes float {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(8px); }
}
.scroll-arrow {
    width: 24px; height: 24px;
    border-right: 2px solid rgba(124,58,237,0.6);
    border-bottom: 2px solid rgba(124,58,237,0.6);
    transform: rotate(45deg);
}

/* ══════════════════════════════════
   MAIN APP BODY
══════════════════════════════════ */
.app-body {
    padding: 60px clamp(20px, 5vw, 80px) 100px;
    max-width: 1400px;
    margin: 0 auto;
}

/* ─ SECTION HEADER ─ */
.sec-head {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 28px;
    margin-top: 52px;
}
.sec-pip {
    width: 4px;
    height: 28px;
    border-radius: 2px;
    background: linear-gradient(180deg, #7C3AED, #EC4899);
    flex-shrink: 0;
}
.sec-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(232,232,255,0.5);
}

/* ─ GLASS CARD ─ */
.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 28px 32px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: border-color .3s, box-shadow .3s;
}
.glass:hover {
    border-color: rgba(124,58,237,0.3);
    box-shadow: 0 0 40px rgba(124,58,237,0.08);
}

/* ─ STREAMLIT WIDGETS ─ */
[data-testid="stSelectbox"] label p,
[data-testid="stNumberInput"] label p,
[data-testid="stFileUploader"] label p {
    color: rgba(232,232,255,0.45) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #0F0A1A !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #E8E8FF !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(124,58,237,0.7) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.18) !important;
}
/* ─ DROPDOWN LIST ─ */
[data-testid="stSelectbox"] ul,
[data-baseweb="popover"] ul,
[data-baseweb="menu"],
[data-baseweb="popover"] [data-baseweb="menu"] {
    background: #130D22 !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 12px !important;
    padding: 6px !important;
}
/* Every option item */
[data-baseweb="menu"] li,
[data-baseweb="option"],
[data-baseweb="menu"] [role="option"],
[data-baseweb="popover"] li {
    background: transparent !important;
    color: #E8E8FF !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}
/* Hovered option */
[data-baseweb="menu"] li:hover,
[data-baseweb="option"]:hover,
[data-baseweb="menu"] [role="option"]:hover {
    background: rgba(124,58,237,0.25) !important;
    color: #fff !important;
}
/* Selected/active option */
[data-baseweb="menu"] [aria-selected="true"] {
    background: rgba(124,58,237,0.15) !important;
    color: #C4B5FD !important;
}
/* Arrow icon in selectbox */
[data-testid="stSelectbox"] svg {
    fill: rgba(196,181,253,0.6) !important;
}
/* Number input — fix white background & invisible text */
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
    background: #0F0A1A !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
}
[data-testid="stNumberInput"] input {
    background: #0F0A1A !important;
    border: none !important;
    border-radius: 12px !important;
    color: #E8E8FF !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    caret-color: #A78BFA !important;
}
[data-testid="stNumberInput"] input:focus {
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border-color: rgba(124,58,237,0.7) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.18) !important;
}
/* +/- stepper buttons */
[data-testid="stNumberInput"] button {
    background: rgba(124,58,237,0.2) !important;
    border: none !important;
    color: #C4B5FD !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(124,58,237,0.4) !important;
}

/* ─ PREDICT BUTTON ─ */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 16px 40px !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform .2s, box-shadow .2s !important;
}
.stButton > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,.15) 0%, transparent 60%);
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px rgba(124,58,237,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ─ TABS ─ */
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    background: transparent !important;
    padding-bottom: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: rgba(232,232,255,0.4) !important;
    padding: 12px 24px !important;
    border-radius: 10px 10px 0 0 !important;
    border: none !important;
    background: transparent !important;
    transition: color .2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #A78BFA !important;
    background: rgba(124,58,237,0.1) !important;
    border-bottom: 2px solid #7C3AED !important;
}

/* ─ FILE UPLOADER ─ */
[data-testid="stFileUploader"] section {
    background: rgba(124,58,237,0.05) !important;
    border: 1.5px dashed rgba(124,58,237,0.4) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(124,58,237,0.75) !important;
    background: rgba(124,58,237,0.08) !important;
}
/* Browse files button */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzoneInstructions"] + div button,
[data-testid="stFileUploader"] section button {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.4) !important;
    transition: box-shadow .2s, transform .2s !important;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] section button:hover {
    box-shadow: 0 6px 24px rgba(124,58,237,0.6) !important;
    transform: translateY(-1px) !important;
}
/* Drag & drop text color */
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: rgba(232,232,255,0.55) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg {
    fill: rgba(124,58,237,0.7) !important;
}

/* ─ DATAFRAME ─ */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

/* ─ RESULT CINEMATIC ─ */
.result-outer {
    animation: resultReveal .7s cubic-bezier(.16,1,.3,1) both;
}
@keyframes resultReveal {
    from { opacity: 0; transform: scale(.94) translateY(20px); }
    to   { opacity: 1; transform: scale(1)   translateY(0); }
}
.result-churn {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(185,28,28,0.08) 100%);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 24px;
    padding: 48px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-safe-c {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.08) 100%);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 24px;
    padding: 48px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-glow-red  { position:absolute; top:-60px; right:-60px; width:220px; height:220px; border-radius:50%; background:radial-gradient(circle,rgba(239,68,68,0.2) 0%,transparent 70%); pointer-events:none; }
.result-glow-green{ position:absolute; top:-60px; right:-60px; width:220px; height:220px; border-radius:50%; background:radial-gradient(circle,rgba(16,185,129,0.2) 0%,transparent 70%); pointer-events:none; }
.result-icon { font-size: 52px; margin-bottom: 16px; line-height: 1; }
.result-verdict {
    font-size: clamp(28px, 4vw, 40px);
    font-weight: 900;
    letter-spacing: -0.03em;
    margin-bottom: 8px;
}
.result-prob-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: rgba(232,232,255,0.5);
    margin-bottom: 28px;
}
.gauge-wrap {
    position: relative;
    width: 160px;
    height: 80px;
    margin: 0 auto 16px;
}
.prob-dial {
    width: 160px;
    height: 80px;
}

/* ─ METRIC TILES ─ */
.mtile {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: transform .2s, border-color .2s;
}
.mtile:hover {
    transform: translateY(-3px);
    border-color: rgba(124,58,237,0.35);
}
.mtile-val {
    font-size: 36px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 6px;
}
.mtile-lbl {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(232,232,255,0.4);
}

/* ─ TECH BADGES ─ */
.tbadge {
    display: inline-block;
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 8px;
    padding: 5px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #C4B5FD;
    margin: 4px;
    transition: background .2s, border-color .2s;
}
.tbadge:hover {
    background: rgba(124,58,237,0.25);
    border-color: rgba(124,58,237,0.6);
}

/* ─ RECOMMENDATION BOX ─ */
.rec-box {
    border-radius: 14px;
    padding: 20px 24px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-top: 20px;
}
.rec-icon { font-size: 22px; flex-shrink: 0; margin-top: 1px; }
.rec-title { font-size: 13px; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.02em; }
.rec-text  { font-size: 13px; color: rgba(232,232,255,0.6); line-height: 1.55; }

/* ─ FEATURE CHIP ─ */
.feat-chip {
    display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 12px;
    color: rgba(232,232,255,0.55);
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}

/* ─ MISC STREAMLIT CLEANUPS ─ */
[data-testid="stMarkdownContainer"] p {
    font-family: 'Outfit', sans-serif;
}
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}

/* ─ GLITCH TEXT ─ */
.glitch {
    position: relative;
    color: #fff;
}
.glitch::before,.glitch::after {
    content: attr(data-text);
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
}
.glitch::before {
    color: #7C3AED;
    animation: glitchA 4s 2s infinite;
    clip-path: polygon(0 0,100% 0,100% 35%,0 35%);
}
.glitch::after {
    color: #EC4899;
    animation: glitchB 4s 2s infinite;
    clip-path: polygon(0 65%,100% 65%,100% 100%,0 100%);
}
@keyframes glitchA {
    0%,94%,100% { transform: none; }
    95% { transform: translate(-3px,1px); }
    97% { transform: translate(3px,-1px); }
}
@keyframes glitchB {
    0%,94%,100% { transform: none; }
    95% { transform: translate(3px,-1px); }
    97% { transform: translate(-3px,1px); }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION WITH NEURAL-NET CANVAS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <canvas id="neural-canvas"></canvas>
    <div class="hero-content">
        <div class="hero-badge">Powered by Machine Learning</div>
        <div class="hero-title">
            <span class="glitch" data-text="Churn">Churn</span><br>
            <span class="grad">Intelligence</span>
        </div>
        <div class="hero-sub">
            Predict customer departure before it happens.<br>
            Real-time inference. Calibrated probability scores.
        </div>
        <div class="hero-stats">
            <div>
                <div class="hero-stat-val">19</div>
                <div class="hero-stat-lbl">Features</div>
            </div>
            <div>
                <div class="hero-stat-val">~7K</div>
                <div class="hero-stat-lbl">Training Rows</div>
            </div>
            <div>
                <div class="hero-stat-val">ML</div>
                <div class="hero-stat-lbl">Powered</div>
            </div>
        </div>
        <div class="scroll-cta">
            <span>Begin Analysis</span>
            <div class="scroll-arrow"></div>
        </div>
    </div>
</div>

<script>
(function(){
  const canvas = document.getElementById('neural-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize(){
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const N = 90;
  const nodes = Array.from({length: N}, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    vx: (Math.random() - .5) * .45,
    vy: (Math.random() - .5) * .45,
    r:  Math.random() * 2.2 + 1,
    pulse: Math.random() * Math.PI * 2,
  }));

  const LINK_DIST = 140;
  let frame = 0;

  function draw(){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    frame++;

    // Move nodes
    nodes.forEach(n => {
      n.x += n.vx; n.y += n.vy;
      n.pulse += 0.02;
      if (n.x < 0 || n.x > canvas.width)  n.vx *= -1;
      if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
    });

    // Edges
    for (let i = 0; i < N; i++){
      for (let j = i+1; j < N; j++){
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if (d < LINK_DIST){
          const alpha = (1 - d/LINK_DIST) * 0.22;
          const t = (frame * 0.012 + i * 0.3) % 1;
          // traveling dot
          if (Math.random() < 0.002){
            const tx = nodes[i].x + (nodes[j].x - nodes[i].x) * t;
            const ty = nodes[i].y + (nodes[j].y - nodes[i].y) * t;
            ctx.beginPath();
            ctx.arc(tx, ty, 1.5, 0, Math.PI*2);
            ctx.fillStyle = `rgba(167,139,250,0.8)`;
            ctx.fill();
          }
          const grad = ctx.createLinearGradient(nodes[i].x,nodes[i].y,nodes[j].x,nodes[j].y);
          grad.addColorStop(0, `rgba(124,58,237,${alpha})`);
          grad.addColorStop(.5,`rgba(236,72,153,${alpha*1.5})`);
          grad.addColorStop(1, `rgba(124,58,237,${alpha})`);
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = grad;
          ctx.lineWidth   = .6;
          ctx.stroke();
        }
      }
    }

    // Nodes
    nodes.forEach(n => {
      const glow = ctx.createRadialGradient(n.x,n.y,0,n.x,n.y,n.r*5);
      const a = (.5 + .4*Math.sin(n.pulse));
      glow.addColorStop(0, `rgba(167,139,250,${a*0.9})`);
      glow.addColorStop(1, `rgba(124,58,237,0)`);
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r*5, 0, Math.PI*2);
      ctx.fillStyle = glow;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI*2);
      ctx.fillStyle = `rgba(232,232,255,${a*0.85})`;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    import joblib
    model    = joblib.load("models/churn_model.pkl")
    features = joblib.load("models/features.pkl")
    return model, features

try:
    model, features = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)

try:
    from src.preprocess import preprocess_data
    from src.predict    import predict_customer
    imports_ok = True
except Exception:
    imports_ok = False


# ══════════════════════════════════════════════════════════════════════════════
# APP BODY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="app-body">', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["⚡  Single Prediction", "📊  Batch Analysis", "🧬  About"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SINGLE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    if not model_loaded:
        st.warning(f"Model not found — running in **demo mode**. `{model_error}`")

    # DEMOGRAPHICS
    st.markdown('<div class="sec-head"><div class="sec-pip"></div><div class="sec-title">Demographics</div></div>', unsafe_allow_html=True)
    with st.container():
        c1,c2,c3,c4 = st.columns(4)
        with c1: gender     = st.selectbox("Gender",          ["Male","Female"])
        with c2: senior     = st.selectbox("Senior Citizen",  ["No","Yes"])
        with c3: partner    = st.selectbox("Partner",         ["Yes","No"])
        with c4: dependents = st.selectbox("Dependents",      ["Yes","No"])

    # ACCOUNT
    st.markdown('<div class="sec-head"><div class="sec-pip"></div><div class="sec-title">Account & Billing</div></div>', unsafe_allow_html=True)
    with st.container():
        c1,c2,c3 = st.columns(3)
        with c1: tenure  = st.number_input("Tenure (months)",    0, 120, 12, step=1)
        with c2: monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        with c3: total   = st.number_input("Total Charges ($)",   0.0, 10000.0, 780.0, step=10.0)

        c1,c2,c3 = st.columns(3)
        with c1: contract  = st.selectbox("Contract",        ["Month-to-month","One year","Two year"])
        with c2: payment   = st.selectbox("Payment Method",  ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        with c3: paperless = st.selectbox("Paperless Billing",["Yes","No"])

    # SERVICES
    st.markdown('<div class="sec-head"><div class="sec-pip"></div><div class="sec-title">Services</div></div>', unsafe_allow_html=True)
    with st.container():
        c1,c2,c3,c4 = st.columns(4)
        with c1: phone_svc   = st.selectbox("Phone Service",    ["Yes","No"])
        with c2: multi_lines = st.selectbox("Multiple Lines",   ["No","Yes","No phone service"])
        with c3: internet    = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        with c4: online_sec  = st.selectbox("Online Security",  ["No","Yes","No internet service"])

        c1,c2,c3,c4 = st.columns(4)
        with c1: backup      = st.selectbox("Online Backup",    ["No","Yes","No internet service"])
        with c2: device_prot = st.selectbox("Device Protection",["No","Yes","No internet service"])
        with c3: tech_supp   = st.selectbox("Tech Support",     ["No","Yes","No internet service"])
        with c4: stream_tv   = st.selectbox("Streaming TV",     ["No","Yes","No internet service"])

        c1,_,_ = st.columns(3)
        with c1: stream_mov = st.selectbox("Streaming Movies",["No","Yes","No internet service"])

    st.markdown("<br>", unsafe_allow_html=True)

    # BUTTON
    _,bcol,_ = st.columns([1.5, 1, 1.5])
    with bcol:
        run = st.button("⚡  Analyze Customer")

    if run:
        row = {
            "gender"          : gender,
            "SeniorCitizen"   : 1 if senior=="Yes" else 0,
            "Partner"         : partner,
            "Dependents"      : dependents,
            "tenure"          : tenure,
            "PhoneService"    : phone_svc,
            "MultipleLines"   : multi_lines,
            "InternetService" : internet,
            "OnlineSecurity"  : online_sec,
            "OnlineBackup"    : backup,
            "DeviceProtection": device_prot,
            "TechSupport"     : tech_supp,
            "StreamingTV"     : stream_tv,
            "StreamingMovies" : stream_mov,
            "Contract"        : contract,
            "PaperlessBilling": paperless,
            "PaymentMethod"   : payment,
            "MonthlyCharges"  : monthly,
            "TotalCharges"    : str(total),
        }
        df_in = pd.DataFrame([row])

        with st.spinner("Running inference…"):
            if not model_loaded or not imports_ok:
                import time; time.sleep(0.6)
                prob = float(np.random.uniform(0.1, 0.9))
                pred = int(prob > 0.5)
            else:
                pa, pb = predict_customer(df_in)
                pred = int(pa[0]); prob = float(pb[0])

        pct = round(prob * 100, 1)
        st.markdown('<div class="sec-head"><div class="sec-pip"></div><div class="sec-title">Prediction Result</div></div>', unsafe_allow_html=True)

        lcol, rcol = st.columns([1.1, 0.9])

        with lcol:
            st.markdown('<div class="result-outer">', unsafe_allow_html=True)
            if pred == 1:
                st.markdown(f"""
                <div class="result-churn">
                    <div class="result-glow-red"></div>
                    <div class="result-icon">🔴</div>
                    <div class="result-verdict" style="color:#FCA5A5">Likely to Churn</div>
                    <div class="result-prob-text">CHURN PROBABILITY → {pct}%</div>
                    <div style="height:10px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden">
                        <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#EF4444,#F97316);border-radius:99px;
                                    transition:width 1s cubic-bezier(.4,0,.2,1);animation:barIn 1.2s .2s cubic-bezier(.4,0,.2,1) both">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe-c">
                    <div class="result-glow-green"></div>
                    <div class="result-icon">🟢</div>
                    <div class="result-verdict" style="color:#6EE7B7">Likely to Stay</div>
                    <div class="result-prob-text">CHURN PROBABILITY → {pct}%</div>
                    <div style="height:10px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden">
                        <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#10B981,#34D399);border-radius:99px">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with rcol:
            # Semicircle gauge via SVG
            angle   = prob * 180
            rad     = (180 - angle) * 3.14159 / 180
            gx      = 100 + 75 * (-1 if prob > 0.5 else 1) * abs(round(75 * (1 - 2*abs(prob-0.5)) if prob != 0.5 else 0, 1))
            # simple arc calculation
            import math
            end_rad = math.radians(180 - prob * 180)
            ex = 100 + 75 * math.cos(end_rad)
            ey = 90  - 75 * math.sin(end_rad)
            fill_color = f"{'#EF4444' if pred==1 else '#10B981'}"
            st.markdown(f"""
            <div class="glass" style="padding:32px 24px;text-align:center">
                <svg viewBox="0 0 200 105" width="100%" style="display:block;margin:0 auto 8px">
                  <defs>
                    <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%"   stop-color="#10B981"/>
                      <stop offset="50%"  stop-color="#F59E0B"/>
                      <stop offset="100%" stop-color="#EF4444"/>
                    </linearGradient>
                  </defs>
                  <!-- Background arc -->
                  <path d="M 25 90 A 75 75 0 0 1 175 90"
                        fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="10" stroke-linecap="round"/>
                  <!-- Filled arc -->
                  <path d="M 25 90 A 75 75 0 0 1 {round(ex,1)} {round(ey,1)}"
                        fill="none" stroke="url(#arcGrad)" stroke-width="10" stroke-linecap="round"
                        style="filter:drop-shadow(0 0 6px {fill_color})"/>
                  <!-- Center needle -->
                  <line x1="100" y1="90"
                        x2="{round(100 + 58*math.cos(end_rad), 1)}"
                        y2="{round(90  - 58*math.sin(end_rad), 1)}"
                        stroke="{fill_color}" stroke-width="2.5" stroke-linecap="round"/>
                  <circle cx="100" cy="90" r="5" fill="{fill_color}" style="filter:drop-shadow(0 0 4px {fill_color})"/>
                  <!-- Labels -->
                  <text x="22"  y="104" fill="rgba(255,255,255,0.3)" font-size="9" font-family="JetBrains Mono">0%</text>
                  <text x="168" y="104" fill="rgba(255,255,255,0.3)" font-size="9" font-family="JetBrains Mono">100%</text>
                  <text x="97"  y="72"  fill="rgba(255,255,255,0.2)" font-size="8" font-family="JetBrains Mono" text-anchor="middle">RISK</text>
                </svg>
                <div style="font-size:42px;font-weight:900;color:{fill_color};letter-spacing:-0.04em;line-height:1">{pct}%</div>
                <div style="font-size:11px;letter-spacing:0.1em;color:rgba(232,232,255,0.35);margin-top:6px;text-transform:uppercase;font-family:'JetBrains Mono',monospace">Churn Probability</div>
                <div style="margin-top:24px;display:grid;grid-template-columns:1fr 1fr;gap:12px">
                    <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);border-radius:10px;padding:14px;text-align:center">
                        <div style="font-size:20px;font-weight:800;color:#34D399">{round((1-prob)*100,1)}%</div>
                        <div style="font-size:10px;color:rgba(232,232,255,0.35);margin-top:3px;font-family:'JetBrains Mono',monospace;letter-spacing:.06em">RETENTION</div>
                    </div>
                    <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.25);border-radius:10px;padding:14px;text-align:center">
                        <div style="font-size:20px;font-weight:800;color:#FCA5A5">{pct}%</div>
                        <div style="font-size:10px;color:rgba(232,232,255,0.35);margin-top:3px;font-family:'JetBrains Mono',monospace;letter-spacing:.06em">CHURN RISK</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Recommendation
        if pred == 1 and prob > 0.75:
            rec_bg, rec_border, rec_icon, rec_title, rec_msg = (
                "rgba(239,68,68,0.08)", "rgba(239,68,68,0.3)", "🚨",
                "High-Priority Intervention",
                "This customer is at critical risk. Assign a dedicated retention specialist, offer a personalized discount, or propose a contract upgrade immediately."
            )
        elif pred == 1:
            rec_bg, rec_border, rec_icon, rec_title, rec_msg = (
                "rgba(245,158,11,0.08)", "rgba(245,158,11,0.3)", "⚠️",
                "Proactive Retention Needed",
                "Moderate churn risk detected. Enroll this customer in a loyalty program or schedule a proactive check-in call within the next 7 days."
            )
        else:
            rec_bg, rec_border, rec_icon, rec_title, rec_msg = (
                "rgba(16,185,129,0.08)", "rgba(16,185,129,0.3)", "✅",
                "Customer is Stable",
                "Strong retention signals detected. Standard engagement cadence is sufficient — consider a satisfaction survey to maintain loyalty."
            )

        st.markdown(f"""
        <div class="rec-box" style="background:{rec_bg};border:1px solid {rec_border}">
            <div class="rec-icon">{rec_icon}</div>
            <div>
                <div class="rec-title">{rec_title}</div>
                <div class="rec-text">{rec_msg}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BATCH
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    st.markdown('<div class="sec-head"><div class="sec-pip"></div><div class="sec-title">Upload Customer CSV</div></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:rgba(232,232,255,0.4);font-size:13px;margin-bottom:16px">Upload a CSV with the same schema used during training. CustomerID column is ignored automatically.</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop CSV here", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.markdown(f'<p style="color:rgba(232,232,255,0.4);font-size:13px;margin:10px 0">{len(df_raw):,} customer records detected</p>', unsafe_allow_html=True)

        if st.button("⚡  Run Batch Predictions"):
            with st.spinner("Processing all customers…"):
                if not model_loaded or not imports_ok:
                    import time; time.sleep(0.8)
                    df_result = df_raw.copy()
                    df_result["Churn_Prediction"] = np.random.randint(0, 2, len(df_raw))
                    df_result["Churn_Probability"] = np.random.uniform(0.05, 0.95, len(df_raw)).round(3)
                else:
                    preds, probs = predict_customer(df_raw.copy())
                    df_result = df_raw.copy()
                    df_result["Churn_Prediction"] = preds
                    df_result["Churn_Probability"] = probs.round(3)

            n_churn  = int(df_result["Churn_Prediction"].sum())
            n_retain = len(df_result) - n_churn
            avg_risk = df_result["Churn_Probability"].mean()
            hi_risk  = int((df_result["Churn_Probability"] > 0.75).sum())

            st.markdown('<div class="sec-head"><div class="sec-pip"></div><div class="sec-title">Summary</div></div>', unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            tiles = [
                (f"{len(df_result):,}", "Total Customers", "#A78BFA"),
                (f"{n_churn:,}",        "At-Risk",         "#FCA5A5"),
                (f"{n_retain:,}",       "Retained",        "#6EE7B7"),
                (f"{hi_risk:,}",        "High Priority",   "#FCD34D"),
            ]
            for col,(val,lbl,color) in zip([m1,m2,m3,m4],tiles):
                with col:
                    st.markdown(f"""
                    <div class="mtile">
                        <div class="mtile-val" style="color:{color}">{val}</div>
                        <div class="mtile-lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            # Risk distribution bar
            churn_pct = n_churn / max(len(df_result),1) * 100
            st.markdown(f"""
            <div style="margin:28px 0 8px">
                <div style="display:flex;justify-content:space-between;font-size:11px;color:rgba(232,232,255,0.4);margin-bottom:8px;font-family:'JetBrains Mono',monospace">
                    <span>RETAINED — {round(100-churn_pct,1)}%</span>
                    <span>AT RISK — {round(churn_pct,1)}%</span>
                </div>
                <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;display:flex">
                    <div style="width:{100-churn_pct}%;background:linear-gradient(90deg,#10B981,#059669)"></div>
                    <div style="width:{churn_pct}%;background:linear-gradient(90deg,#F59E0B,#EF4444)"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sec-head"><div class="sec-pip"></div><div class="sec-title">Full Results Table</div></div>', unsafe_allow_html=True)
            st.dataframe(
                df_result.style.background_gradient(subset=["Churn_Probability"], cmap="RdYlGn_r", vmin=0, vmax=1),
                use_container_width=True,
                height=420,
            )

            st.download_button(
                "⬇  Download Predictions CSV",
                df_result.to_csv(index=False).encode(),
                "churn_predictions.csv",
                "text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:

    st.markdown('<div class="sec-head"><div class="sec-pip"></div><div class="sec-title">Project</div></div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass">
            <div style="font-size:13px;font-weight:700;color:rgba(232,232,255,0.5);letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px">What is ChurnLens?</div>
            <div style="font-size:14px;color:rgba(232,232,255,0.65);line-height:1.75">
                An end-to-end machine learning pipeline that predicts customer churn for telecom companies.
                Takes 19 subscriber features — demographics, account details, and services — and returns
                a binary prediction plus a calibrated probability score.<br><br>
                Built with scikit-learn, preprocessed with pandas, and deployed with Streamlit.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass">
            <div style="font-size:13px;font-weight:700;color:rgba(232,232,255,0.5);letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px">Tech Stack</div>
            <div>
                <span class="tbadge">Python 3.x</span>
                <span class="tbadge">Scikit-learn</span>
                <span class="tbadge">Pandas</span>
                <span class="tbadge">NumPy</span>
                <span class="tbadge">Joblib</span>
                <span class="tbadge">Streamlit</span>
                <span class="tbadge">Matplotlib</span>
            </div>
            <div style="margin-top:20px">
                <div style="font-size:11px;color:rgba(232,232,255,0.3);margin-bottom:10px;font-family:'JetBrains Mono',monospace;letter-spacing:.06em">MODEL FEATURES</div>
        """ + "".join([
            f'<span class="feat-chip">{f}</span>' for f in [
                "gender","SeniorCitizen","Partner","Dependents","tenure",
                "PhoneService","MultipleLines","InternetService","OnlineSecurity",
                "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
                "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
                "MonthlyCharges","TotalCharges"
            ]
        ]) + """
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><div class="sec-pip"></div><div class="sec-title">Pipeline Overview</div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
        <div style="display:flex;align-items:center;gap:0;flex-wrap:wrap;justify-content:center;padding:8px 0">
    """ + "".join([
        f"""<div style="display:flex;align-items:center;gap:0">
              <div style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.35);border-radius:10px;
                          padding:12px 18px;font-size:13px;font-weight:600;color:#C4B5FD;white-space:nowrap">{step}</div>
              <div style="color:rgba(124,58,237,0.5);font-size:18px;margin:0 6px">{'→' if i<4 else ''}</div>
           </div>"""
        for i, step in enumerate(["Raw CSV", "Preprocess", "Feature Align", "ML Model", "Probability Score"])
    ]) + """
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # app-body

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 0 32px;border-top:1px solid rgba(255,255,255,0.06);
            margin-top:20px;color:rgba(232,232,255,0.2);font-size:12px;font-family:'JetBrains Mono',monospace;
            letter-spacing:0.08em">
    CHURNLENS &nbsp;·&nbsp; CUSTOMER INTELLIGENCE &nbsp;·&nbsp; BUILT WITH STREAMLIT
</div>
""", unsafe_allow_html=True)