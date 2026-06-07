import gc
import os
from pathlib import Path
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="StructInsight AI",
    page_icon="🏗️",
    layout="wide",
)

# =========================================================
# CUSTOM CSS — cinematic engineering aesthetic v3
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');


/* ══════════════════════════════════════════════════════════
   CSS CUSTOM PROPERTIES
══════════════════════════════════════════════════════════ */
:root {
    --c-bg:        #03060f;
    --c-surface:   #070d1a;
    --c-card:      #0a1020;
    --c-border:    #0f1d35;
    --c-border2:   #162038;
    --c-sky:       #38bdf8;
    --c-indigo:    #818cf8;
    --c-amber:     #fbbf24;
    --c-emerald:   #34d399;
    --c-rose:      #fb7185;
    --c-text:      #e2e8f0;
    --c-muted:     #64748b;
    --c-faint:     #1e293b;
    --font-display: 'Syne', sans-serif;
    --font-body:    'Space Grotesk', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

/* ══════════════════════════════════════════════════════════
   BASE
══════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background: var(--c-bg) !important;
    color: var(--c-text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1320px; }
::-webkit-scrollbar { width: 5px; background: var(--c-bg); }
::-webkit-scrollbar-thumb { background: var(--c-border2); border-radius: 4px; }

/* ══════════════════════════════════════════════════════════
   KEYFRAME LIBRARY
══════════════════════════════════════════════════════════ */
@keyframes fadeUp   { from { opacity:0; transform:translateY(24px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeIn   { from { opacity:0; } to { opacity:1; } }
@keyframes slideRight { from { transform:scaleX(0); } to { transform:scaleX(1); } }
@keyframes scanline { 0%{top:-100%} 100%{top:200%} }
@keyframes orbFloat { 0%,100%{transform:translate(0,0) scale(1);} 33%{transform:translate(20px,-30px) scale(1.04);} 66%{transform:translate(-15px,20px) scale(0.97);} }
@keyframes gridPan  { from{background-position:0 0} to{background-position:60px 60px} }
@keyframes shimmer  { 0%{background-position:-400px 0} 100%{background-position:400px 0} }
@keyframes rotateSlow { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
@keyframes pulseDot { 0%,100%{opacity:1;box-shadow:0 0 0 0 currentColor;} 50%{opacity:.6;box-shadow:0 0 0 6px transparent;} }
@keyframes borderGlow { 0%,100%{box-shadow:0 0 0 0 rgba(56,189,248,0);} 50%{box-shadow:0 0 24px 2px rgba(56,189,248,0.15);} }
@keyframes typewriter { from{width:0} to{width:100%} }
@keyframes cursorBlink { 0%,100%{border-right-color:var(--c-sky);} 50%{border-right-color:transparent;} }
@keyframes floatY    { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
@keyframes countUp   { from{opacity:0;transform:scale(0.7);} to{opacity:1;transform:scale(1);} }

/* ══════════════════════════════════════════════════════════
   ANIMATED BACKGROUND CANVAS (page-level)
══════════════════════════════════════════════════════════ */
.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(56,189,248,0.055) 0%, transparent 55%),
        radial-gradient(ellipse 60% 70% at 90% 80%, rgba(129,140,248,0.045) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 55%, rgba(251,191,36,0.025) 0%, transparent 60%),
        var(--c-bg) !important;
}

/* ══════════════════════════════════════════════════════════
   HERO
══════════════════════════════════════════════════════════ */
.si-hero {
    position: relative;
    overflow: hidden;
    padding: 4rem 3.5rem 3.5rem;
    margin: -1rem -1rem 3rem;
    background: var(--c-surface);
    border-bottom: 1px solid rgba(56,189,248,0.1);
}

/* animated grid */
.si-hero::before {
    content: '';
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(56,189,248,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56,189,248,0.04) 1px, transparent 1px);
    background-size: 60px 60px;
    animation: gridPan 12s linear infinite;
}

/* scanline sweep */
.si-hero-scan {
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.4), transparent);
    animation: scanline 5s linear infinite;
    pointer-events: none;
}

/* floating orbs */
.si-hero-orb1 {
    position: absolute;
    width: 480px; height: 480px;
    top: -160px; left: -80px;
    background: radial-gradient(circle, rgba(56,189,248,0.10) 0%, transparent 65%);
    border-radius: 50%;
    animation: orbFloat 8s ease-in-out infinite;
    pointer-events: none;
}
.si-hero-orb2 {
    position: absolute;
    width: 360px; height: 360px;
    bottom: -120px; right: 80px;
    background: radial-gradient(circle, rgba(129,140,248,0.09) 0%, transparent 65%);
    border-radius: 50%;
    animation: orbFloat 10s ease-in-out infinite reverse;
    pointer-events: none;
}

/* spinning ring decoration */
.si-hero-ring {
    position: absolute;
    top: 30px; right: 80px;
    width: 180px; height: 180px;
    border: 1px solid rgba(56,189,248,0.08);
    border-top-color: rgba(56,189,248,0.35);
    border-radius: 50%;
    animation: rotateSlow 12s linear infinite;
    pointer-events: none;
}
.si-hero-ring::after {
    content: '';
    position: absolute;
    inset: 20px;
    border: 1px solid rgba(129,140,248,0.06);
    border-bottom-color: rgba(129,140,248,0.3);
    border-radius: 50%;
    animation: rotateSlow 8s linear infinite reverse;
}

.si-hero-inner {
    position: relative; z-index: 2;
    animation: fadeUp 0.9s cubic-bezier(0.16,1,0.3,1) both;
}

.si-hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.25);
    color: var(--c-amber);
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.32rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.4rem;
    animation: fadeIn 1s 0.2s both;
}
.si-hero-tag-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--c-amber);
    animation: pulseDot 2s ease-in-out infinite;
}

.si-hero h1 {
    font-family: var(--font-display);
    color: var(--c-text);
    font-size: 3.8rem;
    font-weight: 800;
    letter-spacing: -2.5px;
    margin: 0 0 0.8rem;
    line-height: 1.0;
    animation: fadeUp 0.8s 0.1s cubic-bezier(0.16,1,0.3,1) both;
}
.si-hero h1 .grad {
    background: linear-gradient(120deg, var(--c-sky) 0%, var(--c-indigo) 50%, var(--c-emerald) 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
}

.si-hero-sub {
    color: var(--c-muted);
    font-size: 0.97rem;
    font-weight: 400;
    margin: 0;
    max-width: 560px;
    line-height: 1.8;
    animation: fadeUp 0.8s 0.2s cubic-bezier(0.16,1,0.3,1) both;
}

/* animated underline on hero title */
.si-hero-underline {
    height: 2px;
    width: 120px;
    background: linear-gradient(90deg, var(--c-sky), var(--c-indigo), transparent);
    border-radius: 2px;
    margin: 1rem 0 0;
    transform-origin: left;
    animation: slideRight 1s 0.5s cubic-bezier(0.16,1,0.3,1) both;
}

.si-hero-badges {
    display: flex; gap: 0.5rem;
    margin-top: 1.8rem; flex-wrap: wrap;
    animation: fadeUp 0.8s 0.35s cubic-bezier(0.16,1,0.3,1) both;
}
.si-badge {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    color: #94a3b8;
    font-family: var(--font-mono);
    font-size: 0.67rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    padding: 0.32rem 0.9rem;
    border-radius: 6px;
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1);
    cursor: default;
}
.si-badge:hover {
    background: rgba(56,189,248,0.1);
    border-color: rgba(56,189,248,0.3);
    color: var(--c-sky);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(56,189,248,0.15);
}

/* stat strip inside hero */
.si-hero-stats {
    display: flex; gap: 2.5rem;
    margin-top: 2.5rem; flex-wrap: wrap;
    padding-top: 2rem;
    border-top: 1px solid var(--c-border);
    animation: fadeIn 1s 0.55s both;
}
.si-hero-stat-num {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--c-sky), var(--c-indigo));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.si-hero-stat-lbl {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--c-muted);
}

/* ══════════════════════════════════════════════════════════
   SECTION HEADERS
══════════════════════════════════════════════════════════ */
.si-section {
    display: flex; align-items: center; gap: 1rem;
    margin: 3rem 0 1.5rem;
    animation: fadeIn 0.6s both;
}
.si-section-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.3), transparent);
    animation: slideRight 0.8s cubic-bezier(0.16,1,0.3,1) both;
    transform-origin: left;
}
.si-section-label {
    font-family: var(--font-mono);
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--c-sky); white-space: nowrap;
}
.si-section-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(129,140,248,0.08));
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; flex-shrink: 0;
    box-shadow: 0 0 20px rgba(56,189,248,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: borderGlow 3s ease-in-out infinite;
}

/* ══════════════════════════════════════════════════════════
   UPLOAD ZONE
══════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(7,13,26,0.95), rgba(10,16,32,0.9)) !important;
    border: 1.5px dashed rgba(56,189,248,0.22) !important;
    border-radius: 20px !important;
    padding: 1.75rem !important;
    transition: all 0.4s cubic-bezier(0.34,1.56,0.64,1) !important;
    box-shadow: inset 0 0 60px rgba(56,189,248,0.025), 0 4px 24px rgba(0,0,0,0.4) !important;
    position: relative;
}
[data-testid="stFileUploader"]::after {
    content: '';
    position: absolute; inset: 0;
    border-radius: 20px;
    background: radial-gradient(ellipse 60% 40% at 50% 50%, rgba(56,189,248,0.03), transparent);
    pointer-events: none;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.5) !important;
    background: linear-gradient(135deg, rgba(56,189,248,0.05), rgba(129,140,248,0.03)) !important;
    box-shadow: inset 0 0 60px rgba(56,189,248,0.06), 0 0 40px rgba(56,189,248,0.1) !important;
    transform: scale(1.005) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: #475569 !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #0c4a6e, #0369a1) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    color: #fff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(3,105,161,0.4) !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(3,105,161,0.6) !important;
}

/* ══════════════════════════════════════════════════════════
   METRIC CARDS
══════════════════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: linear-gradient(160deg, #0a1020 0%, #070d1a 100%);
    border: 1px solid var(--c-border2);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    border-bottom: 2px solid var(--c-sky);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(56,189,248,0.04);
    transition: all 0.35s cubic-bezier(0.34,1.56,0.64,1);
    position: relative; overflow: hidden;
    animation: fadeUp 0.6s cubic-bezier(0.16,1,0.3,1) both;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute; top: 0; left: -100%; right: -100%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.5), transparent);
    animation: shimmer 3s linear infinite;
}
[data-testid="metric-container"]::after {
    content: '';
    position: absolute; bottom: 0; right: 0;
    width: 80px; height: 80px;
    background: radial-gradient(circle, rgba(56,189,248,0.06), transparent 70%);
    pointer-events: none;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 16px 48px rgba(0,0,0,0.6), 0 0 30px rgba(56,189,248,0.12);
    border-bottom-color: var(--c-indigo);
    border-color: rgba(56,189,248,0.25);
}
[data-testid="stMetricLabel"] {
    color: #334155 !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    font-variant-numeric: tabular-nums !important;
    font-family: var(--font-mono) !important;
    animation: countUp 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
}

/* ══════════════════════════════════════════════════════════
   IMAGE PANELS
══════════════════════════════════════════════════════════ */
.si-img-panel {
    background: var(--c-card);
    border: 1px solid var(--c-border2);
    border-radius: 20px;
    padding: 1.4rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
    transition: all 0.4s cubic-bezier(0.16,1,0.3,1);
    position: relative; overflow: hidden;
    animation: fadeUp 0.6s 0.1s cubic-bezier(0.16,1,0.3,1) both;
}
.si-img-panel::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent);
}
.si-img-panel:hover {
    border-color: rgba(56,189,248,0.3);
    box-shadow: 0 12px 60px rgba(0,0,0,0.6), 0 0 40px rgba(56,189,248,0.07);
    transform: translateY(-2px);
}
.si-img-label {
    font-family: var(--font-mono);
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--c-sky); margin-bottom: 0.85rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.si-img-label::before {
    content: '';
    display: inline-block;
    width: 3px; height: 14px;
    background: linear-gradient(180deg, var(--c-sky), var(--c-indigo));
    border-radius: 2px;
}
[data-testid="stImage"] img { border-radius: 12px; }

/* ══════════════════════════════════════════════════════════
   SEVERITY CARD
══════════════════════════════════════════════════════════ */
.si-severity-wrap {
    background: linear-gradient(160deg, #0a1020, #070d1a);
    border: 1px solid var(--c-border2);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
    margin-bottom: 1rem;
    position: relative; overflow: hidden;
    animation: fadeUp 0.6s 0.2s cubic-bezier(0.16,1,0.3,1) both;
}
.si-severity-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 70% 60% at 20% 80%, rgba(56,189,248,0.04), transparent 60%);
    pointer-events: none;
}
.si-sev-header {
    font-family: var(--font-mono);
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: #334155; margin-bottom: 0.7rem;
}
.si-sev-badge {
    display: inline-flex; align-items: center; gap: 0.55rem;
    padding: 0.45rem 1.2rem; border-radius: 999px;
    font-family: var(--font-mono);
    font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 1.1rem;
    animation: fadeIn 0.5s 0.4s both;
}
.si-sev-value {
    font-family: var(--font-display);
    font-size: 3.2rem; font-weight: 800;
    font-variant-numeric: tabular-nums; line-height: 1;
    margin-bottom: 0.3rem;
    animation: countUp 0.6s 0.3s cubic-bezier(0.34,1.56,0.64,1) both;
}
.si-sev-unit { font-size: 1.1rem; font-weight: 400; color: var(--c-muted); }
.si-info-block {
    background: rgba(7,13,26,0.7);
    border: 1px solid var(--c-border);
    border-left: 3px solid var(--c-border2);
    border-radius: 12px;
    padding: 1.1rem 1.3rem; margin-bottom: 0.75rem;
    font-size: 0.875rem; color: #94a3b8; line-height: 1.75;
    transition: border-left-color 0.3s, box-shadow 0.3s;
    animation: fadeUp 0.5s 0.3s cubic-bezier(0.16,1,0.3,1) both;
}
.si-info-block:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
.si-info-block strong {
    display: block; margin-bottom: 0.3rem;
    font-family: var(--font-mono);
    font-size: 0.6rem; text-transform: uppercase;
    letter-spacing: 0.14em; font-weight: 700; color: #cbd5e1;
}

/* ══════════════════════════════════════════════════════════
   ABOUT & TEAM CARDS
══════════════════════════════════════════════════════════ */
.si-card {
    background: linear-gradient(160deg, var(--c-card) 0%, var(--c-surface) 100%);
    border: 1px solid var(--c-border2);
    border-radius: 20px; padding: 2rem; height: 100%;
    box-shadow: 0 8px 40px rgba(0,0,0,0.45); margin-bottom: 1rem;
    transition: all 0.4s cubic-bezier(0.16,1,0.3,1);
    position: relative; overflow: hidden;
    animation: fadeUp 0.6s cubic-bezier(0.16,1,0.3,1) both;
}
.si-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.25), rgba(129,140,248,0.2), transparent);
}
.si-card::after {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(56,189,248,0.04), transparent 70%);
    transition: opacity 0.3s;
    pointer-events: none;
}
.si-card:hover {
    border-color: rgba(56,189,248,0.22);
    transform: translateY(-5px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 40px rgba(56,189,248,0.08);
}
.si-card-tag {
    font-family: var(--font-mono);
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--c-sky); margin-bottom: 0.6rem;
}
.si-card h3 { font-family: var(--font-display); margin: 0 0 0.9rem; color: #f1f5f9; font-size: 1.1rem; font-weight: 700; }
.si-card p  { margin: 0; color: #64748b; font-size: 0.875rem; line-height: 1.8; }
.si-cap-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1.1rem; }
.si-cap {
    background: rgba(3,105,161,0.12);
    border: 1px solid rgba(56,189,248,0.18);
    color: var(--c-sky); font-size: 0.67rem; font-weight: 600;
    padding: 0.22rem 0.75rem; border-radius: 6px; letter-spacing: 0.04em;
    transition: all 0.2s;
}
.si-cap:hover {
    background: rgba(56,189,248,0.15);
    border-color: rgba(56,189,248,0.35);
    transform: translateY(-1px);
}

.si-ref-row { display: flex; align-items: center; gap: 0.8rem; padding: 0.7rem 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.04); transition: all 0.2s; border-radius: 6px; }
.si-ref-row:last-child { border-bottom: none; }
.si-ref-row:hover { background: rgba(255,255,255,0.025); padding-left: 0.8rem; }
.si-ref-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.si-ref-range { font-family: var(--font-mono); font-size: 0.72rem; color: #334155; margin-left: auto; font-variant-numeric: tabular-nums; }
.si-ref-label { font-size: 0.82rem; font-weight: 600; color: #e2e8f0; }
.si-ref-action { font-size: 0.7rem; color: var(--c-muted); }

.si-team-role { font-family: var(--font-mono); font-size: 0.6rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: var(--c-sky); margin-bottom: 0.4rem; }
.si-team-name { font-family: var(--font-display); font-size: 1.1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.4rem; }
.si-team-detail { font-size: 0.78rem; color: var(--c-muted); line-height: 1.7; }
.si-team-divider { width: 32px; height: 2px; background: linear-gradient(90deg, var(--c-sky), var(--c-indigo), transparent); margin: 0.75rem 0; border-radius: 2px; }

/* ══════════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #040810 !important;
    border-right: 1px solid rgba(56,189,248,0.07) !important;
}
[data-testid="stSidebarContent"] { padding-top: 0 !important; }
[data-testid="stSidebar"] > div > div > div > div:first-child::before {
    content: '⬡  STRUCTINSIGHT';
    display: block;
    font-family: var(--font-mono);
    font-size: 0.58rem; font-weight: 700;
    letter-spacing: 0.22em; color: var(--c-sky);
    padding: 1.75rem 1.25rem 1.25rem;
    border-bottom: 1px solid rgba(56,189,248,0.08);
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, var(--c-sky), var(--c-indigo));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider span,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label { color: #334155 !important; font-size: 0.75rem !important; }
[data-testid="stSidebar"] h1 {
    font-family: var(--font-mono) !important;
    font-size: 0.56rem !important; font-weight: 700 !important;
    letter-spacing: 0.2em !important; text-transform: uppercase !important;
    padding: 1.4rem 0 0.55rem !important;
    border-bottom: 1px solid rgba(56,189,248,0.07) !important;
    margin-bottom: 0.9rem !important;
    background: linear-gradient(90deg, var(--c-sky), var(--c-indigo)) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
[data-testid="stSidebar"] [data-baseweb="input"] {
    background: rgba(10,16,32,0.9) !important;
    border-color: var(--c-border2) !important;
    color: #94a3b8 !important; border-radius: 9px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--c-sky) !important;
    box-shadow: 0 0 14px rgba(56,189,248,0.6) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:nth-child(2) {
    background: linear-gradient(90deg, var(--c-sky), var(--c-indigo)) !important;
}
[data-testid="stSidebar"] .stCaption { color: #162038 !important; font-size: 0.68rem !important; }

/* ══════════════════════════════════════════════════════════
   STATUS & ALERTS
══════════════════════════════════════════════════════════ */
[data-testid="stStatusWidget"] {
    background: var(--c-card) !important;
    border: 1px solid var(--c-border2) !important;
    border-radius: 12px !important;
}
.si-error {
    background: rgba(251,113,133,0.06);
    border: 1px solid rgba(251,113,133,0.2);
    border-left: 3px solid var(--c-rose);
    border-radius: 12px; padding: 1.2rem 1.5rem;
    color: #fda4af; font-size: 0.875rem;
    animation: fadeIn 0.4s both;
}
[data-testid="stAlert"] {
    background: rgba(7,13,26,0.85) !important;
    border-radius: 12px !important;
    border: 1px solid var(--c-border2) !important;
}

/* ══════════════════════════════════════════════════════════
   DIVIDER & FOOTER
══════════════════════════════════════════════════════════ */
hr { border: none; border-top: 1px solid var(--c-border); margin: 3rem 0; }

.si-footer {
    text-align: center;
    padding: 3rem 0 2rem;
    color: var(--c-faint);
    font-family: var(--font-mono);
    font-size: 0.68rem; letter-spacing: 0.1em;
    border-top: 1px solid var(--c-border); margin-top: 1rem;
    position: relative;
}
.si-footer::before {
    content: '';
    position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 120px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--c-sky), transparent);
}
.si-footer span { color: var(--c-sky); }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def safe_secret_get(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# =========================================================
# CONFIG
# =========================================================
GITHUB_TOKEN           = safe_secret_get("github_token")
PRIVATE_REPO           = safe_secret_get("private_repo")
CRACK_MODEL_PATH       = safe_secret_get("crack_model_path", "Crackdetection_model.pt")
COIN_MODEL_PATH        = safe_secret_get("coin_model_path",  "Coindetection_model.pt")
DEFAULT_COIN_DIAMETER_MM = float(safe_secret_get("coin_diameter_mm", 18.5))
CACHE_DIR = Path("models_cache")
CACHE_DIR.mkdir(exist_ok=True)

# =========================================================
# SIDEBAR SETTINGS
# =========================================================
st.sidebar.title("Detection Settings")
coin_diameter_mm = st.sidebar.number_input(
    "Coin Diameter (mm)",
    min_value=1.0, max_value=100.0,
    value=DEFAULT_COIN_DIAMETER_MM, step=0.1,
)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05, max_value=1.0, value=0.25, step=0.05,
)
iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.1, max_value=1.0, value=0.45, step=0.05,
)

st.sidebar.title("Width Calibration")
calibration_factor = st.sidebar.slider(
    "Calibration Factor",
    min_value=0.10, max_value=3.00, value=1.00, step=0.01,
)
st.sidebar.caption("Final Width = Measured Width × Calibration Factor")

st.sidebar.title("Scale Calibration")
use_manual_calibration = st.sidebar.checkbox("Use Manual Calibration", value=False)
manual_mm_per_pixel = st.sidebar.number_input(
    "Manual mm/pixel",
    min_value=0.0001, max_value=10.0,
    value=0.05, step=0.001, format="%.4f",
)

# =========================================================
# DOWNLOAD MODEL FROM PRIVATE GITHUB
# =========================================================
def download_from_github(repo, filepath, dest_path):
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    for branch in ("main", "master"):
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{filepath}"
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return dest_path
    raise RuntimeError(f"Download failed for {filepath}")

# =========================================================
# LOAD YOLO MODELS
# =========================================================
@st.cache_resource
def get_model(local_name, repo_path):
    local_file = CACHE_DIR / local_name
    if local_file.exists():
        return YOLO(str(local_file))
    source_path = Path(repo_path)
    if source_path.exists():
        return YOLO(str(source_path))
    if not PRIVATE_REPO:
        raise RuntimeError(
            f"Model not found locally and PRIVATE_REPO is not set: {repo_path}"
        )
    download_from_github(PRIVATE_REPO, repo_path, local_file)
    return YOLO(str(local_file))

# =========================================================
# YOLO PREDICTION
# =========================================================
def _free_memory():
    """Force-clear unreferenced tensors/arrays between inference runs."""
    gc.collect()

def _resize_for_inference(img_np, max_dim=1024):
    """Downscale longest side to max_dim before inference to cap RAM usage."""
    h, w = img_np.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_np

def run_yolo_predict(model, img, conf_threshold, iou_thresh):
    # Shrink image first so each model run uses ~4x less RAM
    small   = _resize_for_inference(img, max_dim=1024)
    pil_img = Image.fromarray(small)
    results = model.predict(
        source=pil_img,
        imgsz=1024,
        conf=conf_threshold,
        iou=iou_thresh,
        retina_masks=True,
        verbose=False,
    )
    result = results[0]
    # Pull everything to CPU and delete the results list immediately
    if result.boxes is not None:
        result.boxes = result.boxes.cpu()
    if result.masks is not None:
        result.masks = result.masks.cpu()
    del results
    _free_memory()
    return result

# =========================================================
# WIDTH CALCULATION — perpendicular cross-section method
# =========================================================
def mask_max_width_pixels(mask):
    """
    Returns (max_width_px, (x, y)) where max_width_px is the widest
    perpendicular cross-section of the crack mask, measured by:
      1. Skeletonising the mask to find the crack centreline.
      2. At each skeleton pixel, casting a perpendicular ray in both
         directions until it exits the mask.
      3. Returning the longest such measurement and the pixel where it occurs.
    Falls back to the distance-transform estimate if skeleton fails.
    """
    binary = (mask > 0).astype(np.uint8)

    # -- skeletonise (Zhang-Suen thinning via morphology) --------------------
    skel   = np.zeros_like(binary)
    temp   = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded   = cv2.erode(temp, kernel)
        opened   = cv2.dilate(eroded, kernel)
        skel     = cv2.bitwise_or(skel, cv2.subtract(temp, opened))
        temp     = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    skel_pts = np.column_stack(np.where(skel > 0))   # (row, col) pairs

    if len(skel_pts) < 2:
        # fallback: distance transform
        dist      = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        max_r     = float(dist.max())
        loc       = np.unravel_index(np.argmax(dist), dist.shape)
        return max_r * 2.0, (int(loc[1]), int(loc[0]))

    # -- estimate local orientation via PCA on skeleton points ---------------
    skel_f   = skel_pts.astype(np.float32)
    mean_pt  = skel_f.mean(axis=0)
    _, _, vt = np.linalg.svd(skel_f - mean_pt, full_matrices=False)
    tang     = vt[0]                          # principal direction (tangent)
    perp     = np.array([-tang[1], tang[0]])  # perpendicular direction

    h, w     = binary.shape
    max_width = 0.0
    best_pt   = (int(mean_pt[1]), int(mean_pt[0]))

    # sample every 4th skeleton pixel for speed
    for pt in skel_pts[::4]:
        r, c = pt
        # cast ray in +perp and -perp directions
        width = 0.0
        for sign in (1, -1):
            for step in range(1, max(h, w)):
                nr = int(round(r + sign * step * perp[0]))
                nc = int(round(c + sign * step * perp[1]))
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    break
                if binary[nr, nc] == 0:
                    break
                width += 1.0
        if width > max_width:
            max_width = width
            best_pt   = (c, r)   # (x, y) for cv2

    return max_width, best_pt

# =========================================================
# CLEAN MASK
# =========================================================
def create_clean_mask(res_crack, orig_h, orig_w):
    if res_crack.masks is None:
        return None
    masks = res_crack.masks.data.cpu().numpy()
    if len(masks) == 0:
        return None
    combined_mask = np.zeros(
        (masks[0].shape[0], masks[0].shape[1]), dtype=np.uint8
    )
    for mask in masks:
        binary = (mask > 0.5).astype(np.uint8) * 255
        combined_mask = cv2.bitwise_or(combined_mask, binary)
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.medianBlur(combined_mask, 5)
    # Resize mask to match the original image so overlays align correctly
    if combined_mask.shape[0] != orig_h or combined_mask.shape[1] != orig_w:
        combined_mask = cv2.resize(
            combined_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
    return combined_mask

# =========================================================
# DRAW RESULTS
# =========================================================
def draw_combined_results(img, coin_boxes, crack_mask, width_mm=None, max_width_point=None):
    output = img.copy()
    h, w   = output.shape[:2]
    thick  = max(2, h // 400)   # scale line thickness to image size
    font_s = max(0.6, h / 1200)

    # ── Crack mask overlay (semi-transparent red) ──────────────────────────
    if crack_mask is not None:
        red_layer          = output.copy()
        crack_bool         = crack_mask > 0
        red_layer[crack_bool] = [220, 50, 50]   # BGR red
        output = cv2.addWeighted(output, 0.65, red_layer, 0.35, 0)

        # Draw crack contours for crisp edges
        contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 80, 80), thick)

    # ── Coin detection overlay ─────────────────────────────────────────────
    if coin_boxes:
        for box in coin_boxes:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rx, ry = (x2 - x1) // 2, (y2 - y1) // 2
            radius = (rx + ry) // 2

            # Semi-transparent green fill
            overlay = output.copy()
            cv2.circle(overlay, (cx, cy), radius, (0, 220, 100), -1)
            output = cv2.addWeighted(output, 0.85, overlay, 0.15, 0)

            # Circle outline
            cv2.circle(output, (cx, cy), radius, (0, 220, 100), thick + 1)
            # Cross-hair centre
            cv2.line(output, (cx - 12, cy), (cx + 12, cy), (0, 220, 100), thick)
            cv2.line(output, (cx, cy - 12), (cx, cy + 12), (0, 220, 100), thick)

            # Label with background
            label     = "REF COIN"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_s * 0.75, thick)
            lx, ly    = x1, max(y1 - 10, lh + 6)
            cv2.rectangle(output, (lx, ly - lh - 4), (lx + lw + 8, ly + 4), (0, 220, 100), -1)
            cv2.putText(output, label, (lx + 4, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, font_s * 0.75, (0, 0, 0), thick)

    # ── Max-width point ────────────────────────────────────────────────────
    if max_width_point is not None:
        x, y = max_width_point
        cv2.circle(output, (x, y), 14, (255, 220, 0), -1)
        cv2.circle(output, (x, y), 14, (255, 255, 255), 2)
        label     = "MAX WIDTH"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_s * 0.75, thick)
        lx = min(x + 18, w - lw - 10)
        ly = max(y - 10, lh + 6)
        cv2.rectangle(output, (lx - 2, ly - lh - 4), (lx + lw + 6, ly + 4), (40, 40, 40), -1)
        cv2.putText(output, label, (lx + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_s * 0.75, (255, 220, 0), thick)

    # ── Width annotation (top-left panel) ─────────────────────────────────
    if width_mm is not None:
        text      = f"Crack Width: {width_mm:.3f} mm"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_s, thick + 1)
        pad       = 12
        cv2.rectangle(output, (20, 20), (20 + tw + pad * 2, 20 + th + pad * 2), (15, 15, 15), -1)
        cv2.rectangle(output, (20, 20), (20 + tw + pad * 2, 20 + th + pad * 2), (255, 220, 0), 2)
        cv2.putText(output, text, (20 + pad, 20 + pad + th),
                    cv2.FONT_HERSHEY_SIMPLEX, font_s, (255, 220, 0), thick + 1)

    return output

# =========================================================
# PAGE HEADER
# =========================================================
st.markdown("""
<div class="si-hero">
    <div class="si-hero-scan"></div>
    <div class="si-hero-orb1"></div>
    <div class="si-hero-orb2"></div>
    <div class="si-hero-ring"></div>
    <div class="si-hero-inner">
        <div class="si-hero-tag">
            <span class="si-hero-tag-dot"></span>
            AI-Powered Structural Analysis
        </div>
        <h1>Struct<span class="grad">Insight</span> AI</h1>
        <div class="si-hero-underline"></div>
        <p class="si-hero-sub">Automated concrete crack detection &amp; severity assessment using YOLOv11 instance segmentation. Upload an image with a reference coin for precise real-world measurements.</p>
        <div class="si-hero-badges">
            <span class="si-badge">🔬 YOLOv11 Segmentation</span>
            <span class="si-badge">📐 Coin Calibration</span>
            <span class="si-badge">📊 ACI 224R Severity Scale</span>
            <span class="si-badge">🏛 COMSATS University Wah</span>
        </div>
        <div class="si-hero-stats">
            <div>
                <span class="si-hero-stat-num">YOLOv11</span>
                <span class="si-hero-stat-lbl">Model Architecture</span>
            </div>
            <div>
                <span class="si-hero-stat-num">±0.01mm</span>
                <span class="si-hero-stat-lbl">Measurement Precision</span>
            </div>
            <div>
                <span class="si-hero-stat-num">ACI 224R</span>
                <span class="si-hero-stat-lbl">Severity Standard</span>
            </div>
            <div>
                <span class="si-hero-stat-num">Real-time</span>
                <span class="si-hero-stat-lbl">Inference Speed</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# IMAGE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a concrete surface image",
    type=["jpg", "jpeg", "png"],
)

# =========================================================
# AUTO CLEANUP: wipe previous run's data when a new image arrives
# =========================================================
current_id = id(uploaded_file) if uploaded_file is not None else None

if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

if current_id != st.session_state.last_file_id:
    # New image detected — clear everything from the previous run
    for key in ["result_image", "combined_mask", "coin_boxes",
                "avg_conf", "mask_coverage", "width_mm",
                "mm_per_pixel", "coin_px", "crack_px", "max_width_point"]:
        if key in st.session_state:
            del st.session_state[key]
    _free_memory()
    st.session_state.last_file_id = current_id

# =========================================================
# PROCESS IMAGE
# =========================================================
if uploaded_file:
    image  = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.markdown('''
    <div class="si-section">
        <div class="si-section-icon">📷</div>
        <span class="si-section-label">Input Image</span>
        <div class="si-section-line"></div>
    </div>''', unsafe_allow_html=True)
    st.markdown('<div class="si-img-panel"><div class="si-img-label">Uploaded Image</div>', unsafe_allow_html=True)
    st.image(img_np, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        # Load models (cached — only loads once ever)
        with st.status("Loading AI models…") as status:
            crack_model = get_model("crack_model.pt", CRACK_MODEL_PATH)
            coin_model  = get_model("coin_model.pt",  COIN_MODEL_PATH)
            status.update(label="Models loaded successfully.", state="complete")

        # --- Coin model ---
        try:
            res_coin = run_yolo_predict(coin_model, img_np, confidence_threshold, iou_threshold)
        except Exception as e:
            st.warning(f"Coin detection failed: {e}")
            res_coin = None

        # Extract what we need from coin result immediately, then free it
        coin_px    = None
        coin_boxes = []   # scaled to original image size for drawing
        orig_h, orig_w = img_np.shape[:2]
        if res_coin is not None and res_coin.boxes is not None and len(res_coin.boxes) > 0:
            boxes_raw = res_coin.boxes.xyxy.cpu().numpy()  # in inference (1024-scaled) coords

            # Scale factor: inference image was resized so longest side = 1024
            infer_longest = 1024.0
            orig_longest  = float(max(orig_h, orig_w))
            scale_back    = orig_longest / infer_longest

            boxes_orig = boxes_raw * scale_back   # back to original pixel space
            coin_boxes = boxes_orig.tolist()

            largest    = max(boxes_orig, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            x1, y1, x2, y2 = map(int, largest)
            # coin_px measured in original-image pixels (consistent with mask which is also orig size)
            coin_px    = ((x2 - x1) + (y2 - y1)) / 2.0
        del res_coin
        _free_memory()   # free coin tensors before crack model runs

        # --- Crack model ---
        try:
            res_crack = run_yolo_predict(crack_model, img_np, confidence_threshold, iou_threshold)
        except Exception as e:
            st.error(f"Crack detection failed: {e}")
            st.stop()

        # -------------------------------------------------
        # CRACK SEGMENTATION → extract everything to numpy, then free
        # -------------------------------------------------
        combined_mask   = create_clean_mask(res_crack, orig_h, orig_w)
        avg_conf = None
        if res_crack.boxes is not None and len(res_crack.boxes) > 0:
            avg_conf = float(res_crack.boxes.conf.cpu().numpy().mean())
        del res_crack
        _free_memory()

        crack_px        = None
        max_width_point = None
        if combined_mask is not None:
            crack_px, max_width_point = mask_max_width_pixels(combined_mask)

        # -------------------------------------------------
        # CALIBRATION
        # -------------------------------------------------
        width_mm    = None
        mm_per_pixel = None
        if use_manual_calibration:
            mm_per_pixel = manual_mm_per_pixel
        elif coin_px is not None:
            mm_per_pixel = coin_diameter_mm / coin_px

        if mm_per_pixel is not None and crack_px is not None:
            width_mm = crack_px * mm_per_pixel * calibration_factor

        # -------------------------------------------------
        # ANNOTATED IMAGE
        # -------------------------------------------------
        result_image = draw_combined_results(
            img_np, coin_boxes, combined_mask, width_mm, max_width_point
        )
        st.markdown('''
        <div class="si-section">
            <div class="si-section-icon">🔍</div>
            <span class="si-section-label">Detection Output</span>
            <div class="si-section-line"></div>
        </div>''', unsafe_allow_html=True)

        col_img_l, col_img_r = st.columns(2)
        with col_img_l:
            st.markdown('<div class="si-img-panel"><div class="si-img-label">Original</div>', unsafe_allow_html=True)
            st.image(img_np, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_img_r:
            st.markdown('<div class="si-img-panel"><div class="si-img-label">AI Detection — Cracks Highlighted</div>', unsafe_allow_html=True)
            st.image(result_image, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('''
        <div class="si-section">
            <div class="si-section-icon">📊</div>
            <span class="si-section-label">Analysis Metrics</span>
            <div class="si-section-line"></div>
        </div>''', unsafe_allow_html=True)

        # -------------------------------------------------
        # METRICS — actual model outputs, not slider values
        # avg_conf already extracted before res_crack was freed above
        # -------------------------------------------------

        mask_coverage = None
        if combined_mask is not None:
            crack_area    = np.sum(combined_mask > 0)
            total_area    = combined_mask.shape[0] * combined_mask.shape[1]
            mask_coverage = crack_area / total_area

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Detection Confidence",
                f"{avg_conf:.0%}" if avg_conf is not None else "N/A",
                help="Mean confidence score across all detected crack instances.",
            )
        with col2:
            st.metric(
                "Crack Coverage",
                f"{mask_coverage:.3%}" if mask_coverage is not None else "N/A",
                help="Percentage of image area classified as cracked.",
            )
        with col3:
            st.metric(
                "Scale",
                f"{mm_per_pixel:.4f} mm/px" if mm_per_pixel is not None else "Unavailable",
                help="Pixel-to-mm ratio derived from coin size or manual input.",
            )
        with col4:
            st.metric(
                "Max Crack Width",
                f"{width_mm:.2f} mm" if width_mm is not None else "Unavailable",
                help="Maximum crack width after applying calibration factor.",
            )

        # -------------------------------------------------
        # SEVERITY ASSESSMENT
        # -------------------------------------------------
        if width_mm is not None:
            st.markdown('''
            <div class="si-section">
                <div class="si-section-icon">🏗</div>
                <span class="si-section-label">Structural Condition Assessment</span>
                <div class="si-section-line"></div>
            </div>''', unsafe_allow_html=True)

            if width_mm < 0.1:
                severity       = "Hairline"
                badge_color    = "#16a34a"
                description    = "Barely visible cracks; typically non-structural in nature. Monitor periodically."
                recommendation = "Routine monitoring recommended. No immediate intervention required."
            elif width_mm < 0.3:
                severity       = "Minor / Fine"
                badge_color    = "#ca8a04"
                description    = (
                    "Generally acceptable in dry or interior environments. "
                    "Protective surface sealing may be required in moisture-exposed areas."
                )
                recommendation = "Consider protective sealing if the surface is exposed to moisture or weathering."
            elif width_mm <= 0.5:
                severity       = "Moderate"
                badge_color    = "#ea580c"
                description    = (
                    "Exceeds ACI 224R crack-width limits. "
                    "May indicate structural overstress or durability concerns."
                )
                recommendation = "Engineering evaluation recommended. Investigate loading and rebar condition."
            else:
                severity       = "Severe"
                badge_color    = "#dc2626"
                description    = (
                    "Significant structural distress. High risk of rebar corrosion "
                    "and progressive structural deterioration."
                )
                recommendation = "Immediate intervention required — epoxy injection, patching, or full rehabilitation."

            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""
                <div class="si-severity-wrap" style="border-top: 4px solid {badge_color};">
                    <div class="si-sev-header">Severity Level</div>
                    <div class="si-sev-badge" style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}55;">
                        ● {severity}
                    </div>
                    <div class="si-sev-value" style="color:{badge_color};">{width_mm:.2f}
                        <span class="si-sev-unit">mm</span>
                    </div>
                    <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.5rem;">Maximum crack width</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="si-info-block" style="border-left-color:{badge_color};">
                    <strong>Condition</strong>{description}
                </div>
                <div class="si-info-block" style="border-left-color:{badge_color};">
                    <strong>Recommended Action</strong>{recommendation}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="si-error">
                ⚠️ Width measurement unavailable — coin not detected or crack not segmented.
                Enable <strong>Manual Calibration</strong> in the sidebar to proceed without a reference coin.
            </div>''', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Processing error: {e}")

# =========================================================
# ABOUT SECTION
# =========================================================
st.markdown("""
<div class="si-section">
    <div class="si-section-icon">📘</div>
    <span class="si-section-label">About the Project</span>
    <div class="si-section-line"></div>
</div>
""", unsafe_allow_html=True)

col_about_l, col_about_r = st.columns([3, 2])
with col_about_l:
    st.markdown("""
    <div class="si-card">
        <div class="si-card-tag">Project Overview</div>
        <h3>AI-Powered Infrastructure Inspection</h3>
        <p>
        StructInsight AI applies <strong>YOLOv11 instance segmentation</strong> to automate the
        detection and severity classification of structural cracks in concrete. Traditional inspection
        methods are labour-intensive, subjective, and difficult to scale. By integrating computer vision
        into the workflow, this system delivers real-time, repeatable, and quantitative results —
        enabling engineers to prioritise repairs, reduce costs, and improve public safety.
        </p>
        <div class="si-cap-row">
            <span class="si-cap">Real-time Segmentation</span>
            <span class="si-cap">Coin Calibration</span>
            <span class="si-cap">ACI 224R Aligned</span>
            <span class="si-cap">Field-Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_about_r:
    st.markdown("""
    <div class="si-card">
        <div class="si-card-tag">Severity Reference — ACI 224R</div>
        <h3>Crack Width Classification</h3>
        <div class="si-ref-row">
            <div class="si-ref-dot" style="background:#16a34a;"></div>
            <span class="si-ref-label">Hairline</span>
            <span class="si-ref-action">Monitor only</span>
            <span class="si-ref-range">&lt; 0.1 mm</span>
        </div>
        <div class="si-ref-row">
            <div class="si-ref-dot" style="background:#ca8a04;"></div>
            <span class="si-ref-label">Minor</span>
            <span class="si-ref-action">Protective seal</span>
            <span class="si-ref-range">0.1 – 0.3 mm</span>
        </div>
        <div class="si-ref-row">
            <div class="si-ref-dot" style="background:#ea580c;"></div>
            <span class="si-ref-label">Moderate</span>
            <span class="si-ref-action">Engineering review</span>
            <span class="si-ref-range">0.3 – 0.5 mm</span>
        </div>
        <div class="si-ref-row">
            <div class="si-ref-dot" style="background:#dc2626;"></div>
            <span class="si-ref-label">Severe</span>
            <span class="si-ref-action">Immediate repair</span>
            <span class="si-ref-range">&gt; 0.5 mm</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="si-section">
    <div class="si-section-icon">👥</div>
    <span class="si-section-label">Research Team</span>
    <div class="si-section-line"></div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3 = st.columns(3)
team = [
    (t1, "Project Lead", "Hasnain Haroon",
     "Final Year Project — Department of Civil Engineering, COMSATS University Islamabad, Wah Campus"),
    (t2, "Supervisor", "Engr. Sandeerah Choudhary",
     "Faculty Advisor, Department of Civil Engineering"),
    (t3, "Team Members", "Liza Liaqat · Ammar Faheem Khawaja",
     "Contributing Researchers, Civil Engineering FYP 2025"),
]
for col, role, name, detail in team:
    with col:
        st.markdown(f"""
        <div class="si-card">
            <div class="si-team-role">{role}</div>
            <div class="si-team-divider"></div>
            <div class="si-team-name">{name}</div>
            <div class="si-team-detail">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="si-footer">
    <span>StructInsight AI</span> · COMSATS University Islamabad, Wah Campus · Civil Engineering FYP 2025
</div>
""", unsafe_allow_html=True)
