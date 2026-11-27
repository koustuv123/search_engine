from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import hashlib
import os
from dotenv import load_dotenv
load_dotenv()
hf_token=os.getenv("HF_TOKEN")
import streamlit as st

st.set_page_config(
    page_title="Your friendly neighbourhood chatman",
    page_icon="🕷️",
    layout="centered"
)

# ----------------------------------------------------------------------
# UI THEME + ANIMATIONS (presentation only)
# ----------------------------------------------------------------------
def _flat(html: str) -> str:
    # Markdown treats indented / blank-line HTML as code, so flatten it.
    return "".join(line.strip() for line in html.splitlines())


THEMES = {
    "Normal": {
        "accent": "#6c8cff", "accent2": "#22d3ee",
        "glow1": "rgba(108,140,255,0.22)", "glow2": "rgba(34,211,238,0.14)",
        "icon": "🤖",
        "title": "Normal mode",
        "tagline": "Clear, balanced answers — with tools on standby",
    },
    "Friend": {
        "accent": "#ff6fa8", "accent2": "#ffb45e",
        "glow1": "rgba(255,111,168,0.22)", "glow2": "rgba(255,180,94,0.14)",
        "icon": "🤝",
        "title": "Friend mode",
        "tagline": "Aree yaar, I'm all ears ❤️",
    },
    "Professional": {
        "accent": "#f5c26b", "accent2": "#4fb3ff",
        "glow1": "rgba(245,194,107,0.18)", "glow2": "rgba(79,179,255,0.16)",
        "icon": "💼",
        "title": "Professional mode",
        "tagline": "Honest, direct, no sugarcoating",
    },
}

import math
from urllib.parse import quote


def web_uri(corner="tl", size=200, rays=6, rings=5, stroke="#ffffff", sw=1.3):
    """Corner spider-web drawn as an SVG data-URI (tl / tr / bl / br)."""
    def pt(r, a):
        x = r * math.cos(a)
        y = r * math.sin(a)
        if "r" in corner:
            x = size - x
        if "b" in corner:
            y = size - y
        return x, y

    ox, oy = pt(0, 0)
    d = []
    for i in range(rays):
        a = (math.pi / 2) * i / (rays - 1)
        x, y = pt(size, a)
        d.append(f"M{ox:.1f} {oy:.1f}L{x:.1f} {y:.1f}")
    for k in range(1, rings + 1):
        r = size * k / rings
        prev = None
        for i in range(rays):
            a = (math.pi / 2) * i / (rays - 1)
            cur = pt(r, a)
            if prev:
                am = (math.pi / 2) * (i - 0.5) / (rays - 1)
                cx, cy = pt(r * 0.82, am)
                d.append(
                    f"M{prev[0]:.1f} {prev[1]:.1f}"
                    f"Q{cx:.1f} {cy:.1f} {cur[0]:.1f} {cur[1]:.1f}"
                )
            prev = cur
    path = "".join(d)
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {size} {size}' "
        f"fill='none' stroke='{stroke}' stroke-width='{sw}' stroke-linecap='round'>"
        f"<path d='{path}'/></svg>"
    )
    return "data:image/svg+xml;utf8," + quote(svg)


def spider_uri(color="#6c8cff"):
    """A small original cartoon spider, as an SVG data-URI."""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 44 44'>"
        "<g stroke='#dfe4ff' stroke-width='2' fill='none' stroke-linecap='round'>"
        "<path d='M16 20L6 12L2 16M16 23L5 23L1 28M17 27L8 35L5 41M19 29L15 38L16 43"
        "M28 20L38 12L42 16M28 23L39 23L43 28M27 27L36 35L39 41M25 29L29 38L28 43'/>"
        "</g>"
        f"<ellipse cx='22' cy='26' rx='8' ry='9.5' fill='#12162d' stroke='{color}' stroke-width='2'/>"
        f"<circle cx='22' cy='14' r='6' fill='#12162d' stroke='{color}' stroke-width='2'/>"
        "<circle cx='19.5' cy='13' r='1.8' fill='#fff'/>"
        "<circle cx='24.5' cy='13' r='1.8' fill='#fff'/>"
        "<circle cx='19.9' cy='13.3' r='.8' fill='#000'/>"
        "<circle cx='24.1' cy='13.3' r='.8' fill='#000'/>"
        "</svg>"
    )
    return "data:image/svg+xml;utf8," + quote(svg)


def render_hist_badge(slot):
    n = len(st.session_state.messages)
    label = "message" if n == 1 else "messages"
    slot.markdown(
        _flat(f'<div class="hist-badge">💬 {n} {label} in this chat</div>'),
        unsafe_allow_html=True,
    )


DECOR_HTML = (
    '<div class="web tr"></div>'
    '<div class="web br"></div>'
    '<div class="hang"><div class="thread"></div><div class="spidey"></div></div>'
)

BASE_CSS = """
<style>
:root{
  --accent: __ACCENT__;
  --accent2: __ACCENT2__;
  --bg: #0a0d1a;
  --card: rgba(255,255,255,0.05);
  --border: rgba(255,255,255,0.10);
  --text: #e9ecf8;
  --muted: #98a1c0;
}

/* ---------- App background ---------- */
.stApp{
  background:
    radial-gradient(900px 500px at 10% -10%, __GLOW1__, transparent 60%),
    radial-gradient(800px 500px at 100% 0%, __GLOW2__, transparent 60%),
    radial-gradient(700px 500px at 50% 120%, __GLOW1__, transparent 60%),
    var(--bg);
  background-attachment: fixed;
  color: var(--text);
}
[data-testid="stHeader"]{ background: transparent; }
#MainMenu, footer{ visibility: hidden; }
.block-container, [data-testid="stMainBlockContainer"]{
  position: relative;
  padding-top: 3rem;
  max-width: 860px;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span{ color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"]{ color: var(--muted) !important; }

/* ---------- Title ---------- */
.stApp h1{
  font-weight: 800;
  font-size: clamp(1.5rem, 4.2vw, 2.1rem);
  letter-spacing: -0.5px;
  text-align: center;
  background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent));
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shine 6s linear infinite, fadeUp .8s ease both;
}
.stApp h3{ font-weight: 700; }
@keyframes shine{ to{ background-position: 200% center; } }
@keyframes fadeUp{
  from{ opacity:0; transform: translateY(14px); }
  to{ opacity:1; transform: translateY(0); }
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border-right: 1px solid var(--border);
  backdrop-filter: blur(14px);
}
[data-testid="stSidebar"] h1{
  font-size: 1.5rem;
  text-align: left;
}
[data-testid="stSidebar"] hr{ border-color: var(--border); }

/* ---------- Buttons ---------- */
.stButton > button{
  width: 100%;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: var(--card);
  color: var(--text);
  transition: all .25s ease;
}
.stButton > button:hover{
  border-color: var(--accent);
  color: #fff;
  transform: translateY(-2px);
  box-shadow: 0 8px 24px -8px var(--accent);
}

/* ---------- Select box ---------- */
[data-baseweb="select"] > div{
  border-radius: 12px !important;
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  transition: border-color .25s ease, box-shadow .25s ease;
}
[data-baseweb="select"] > div:hover{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(255,255,255,0.04);
}

/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"]{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1rem 1.2rem;
  margin-bottom: .8rem;
  backdrop-filter: blur(10px);
  box-shadow: 0 10px 30px -18px rgba(0,0,0,.8);
  position: relative;
  overflow: hidden;
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
  animation: msgIn .45s cubic-bezier(.2,.8,.2,1) backwards;
}
/* assistant slides in from the left, user from the right */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
  border-left: 3px solid var(--accent);
  animation-name: slideLeft;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  border-right: 3px solid var(--accent2);
  animation-name: slideRight;
}
[data-testid="stChatMessage"]:hover{
  transform: translateY(-3px);
  border-color: var(--accent);
  box-shadow: 0 18px 40px -18px var(--accent);
}
/* little web in the corner of every message */
[data-testid="stChatMessage"]::after{
  content: "";
  position: absolute;
  top: 0; right: 0;
  width: 70px; height: 70px;
  background: url("__WEB_TR__") no-repeat;
  background-size: contain;
  opacity: .16;
  pointer-events: none;
}
@keyframes msgIn{
  from{ opacity:0; transform: translateY(16px) scale(.98); }
  to{ opacity:1; transform: translateY(0) scale(1); }
}
@keyframes slideLeft{
  from{ opacity:0; transform: translateX(-40px); }
  to{ opacity:1; transform: translateX(0); }
}
@keyframes slideRight{
  from{ opacity:0; transform: translateX(40px); }
  to{ opacity:1; transform: translateX(0); }
}
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"]{
  animation: avatarPulse 3s ease-in-out infinite;
}
@keyframes avatarPulse{
  0%,100%{ box-shadow: 0 0 0 0 rgba(255,255,255,0); }
  50%{ box-shadow: 0 0 0 6px rgba(255,255,255,0.07); }
}

/* ---------- Chat input ---------- */
[data-testid="stChatInput"]{
  border-radius: 18px;
  border: 1px solid var(--border);
  background: #f4f6ff !important;
  transition: all .25s ease;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea{
  background: transparent !important;
  color: #000000 !important;
  -webkit-text-fill-color: #000000 !important;
  caret-color: #000000;
}
[data-testid="stChatInput"] textarea::placeholder{
  color: #5b6280 !important;
  -webkit-text-fill-color: #5b6280 !important;
  opacity: 1;
}
[data-testid="stChatInput"]:focus-within{
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(255,255,255,0.05), 0 10px 40px -10px var(--accent);
}
[data-testid="stBottom"] > div{ background: transparent; }

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"]{
  border: 1.5px dashed var(--accent);
  border-radius: 16px;
  background: var(--card);
  transition: all .3s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
  background: rgba(255,255,255,0.08);
  transform: translateY(-2px);
  box-shadow: 0 12px 30px -14px var(--accent);
}

/* ---------- Alerts / spinner ---------- */
[data-testid="stAlert"]{
  border-radius: 14px;
  border: 1px solid var(--border);
  animation: fadeUp .5s ease both;
}
[data-testid="stSpinner"] i{ border-top-color: var(--accent) !important; }

/* ---------- Tool chips ---------- */
.tool-chip{
  display: inline-block;
  margin: 4px 8px 4px 0;
  padding: 6px 14px;
  border-radius: 999px;
  font-size: .9rem;
  border: 1px solid var(--accent);
  background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  animation: fadeUp .5s ease both;
}

/* ---------- Mode hero card ---------- */
.hero{
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  gap: 22px;
  padding: 18px 24px;
  margin: 6px 0 22px;
  border-radius: 22px;
  border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
  backdrop-filter: blur(14px);
  box-shadow: 0 20px 50px -25px var(--accent);
  animation: fadeUp .6s ease both;
}
.hero::before{
  content:"";
  position:absolute; inset:-1px;
  border-radius: 22px;
  padding: 1px;
  background: linear-gradient(120deg, var(--accent), transparent 40%, var(--accent2));
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  opacity: .7;
  pointer-events: none;
}
.hero-text h2{
  margin: 0;
  font-size: 1.5rem;
  font-weight: 800;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-text p{ margin: 4px 0 0; color: var(--muted) !important; }
.stage{
  position: relative;
  flex: 0 0 170px;
  height: 110px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* ===== FRIEND: two hands meet and shake ===== */
.hand{ position:absolute; font-size: 3rem; opacity: 0; }
.hand.l{ animation: handL 1.3s ease-out forwards; }
.hand.r{ animation: handR 1.3s ease-out forwards; }
@keyframes handL{
  0%{ transform: translateX(-120px) rotate(-25deg); opacity:0; }
  45%{ opacity:1; }
  80%{ transform: translateX(-12px) rotate(0); opacity:1; }
  100%{ transform: translateX(-12px) rotate(0); opacity:0; }
}
@keyframes handR{
  0%{ transform: translateX(120px) rotate(25deg); opacity:0; }
  45%{ opacity:1; }
  80%{ transform: translateX(12px) rotate(0); opacity:1; }
  100%{ transform: translateX(12px) rotate(0); opacity:0; }
}
.shake-wrap{
  position:absolute; font-size: 3.6rem; opacity: 0;
  animation: popIn .35s cubic-bezier(.3,1.6,.5,1) 1.15s forwards;
}
.shake{ display:inline-block; animation: shake .7s ease-in-out 1.5s infinite; transform-origin: 50% 80%; }
@keyframes popIn{
  from{ opacity:0; transform: scale(.4); }
  to{ opacity:1; transform: scale(1); }
}
@keyframes shake{
  0%,100%{ transform: translateY(0) rotate(0); }
  25%{ transform: translateY(-7px) rotate(-9deg); }
  75%{ transform: translateY(5px) rotate(9deg); }
}
.heart{ position:absolute; bottom: 20px; opacity: 0; font-size: 1.1rem; }
.heart.h1{ left: 28px; animation: floatUp 2.6s ease-in 1.6s infinite; }
.heart.h2{ left: 78px; animation: floatUp 2.6s ease-in 2.2s infinite; }
.heart.h3{ left: 125px; animation: floatUp 2.6s ease-in 2.8s infinite; }
@keyframes floatUp{
  0%{ transform: translateY(0) scale(.6); opacity:0; }
  20%{ opacity:1; }
  100%{ transform: translateY(-70px) scale(1.1); opacity:0; }
}

/* ===== NORMAL: floating robot that waves ===== */
.bot{ position:absolute; font-size: 3.6rem; animation: bob 3s ease-in-out infinite; }
.wave{
  position:absolute; left: 98px; top: 18px; font-size: 1.8rem;
  transform-origin: 70% 80%;
  animation: wave 1.6s ease-in-out infinite;
}
@keyframes bob{
  0%,100%{ transform: translateY(0); }
  50%{ transform: translateY(-10px); }
}
@keyframes wave{
  0%,60%,100%{ transform: rotate(0); }
  10%,30%,50%{ transform: rotate(18deg); }
  20%,40%{ transform: rotate(-12deg); }
}
.spark{ position:absolute; font-size: 1rem; animation: twinkle 2.2s ease-in-out infinite; }
.spark.s1{ left: 14px; top: 14px; }
.spark.s2{ right: 12px; top: 8px; animation-delay: .7s; }
.spark.s3{ left: 30px; bottom: 6px; animation-delay: 1.3s; }
@keyframes twinkle{
  0%,100%{ opacity: .15; transform: scale(.6) rotate(0); }
  50%{ opacity: 1; transform: scale(1.2) rotate(25deg); }
}
.dots{ position:absolute; right: 8px; bottom: 6px; display:flex; gap: 4px; }
.dots i{
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent2);
  animation: dot 1.2s ease-in-out infinite;
}
.dots i:nth-child(2){ animation-delay: .15s; }
.dots i:nth-child(3){ animation-delay: .3s; }
@keyframes dot{
  0%,80%,100%{ transform: translateY(0); opacity:.4; }
  40%{ transform: translateY(-6px); opacity:1; }
}

/* ===== PROFESSIONAL: briefcase drop + growth bars ===== */
.case{
  position:absolute; left: 6px; font-size: 3.2rem; opacity: 0;
  animation: drop .9s cubic-bezier(.3,1.5,.5,1) .1s forwards;
}
@keyframes drop{
  0%{ transform: translateY(-80px) rotate(-12deg); opacity:0; }
  100%{ transform: translateY(0) rotate(0); opacity:1; }
}
.bars{
  position:absolute; right: 4px; bottom: 10px;
  display:flex; align-items:flex-end; gap: 7px; height: 80px;
}
.bars i{
  display:block; width: 13px; border-radius: 5px 5px 2px 2px;
  background: linear-gradient(180deg, var(--accent), var(--accent2));
  transform-origin: bottom;
  transform: scaleY(0);
  animation: grow .8s cubic-bezier(.2,.9,.3,1.2) forwards, breathe 3s ease-in-out 2s infinite;
}
.bars i:nth-child(1){ height: 28px; animation-delay: .6s, 2s; }
.bars i:nth-child(2){ height: 44px; animation-delay: .8s, 2.2s; }
.bars i:nth-child(3){ height: 36px; animation-delay: 1.0s, 2.4s; }
.bars i:nth-child(4){ height: 62px; animation-delay: 1.2s, 2.6s; }
.bars i:nth-child(5){ height: 80px; animation-delay: 1.4s, 2.8s; }
@keyframes grow{ to{ transform: scaleY(1); } }
@keyframes breathe{
  0%,100%{ filter: brightness(1); }
  50%{ filter: brightness(1.35); }
}
.trend{
  position:absolute; right: 2px; top: 0; font-size: 1.4rem; opacity: 0;
  animation: popIn .4s ease 2s forwards;
}

/* ===== Spider webs ===== */
.hero::after{
  content: "";
  position: absolute;
  top: 0; right: 0;
  width: 120px; height: 120px;
  background: url("__WEB_TR__") no-repeat;
  background-size: contain;
  opacity: .22;
  pointer-events: none;
}
[data-testid="stElementContainer"]:has(.web){
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 0;
  margin: 0;
  padding: 0;
}
.web{
  position: fixed;
  width: 260px; height: 260px;
  pointer-events: none;
  z-index: 0;
  opacity: .17;
  background-repeat: no-repeat;
  background-size: contain;
}
.web.tr{
  top: 0; right: 0;
  background-image: url("__WEB_TR__");
  transform-origin: 100% 0;
  animation: sway 7s ease-in-out infinite;
}
.web.br{
  bottom: 0; right: 0;
  background-image: url("__WEB_BR__");
  transform-origin: 100% 100%;
  animation: sway 9s ease-in-out infinite reverse;
}
@keyframes sway{
  0%,100%{ transform: rotate(0deg); }
  50%{ transform: rotate(2.5deg); }
}

/* ===== Little spider hanging from the title ===== */
.hang{
  position: absolute;
  top: 0; right: 5%;
  width: 40px;
  display: flex;
  flex-direction: column;
  align-items: center;
  transform-origin: 50% 0;
  animation: swing 3.6s ease-in-out 1.4s infinite;
  z-index: 5;
  pointer-events: none;
}
.hang .thread{
  width: 1.5px;
  height: 70px;
  background: linear-gradient(rgba(255,255,255,.12), rgba(255,255,255,.7));
  animation: lower 1.4s cubic-bezier(.3,.9,.4,1) backwards;
}
.hang .spidey{
  width: 38px; height: 38px;
  margin-top: -4px;
  background: url("__SPIDER__") center / contain no-repeat;
  animation: bobs 1.1s ease-in-out infinite;
}
@keyframes lower{ from{ height: 0; } to{ height: 70px; } }
@keyframes swing{
  0%,100%{ transform: rotate(0deg); }
  25%{ transform: rotate(8deg); }
  75%{ transform: rotate(-8deg); }
}
@keyframes bobs{
  0%,100%{ transform: scale(1); }
  50%{ transform: scale(1.08,.94); }
}

/* ===== Sidebar: crawling spider, badge, web ===== */
.crawl{ position: relative; height: 34px; margin: 2px 0 8px; }
.crawl::before{
  content: "";
  position: absolute;
  left: 0; right: 0; top: 50%;
  border-top: 1.5px dashed rgba(255,255,255,.25);
}
.crawl i{
  position: absolute;
  top: 4px; left: 0;
  width: 26px; height: 26px;
  background: url("__SPIDER__") center / contain no-repeat;
  animation: crawl 6s ease-in-out infinite alternate;
}
@keyframes crawl{
  from{ left: 0; }
  to{ left: calc(100% - 26px); }
}
.hist-badge{
  display: inline-block;
  margin: 2px 0 10px;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: .82rem;
  border: 1px solid var(--border);
  background: var(--card);
  animation: badgePop .4s cubic-bezier(.3,1.6,.5,1) backwards;
}
@keyframes badgePop{
  from{ transform: scale(.7); opacity: 0; }
  to{ transform: scale(1); opacity: 1; }
}
.side-web{
  height: 150px;
  margin-top: 20px;
  background: url("__WEB_BL__") left bottom / contain no-repeat;
  opacity: .22;
}

@media (max-width: 640px){
  .hero{ flex-direction: column; text-align: center; }
  .hang{ display: none; }
}
</style>
"""

HEROES = {
    "Friend": """
<div class="stage">
<span class="hand l">🫱</span>
<span class="hand r">🫲</span>
<span class="shake-wrap"><span class="shake">🤝</span></span>
<span class="heart h1">❤️</span>
<span class="heart h2">💖</span>
<span class="heart h3">💕</span>
</div>
""",
    "Normal": """
<div class="stage">
<span class="spark s1">✨</span>
<span class="spark s2">✨</span>
<span class="spark s3">✦</span>
<span class="bot">🤖</span>
<span class="wave">👋</span>
<span class="dots"><i></i><i></i><i></i></span>
</div>
""",
    "Professional": """
<div class="stage">
<span class="case">💼</span>
<div class="bars"><i></i><i></i><i></i><i></i><i></i></div>
<span class="trend">📈</span>
</div>
""",
}


def apply_theme(selected_mode: str):
    t = THEMES[selected_mode]
    css = (
        BASE_CSS
        .replace("__ACCENT2__", t["accent2"])
        .replace("__ACCENT__", t["accent"])
        .replace("__GLOW1__", t["glow1"])
        .replace("__GLOW2__", t["glow2"])
        .replace("__WEB_TR__", web_uri("tr"))
        .replace("__WEB_BR__", web_uri("br"))
        .replace("__WEB_BL__", web_uri("bl"))
        .replace("__SPIDER__", spider_uri(t["accent"]))
    )
    st.markdown(css + "\n" + DECOR_HTML, unsafe_allow_html=True)


def render_hero(selected_mode: str):
    t = THEMES[selected_mode]
    html = (
        '<div class="hero">'
        + HEROES[selected_mode]
        + '<div class="hero-text">'
        + f'<h2>{t["icon"]} {t["title"]}</h2>'
        + f'<p>{t["tagline"]}</p>'
        + '</div></div>'
    )
    st.markdown(_flat(html), unsafe_allow_html=True)


#chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history={
        "Normal":[],
        "Friend":[],
        "Professional":[]
    }


prompts = {
    "Normal": """
You are a helpful AI assistant and document analysis expert.

DOCUMENT HANDLING RULES:

1. When the user asks about an uploaded PDF,
   ALWAYS use the PDF_Search tool first.

2. Base your answer on the retrieved document
   content, not assumptions or unrelated knowledge.

3. Do not invent document sections, topics,
   examples, or information.

4. If the retrieved content is insufficient,
   clearly explain what is missing.

5. If multiple documents are uploaded,
   identify which document is relevant.

6. Explain the document in a structured,
   beginner-friendly manner.

7. If the user says "explain this document",
   interpret it as a request to explain the
   currently uploaded document.

For general questions unrelated to uploaded
documents, answer normally.
""",

    "Friend": """You are the user's close friend, not a professional AI assistant.

YOUR PERSONALITY:
- Talk naturally, casually, warmly and emotionally.
- Sound like someone chatting with a close friend on WhatsApp.
- Be expressive. Use emojis naturally ❤️🥹😂🤗
- You can use casual expressions like "Aree yaar!", "Brooo!",
  "Wait, seriously?", "Haha", "I get you".
- Don't sound like a therapist, teacher, customer support agent,
  or motivational speaker.

HOW TO RESPOND:
- Keep replies short and conversational by default.
- React to what the user says before giving advice.
- If the user shares something sad, comfort them first.
- If they're excited, share their excitement.
- If they're joking, joke back.
- Ask natural follow-up questions.
- Don't turn every conversation into a problem-solving session.
- Sometimes just listen and respond like a friend.
- Don't give numbered lists unless explicitly requested.
- Don't over-explain simple things.
- Don't start every answer with "I understand".
- Don't end every response with "How can I help you?"

IMPORTANT:
- Never announce that you are an AI assistant.
- Don't pretend to be a human or claim real-life experiences.
- Be honest, but don't be unnecessarily formal.
- Match the user's language and energy.

EXAMPLE:

User: I'm not feeling well.
You: Aree yaar 🥺❤️ What happened? Are you physically unwell
or is something bothering you? Tell me.

User: I had a terrible day at work.
You: Oof bro, what happened now? 😭
Work ne phir se dimaag kharab kar diya kya?

User: I finally completed my project!
You: BROOO! LET'S GOOO! 😂🔥
That's amazing! How does it feel to finally get it done?

Remember: Have a conversation. Don't deliver a presentation.
"""
,

    "Professional": """
    You are an experienced senior mentor who values honesty,
    critical thinking and professional growth.

    - Be direct, realistic and straightforward.
    - Do not sugarcoat uncomfortable truths.
    - Never agree with the user just to please them.
    - Challenge incorrect assumptions and weak reasoning.
    - Point out mistakes clearly and explain why they are mistakes.
    - Give practical, actionable recommendations.
    - Prioritize facts, logic and evidence over emotions.
    - Avoid unnecessary compliments and motivational fluff.
    - Be respectful, but never compromise honesty.

    Your goal is to help the user improve, not simply make
    the user feel good.
    """
}    
#tools

@st.cache_resource
def model_and_tools_initialization():

    # Wikipedia tool
    wiki_api = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=250
    )
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

    # Arxiv tool
    arxiv_api = ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=250
    )
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api)

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Persistent vector database for uploaded PDFs
    vectorstore = Chroma(
        collection_name="uploaded_pdfs",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

    # PDF retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    from langchain_core.tools import create_retriever_tool

    pdf_tool = create_retriever_tool(
        retriever=retriever,
        name="PDF_Search",
        description=(
            "Search information from PDFs uploaded by the user. "
            "Use this tool whenever the user asks questions about "
            "their uploaded documents. Retrieve relevant passages "
            "and answer based on their content."
        )
    )

    available_tools = [
        wiki_tool,
        arxiv_tool,
        pdf_tool
    ]

    # LLM
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.4
    )

    return llm, available_tools, vectorstore


llm, available_tools, vectorstore = (
        model_and_tools_initialization()
    )
@st.cache_resource
def create_agent_by_this_fun(mode):

    llm, available_tools, _ = (
        model_and_tools_initialization()
    )

    from langchain.agents import create_agent

    agent = create_agent(
        model=llm,
        tools=available_tools,
        system_prompt=prompts[mode]
    )

    return agent

with st.sidebar:
    st.title("History")
    st.markdown(
        '<div class="crawl"><i></i></div>',
        unsafe_allow_html=True
    )

    hist_slot = st.empty()

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Personality selection
    st.subheader("Choose Personality")

    mode = st.selectbox(
        "Select your mode",
        ["Normal", "Friend", "Professional"],
        key="mode"
    )

    if "previous_mode" not in st.session_state:
        st.session_state.previous_mode = mode
        st.session_state.mode_switched = False

    elif st.session_state.previous_mode != mode:
        st.session_state.previous_mode = mode
        st.session_state.mode_switched = True

    else:
        st.session_state.mode_switched = False

    # Load history belonging to the selected mode
    st.session_state.messages = st.session_state.chat_history[mode]

    st.toast(
        f"{mode} mode activated",
        icon=THEMES[mode]["icon"]
    )

    mode_descriptions = {
        "Normal": "🤖 Your everyday AI assistant",
        "Friend": "❤️ A friendly, supportive companion",
        "Professional": "💼 Honest advice, no sugarcoating"
    }

    st.caption(mode_descriptions[mode])

    st.divider()

    # Document selection
    st.subheader("📚 Document Library")

    all_records = vectorstore.get(
        include=["metadatas"]
    )

    document_names = sorted({
        metadata["source"]
        for metadata in all_records["metadatas"]
        if metadata and metadata.get("source")
    })

    if document_names:
        selected_document = st.selectbox(
            "Select a document to discuss",
            options=document_names,
            index=len(document_names) - 1,
            key="selected_document"
        )

        st.caption(f"📄 {len(document_names)} documents available")

    else:
        selected_document = None
        st.info("Upload a PDF to start document-based Q&A.")

    st.markdown(
        '<div class="side-web"></div>',
        unsafe_allow_html=True
    )

    

apply_theme(mode)
render_hist_badge(hist_slot)


agent=create_agent_by_this_fun(mode)
#streamlit
st.title("Your friendly neighbourhood chatman")

# Animated mode banner (replays whenever the mode changes)
render_hero(mode)


# Display previous messages
for message in st.session_state.messages:

    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])

    elif message["role"] == "assistant" and message["content"]:
        with st.chat_message("assistant"):
            st.write(message["content"])


# Initialize processed PDF tracking
if "processed_pdfs" not in st.session_state:
    st.session_state.processed_pdfs = set()

# PDF uploader

# --------------------------------------------------
# PDF UPLOAD AND PERSISTENT INDEXING
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "＋ Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        # Unique identity for each PDF
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Check ChromaDB itself, not just session state
        existing_docs = vectorstore.get(
            where={"file_hash": file_hash},
            limit=1
        )

        if existing_docs["ids"]:
            st.info(
                f"{uploaded_file.name} is already indexed."
            )
            continue

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file_bytes)
            pdf_path = temp_file.name

        try:
            # Load PDF
            pdf_loader = PyPDFLoader(pdf_path)
            pdf_docs = pdf_loader.load()

            # Split into chunks
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            ).split_documents(pdf_docs)

            # Attach document metadata
            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name
                chunk.metadata["file_hash"] = file_hash

            # Generate deterministic chunk IDs
            ids = [
                hashlib.sha256(
                    f"{file_hash}_{i}".encode()
                ).hexdigest()
                for i in range(len(chunks))
            ]

            # Store in persistent ChromaDB
            vectorstore.add_documents(
                documents=chunks,
                ids=ids
            )

            st.success(
                f"Successfully indexed {uploaded_file.name} "
                f"({len(chunks)} chunks)"
            )

        except Exception as e:
            st.error(
                f"Error processing {uploaded_file.name}: {e}"
            )

        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)



# --------------------------------------------------
# CHAT EXECUTION
# --------------------------------------------------

from langchain_core.messages import SystemMessage, HumanMessage

user_prompt = st.chat_input("Ask anything...")

if user_prompt:

    with st.chat_message("user"):
        st.write(user_prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    with st.chat_message("assistant"):

        if selected_document:

            # Search ONLY the selected PDF
            docs = vectorstore.similarity_search(
                user_prompt,
                k=8,
                filter={
                    "source": selected_document
                }
            )

            # Debugging
            print("Selected document:", selected_document)
            print("Retrieved chunks:", len(docs))

            for doc in docs:
                print(doc.page_content)
                print("-" * 40)

            if not docs:
                answer = (
                    "I couldn't find relevant information "
                    f"in {selected_document}."
                )

                st.write(answer)

            else:

                context = "\n\n".join(
                    doc.page_content for doc in docs
                )

                messages = [
                    SystemMessage(content=f"""
                    You are a document analysis assistant.

                    The user has selected this document:
                    {selected_document}

                    Answer using ONLY the provided document context.

                    Do not introduce unrelated topics.
                    Do not invent information.
                    If information is missing, say so.

                    DOCUMENT CONTEXT:
                    {context}
                    """),

                    HumanMessage(content=user_prompt)
                ]

                response = llm.invoke(messages)

                answer = response.content
                st.write(answer)

        else:

            # General-purpose AI agent
            response = agent.stream(
                {"messages": st.session_state.messages},
                stream_mode="messages"
            )

            answer = st.write_stream(
                chunk.content
                for chunk, metadata in response
                if isinstance(chunk.content, str)
                and chunk.content
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })



# keep the sidebar message counter in sync after a new reply
render_hist_badge(hist_slot)