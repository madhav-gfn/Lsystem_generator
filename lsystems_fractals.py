"""
Streamlit app: Advanced L-systems + IFS (single file)
Save as: streamlit_lsystems_fractals.py
Run:
    pip install streamlit numpy pillow
    streamlit run streamlit_lsystems_fractals.py
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import math
import random
from typing import List, Tuple, Dict
import json
from dataclasses import dataclass
import colorsys

st.set_page_config(page_title="L-Systems & IFS Studio", layout="wide")


def weighted_choice(choices, rng: random.Random):
    total = sum(w for _, w in choices)
    if total == 0:
        return choices[0][0]
    pick = rng.random() * total
    upto = 0
    for val, w in choices:
        if upto + w >= pick:
            return val
        upto += w
    return choices[-1][0]

def rescale(value, old_min, old_max, new_min, new_max):
    if old_max == old_min:
        return new_min
    t = (value - old_min) / (old_max - old_min)
    return new_min + t * (new_max - new_min)


def generate_lsystem(axiom: str, rules: Dict[str, List[Tuple[str, float]]],
                     iterations: int, seed: int=None, max_length: int=5_000_000) -> str:
    rng = random.Random(seed)
    current = axiom
    for i in range(iterations):
        if len(current) > max_length:
            st.warning(f"String too large at iteration {i}; stopping further expansion.")
            break
        out = []
        for c in current:
            if c in rules:
                out.append(weighted_choice(rules[c], rng))
            else:
                out.append(c)
        current = ''.join(out)
    return current


def interpret_turtle(program: str, step: float, angle_deg: float,
                     start=(0.0,0.0), start_angle=90.0):
    angle_rad = math.pi/180.0
    x, y = start
    angle = start_angle
    stack = []
    segs = []
    for ch in program:
        if ch == 'F' or ch == 'G':
            nx = x + step * math.cos(angle*angle_rad)
            ny = y - step * math.sin(angle*angle_rad)
            segs.append((x, y, nx, ny))
            x, y = nx, ny
        elif ch == 'f':
            x = x + step * math.cos(angle*angle_rad)
            y = y - step * math.sin(angle*angle_rad)
        elif ch == '+':
            angle -= angle_deg
        elif ch == '-':
            angle += angle_deg
        elif ch == '[':
            stack.append((x,y,angle))
        elif ch == ']':
            if stack:
                x,y,angle = stack.pop()
        else:
            pass
    return segs


def segments_to_svg(segs: List[Tuple[float,float,float,float]], stroke_width: float=1.0,
                    stroke_color: str="#000000", bg_color: str="#ffffff", padding: float=10.0) -> str:
    if not segs:
        return ""
    xs = [v for s in segs for v in (s[0], s[2])]
    ys = [v for s in segs for v in (s[1], s[3])]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = maxx - minx if maxx>minx else 1.0
    h = maxy - miny if maxy>miny else 1.0
    margin = max(w,h) * 0.03 + padding
    width = w + 2*margin
    height = h + 2*margin
    ox = minx - margin
    oy = miny - margin
    svg_parts = []
    svg_parts.append('<?xml version="1.0" encoding="utf-8"?>')
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg_parts.append(f'<rect width="100%" height="100%" fill="{bg_color}" />')
    svg_parts.append(f'<g transform="translate(0,{height}) scale(1,-1)">')
    svg_parts.append(f'<g stroke="{stroke_color}" fill="none" stroke-linecap="round" stroke-width="{stroke_width}">')
    for (x1,y1,x2,y2) in segs:
        X1 = x1 - ox
        Y1 = y1 - oy
        X2 = x2 - ox
        Y2 = y2 - oy
        svg_parts.append(f'<line x1="{X1:.3f}" y1="{Y1:.3f}" x2="{X2:.3f}" y2="{Y2:.3f}" />')
    svg_parts.append('</g></g></svg>')
    return '\n'.join(svg_parts)


def segments_to_image(segs: List[Tuple[float,float,float,float]], img_size=(1000,1000),
                      stroke_width: float=1.0, stroke_color: str="#000000", bg_color: str="#ffffff",
                      padding=20):
    if not segs:
        return Image.new('RGB', img_size, color=bg_color)
    xs = [v for s in segs for v in (s[0], s[2])]
    ys = [v for s in segs for v in (s[1], s[3])]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = maxx - minx if maxx>minx else 1.0
    h = maxy - miny if maxy>miny else 1.0
    imgw, imgh = img_size
    sx = (imgw - 2*padding) / w
    sy = (imgh - 2*padding) / h
    s = min(sx, sy)
    def to_px(x,y):
        px = int((x - minx) * s + padding)
        py = int((maxy - y) * s + padding)
        return px, py
    img = Image.new('RGB', (imgw, imgh), color=bg_color)
    draw = ImageDraw.Draw(img)
    for (x1,y1,x2,y2) in segs:
        p1 = to_px(x1,y1)
        p2 = to_px(x2,y2)
        draw.line([p1,p2], fill=stroke_color, width=max(1, int(stroke_width)))
    return img


@dataclass
class TurtleState:
    x: float
    y: float
    angle: float
    step: float
    line_width: float
    pen_down: bool = True
    hue: float = 0.0
    sat: float = 0.8
    light: float = 0.5
    opacity: float = 1.0

def hsl_to_rgb_tuple(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return int(r * 255), int(g * 255), int(b * 255)

def render_lsystem_advanced(program: str,
                            width: int,
                            height: int,
                            angle_deg: float,
                            base_step: float,
                            bg_color: str = "white",
                            asterisk_func=None) -> Image.Image:
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    t = TurtleState(x=width/2.0, y=height/2.0, angle=90.0, step=base_step, line_width=1.0)
    stack: List[TurtleState] = []

    def forward(step_mult=1.0, backwards=False):
        nonlocal t
        step = t.step * step_mult
        theta = math.radians(t.angle + (180.0 if backwards else 0.0))
        new_x = t.x + step * math.cos(theta)
        new_y = t.y - step * math.sin(theta)
        color = hsl_to_rgb_tuple(t.hue, t.sat, t.light)
        if t.pen_down:
            draw.line([(t.x, t.y), (new_x, new_y)],
                      fill=color,
                      width=max(1, int(round(t.line_width))))
        t.x, t.y = new_x, new_y

    for ch in program:
        if ch in ("F", "G"):
            forward(1.0, backwards=False)
        elif ch == "f":
            forward(0.5, backwards=False)
        elif ch in ("B", "V"):
            forward(1.0, backwards=True)
        elif ch == "b":
            t.angle = (t.angle + 180.0) % 360.0
            forward(0.5, backwards=False)

        elif ch == "+":
            t.angle = (t.angle + angle_deg) % 360.0
        elif ch == "-":
            t.angle = (t.angle - angle_deg) % 360.0
        elif ch == "@":
            t.angle = (t.angle + 5.0) % 360.0
        elif ch == "&":
            t.angle = (t.angle - 5.0) % 360.0
        elif ch == "r":
            delta = random.choice([10, 15, 30, 45, 60])
            if random.random() < 0.5:
                delta = -delta
            t.angle = (t.angle + delta) % 360.0

        elif ch == "[":
            stack.append(TurtleState(**t.__dict__))
        elif ch == "]":
            if stack:
                t = stack.pop()

        elif ch == "U":
            t.pen_down = False
        elif ch == "D":
            t.pen_down = True

        elif ch == "l":
            t.step += 1.0
        elif ch == "s":
            t.step = max(1.0, t.step - 1.0)

        elif ch in "123456789":
            t.line_width = float(int(ch))
        elif ch == "n":
            t.line_width = 0.5

        elif ch == "T":
            t.hue = random.uniform(0.0, 360.0)
        elif ch == "t":
            t.hue = (t.hue + 5.0) % 360.0
        elif ch == "c":
            t.sat = random.uniform(0.0, 1.0)
        elif ch == "O":
            t.opacity = random.uniform(0.2, 1.0)

        elif ch == "o":
            r = t.step / 4.0
            color = hsl_to_rgb_tuple(t.hue, t.sat, t.light)
            bbox = [t.x - r, t.y - r, t.x + r, t.y + r]
            draw.ellipse(bbox, fill=color)
        elif ch == "q":
            side = t.step / 4.0
            color = hsl_to_rgb_tuple(t.hue, t.sat, t.light)
            bbox = [t.x - side/2, t.y - side/2, t.x + side/2, t.y + side/2]
            draw.rectangle(bbox, outline=color)

        elif ch == "*":
            if asterisk_func is not None:
                asterisk_func(t, draw)

        else:
            pass

    return img

def make_phyllotaxis_asterisk(width: int, height: int):
    center = (width / 2.0, height / 2.0)
    counter = {"value": 0}

    def asterisk_func(turtle: TurtleState, draw: ImageDraw.ImageDraw):
        d = math.dist((turtle.x, turtle.y), center)
        r = rescale(d, 1.0, min(width, height) / 2.0, 3.0, 15.0)
        h = counter["value"] % 360
        rgb = hsl_to_rgb_tuple(h, 0.8, 0.5)
        bbox = [turtle.x - r, turtle.y - r, turtle.x + r, turtle.y + r]
        draw.ellipse(bbox, fill=rgb)
        counter["value"] += 1

    return asterisk_func



def render_barnsley_fern(width: int, height: int, iterations: int=100000, color=(0,120,0)) -> Image.Image:
    transforms = [
        (0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01),
        (0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85),
        (0.2, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07),
        (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07)
    ]
    probs = np.array([t[6] for t in transforms])
    probs = probs / probs.sum()
    x, y = 0.0, 0.0
    xs = []
    ys = []
    for i in range(iterations):
        r = random.random()
        acc = 0.0
        for p, pr in enumerate(probs):
            acc += pr
            if r <= acc:
                a11,a12,a21,a22,a13,a23,_ = transforms[p]
                break
        xn = a11*x + a12*y + a13
        yn = a21*x + a22*y + a23
        x, y = xn, yn
        xs.append(x); ys.append(y)
    xs = np.array(xs); ys = np.array(ys)
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    xs_scaled = (xs - minx) / (maxx - minx + 1e-12)
    ys_scaled = (ys - miny) / (maxy - miny + 1e-12)
    for xi, yi in zip(xs_scaled, ys_scaled):
        px = int(xi * (width-1))
        py = int((1-yi) * (height-1))
        draw.point((px, py), fill=color)
    return img


LSYS_PRESETS = {
    "Fractal Plant (Deterministic)": {
        "axiom": "X",
        "rules": {
            "X": [("F-[[X]+X]+F[+FX]-X", 1.0)],
            "F": [("FF", 1.0)]
        },
        "angle": 25.0,
        "step": 5.0,
        "iterations": 6
    },

    "Bushy Tree": {
        "axiom": "F",
        "rules": {
            "F": [("FF-[-F+F+F]+[+F-F-F]", 1.0)]
        },
        "angle": 22.5,
        "step": 5.0,
        "iterations": 4
    },

    "Sparse Tree (Long Trunk)": {
        "axiom": "F",
        "rules": {
            "F": [("F[+F]F", 0.5), ("F[-F]F", 0.5)]
        },
        "angle": 18.0,
        "step": 7.0,
        "iterations": 6
    },

    "Grass / Weed Patch": {
        "axiom": "F",
        "rules": {
            "F": [("F[+F]F[-F]F", 0.5), ("F[+F]", 0.25), ("F[-F]", 0.25)]
        },
        "angle": 12.0,
        "step": 4.0,
        "iterations": 6
    },

    "Snowflake Crystal 1 (Dendrite)": {
        "axiom": "A++A++A++A++A++A",
        "rules": {
            "A": [("F[+B][-B]A", 1.0)],
            "B": [("F[+C][-C]", 1.0)],
            "C": [("F", 1.0)],
            "F": [("F", 1.0)]
        },
        "angle": 60.0,
        "step": 5.0,
        "iterations": 3
    },

    "Phyllotaxis": {
        "axiom": "A",
        "rules": {
            "A": [("A+[UFD*]ll", 1.0)]
        },
        "angle": 137.5,
        "step": 5.0,
        "iterations": 200
    },
    
    "Koch Curve": {
        "axiom": "F--F--F",
        "rules": {
            "F": [("F+F--F+F", 1.0)]
        },
        "angle": 60.0,
        "step": 6.0,
        "iterations": 4
    },

    "Sierpinski Triangle": {
        "axiom": "F-G-G",
        "rules": {
            "F": [("F-G+F+G-F", 1.0)],
            "G": [("GG", 1.0)]
        },
        "angle": 120.0,
        "step": 6.0,
        "iterations": 6
    },

    "Hilbert Curve": {
        "axiom": "A",
        "rules": {
            "A": [("+BF-AFA-FB+", 1.0)],
            "B": [("-AF+BFB+FA-", 1.0)]
        },
        "angle": 90.0,
        "step": 5.0,
        "iterations": 5
    },

}


st.title("🌿 L-Systems & IFS Studio")
st.markdown("A single-file playground for L-systems (SVG & PNG) and IFS (Barnsley fern).")

mode = st.sidebar.radio("Mode", ["L-System (SVG/PNG)", "IFS (Barnsley Fern)"])

seed = st.sidebar.number_input("Random seed (0 = random)", min_value=0, value=0, step=1)
seed_val = None if seed == 0 else int(seed)


if mode == "L-System (SVG/PNG)":
    st.sidebar.header("L-System Preset")
    preset = st.sidebar.selectbox("Choose preset", list(LSYS_PRESETS.keys()))
    preset_data = LSYS_PRESETS[preset]

    axiom = st.sidebar.text_input("Axiom", value=preset_data["axiom"])

    rules_json = st.sidebar.text_area(
        "Rules (JSON: symbol -> [[replacement,weight],...])",
        value=json.dumps({k: v for k,v in preset_data["rules"].items()}, indent=2),
        height=220
    )

    angle = st.sidebar.slider("Angle (degrees)", 0.0, 180.0, value=float(preset_data["angle"]))
    step = st.sidebar.slider("Step (units)", 0.1, 40.0, value=float(preset_data["step"]))
    iterations = st.sidebar.slider("Iterations", 0, 300, value=int(preset_data["iterations"]))
    stroke_width = st.sidebar.slider("Stroke width (px)", 0.1, 10.0, value=1.0)
    stroke_color = st.sidebar.color_picker("Stroke color", value="#003300")
    bg_color = st.sidebar.color_picker("Background color", value="#ffffff")
    out_width = st.sidebar.slider("Export PNG width (px)", 200, 4000, 1200)
    out_height = st.sidebar.slider("Export PNG height (px)", 200, 4000, 1200)

    try:
        rules_obj = json.loads(rules_json)
        rules = {}
        for k, v in rules_obj.items():
            normalized = []
            for item in v:
                if isinstance(item, (list, tuple)):
                    rhs = str(item[0]); w = float(item[1]) if len(item) > 1 else 1.0
                else:
                    rhs = str(item); w = 1.0
                normalized.append((rhs, w))
            rules[str(k)] = normalized
    except Exception as e:
        st.sidebar.error(f"Failed to parse rules JSON: {e}")
        rules = {k: [(vv[0], vv[1]) for vv in preset_data['rules'][k]] for k in preset_data['rules']}

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Generate L-system (SVG preview)"):
            with st.spinner("Expanding grammar..."):
                program = generate_lsystem(axiom, rules, iterations, seed_val)
                st.success(f"Final string length: {len(program)}")
                segs = interpret_turtle(program, step, angle)
                svg = segments_to_svg(segs, stroke_width=stroke_width, stroke_color=stroke_color, bg_color=bg_color)
                if svg:
                    st.subheader("SVG Preview")
                    st.components.v1.html(svg, height=650)
                    st.download_button("Download SVG", data=svg.encode('utf-8'),
                                       file_name="lsystem.svg", mime="image/svg+xml")
                else:
                    st.warning("No segments produced. Check axiom/rules.")
    with col2:
        if st.button("Generate L-system (PNG raster)"):
            with st.spinner("Expanding grammar and rasterizing..."):
                program = generate_lsystem(axiom, rules, iterations, seed_val)

                if preset == "Phyllotaxis":
                    fx = make_phyllotaxis_asterisk(out_width, out_height)
                    img = render_lsystem_advanced(
                        program,
                        width=out_width,
                        height=out_height,
                        angle_deg=angle,
                        base_step=step,
                        bg_color=bg_color,
                        asterisk_func=fx
                    )
                else:
                    segs = interpret_turtle(program, step, angle)
                    img = segments_to_image(
                        segs,
                        img_size=(out_width, out_height),
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        bg_color=bg_color
                        
                    )
                    img = img.transpose(Image.ROTATE_180)

                st.image(img, caption="Raster Preview", use_column_width=True)
                buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
                st.download_button("Download PNG", data=buf.getvalue(),
                                   file_name="lsystem.png", mime="image/png")

    st.markdown("### Batch / variations")
    batch_count = st.number_input("Generate N variations (batch)", min_value=1, max_value=200, value=1, step=1)
    if st.button("Generate batch"):
        out_zip_buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(out_zip_buf, mode="w") as zf:
            for i in range(batch_count):
                s = None if seed_val is None else seed_val + i + 1
                program = generate_lsystem(axiom, rules, iterations, s)

                if preset == "Phyllotaxis":
                    fx = make_phyllotaxis_asterisk(out_width, out_height)
                    img = render_lsystem_advanced(
                        program,
                        width=out_width,
                        height=out_height,
                        angle_deg=angle,
                        base_step=step,
                        bg_color=bg_color,
                        asterisk_func=fx
                    )
                else:
                    segs = interpret_turtle(program, step, angle)
                    img = segments_to_image(
                        segs,
                        img_size=(out_width, out_height),
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        bg_color=bg_color
                    )

                b = io.BytesIO(); img.save(b, format='PNG'); b.seek(0)
                zf.writestr(f"lsystem_{i+1}.png", b.getvalue())
        out_zip_buf.seek(0)
        st.download_button("Download batch ZIP", data=out_zip_buf.getvalue(),
                           file_name="lsystem_batch.zip", mime="application/zip")

    st.markdown("---")
    st.markdown("**Notes:** Use `[` and `]` for branching. For stochastic rules provide JSON like `{\"X\": [[\"F[+X]\",0.6],[\"F[-X]\",0.4]]}`.")


else:
    st.sidebar.header("Barnsley Fern (IFS)")
    width = st.sidebar.slider("Width (px)", 200, 1600, 800)
    height = st.sidebar.slider("Height (px)", 200, 1200, 600)
    iterations = st.sidebar.slider("Iterations", 1000, 500000, 100000)
    if st.sidebar.button("Generate Barnsley Fern"):
        with st.spinner("Generating fern..."):
            img = render_barnsley_fern(width, height, iterations)
            st.image(img, caption="Barnsley Fern", use_column_width=True)
            buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
            st.download_button("Download PNG", data=buf.getvalue(), file_name="barnsley_fern.png", mime="image/png")

st.markdown("---")
st.markdown("**Tips:** Increase iterations for more detail. Use the JSON rules editor for stochastic grammars. "
            "The 'Phyllotaxis' preset uses extended Julia-style commands with a custom `*` callback to draw colored discs.")