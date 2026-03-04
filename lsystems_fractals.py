"""
Streamlit app: Advanced L-systems + IFS Studio
Run:
    pip install streamlit numpy pillow
    streamlit run lsystems_fractals.py
"""

import streamlit as st
import io
import json

from PIL import Image

# --- Module imports ---
from presets import LSYS_PRESETS
from lsystem_core import generate_lsystem, interpret_turtle
from ifs_core import render_barnsley_fern, generate_barnsley_fern_points
from rendering import (
    segments_to_svg,
    segments_to_image,
    render_lsystem_advanced,
    make_phyllotaxis_asterisk,
)
from react_generator import (
    extract_lsystem_geometry,
    extract_ifs_geometry,
    build_threejs_params,
    generate_react_component,
    generate_iframe_html,
)


st.set_page_config(page_title="L-Systems & IFS Studio", layout="wide")

st.title("🌿 L-Systems & IFS Studio")
st.markdown("A single-file playground for L-systems (SVG & PNG) and IFS (Barnsley fern).")

mode = st.sidebar.radio("Mode", ["L-System (SVG/PNG)", "IFS (Barnsley Fern)"])

seed = st.sidebar.number_input("Random seed (0 = random)", min_value=0, value=0, step=1)
seed_val = None if seed == 0 else int(seed)

# --- React Component Settings (shared by both modes) ---
with st.sidebar.expander("React Component Settings"):
    animation_speed = st.slider("Animation Speed", 0.1, 2.0, 1.0, 0.1)
    parallax_strength = st.slider("Parallax Strength", 0.0, 1.0, 0.4, 0.05)
    glow_layers = st.slider("Glow Layers", 1, 3, 2)
    react_bg_color = st.color_picker("Background Color", "#0a0a0f")
    max_react_points = st.slider("Max Points (React)", 1000, 8000, 4000, 500,
                                  help="Higher = richer geometry but larger component file")
    react_colors_raw = st.text_input("Color palette (comma-separated hex)",
                                      value="#5227FF,#FF9FFC,#B19EEF")
    react_colors = [c.strip() for c in react_colors_raw.split(",") if c.strip()]
    if not react_colors:
        react_colors = ["#5227FF", "#FF9FFC", "#B19EEF"]



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

    tab_svg, tab_png, tab_react = st.tabs(["SVG Preview", "Download PNG", "React Component"])

    with tab_svg:
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

    with tab_png:
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

    with tab_react:
        if st.button("Generate React Component", key="lsys_react_btn"):
            with st.spinner("Generating geometry and React component..."):
                program = generate_lsystem(axiom, rules, iterations, seed_val)
                segs = interpret_turtle(program, step, angle)
                if not segs:
                    st.warning("No segments produced. Check axiom/rules.")
                else:
                    geo = extract_lsystem_geometry(segs, max_points=max_react_points)
                    label = f"{preset} — iter {iterations} — seed {seed}"
                    params_js = build_threejs_params(
                        geo, react_colors,
                        animation_speed=animation_speed,
                        parallax_strength=parallax_strength,
                        particle_size=0.008,
                        glow_layers=glow_layers,
                        bg_color=react_bg_color,
                        label=label,
                    )
                    react_src = generate_react_component(params_js, label=label)
                    iframe_html = generate_iframe_html(react_src, bg_color=react_bg_color)

                    st.subheader("Live Preview")
                    st.components.v1.html(iframe_html, height=500, scrolling=False)

                    st.subheader("Component Source")
                    st.code(react_src, language="jsx")

                    st.download_button(
                        "Download .jsx",
                        data=react_src,
                        file_name="GeneratedBackground.jsx",
                        mime="text/plain",
                    )
                    st.caption(
                        "Add to your React project. Run: `npm install three`. "
                        "Import and drop `<GeneratedBackground />` anywhere."
                    )

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

    tab_fern, tab_fern_react = st.tabs(["Fern Image", "React Component"])

    with tab_fern:
        if st.button("Generate Barnsley Fern"):
            with st.spinner("Generating fern..."):
                img = render_barnsley_fern(width, height, iterations)
                st.image(img, caption="Barnsley Fern", use_column_width=True)
                buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
                st.download_button("Download PNG", data=buf.getvalue(),
                                   file_name="barnsley_fern.png", mime="image/png")

    with tab_fern_react:
        if st.button("Generate React Component", key="ifs_react_btn"):
            with st.spinner("Generating fern geometry and React component..."):
                pts = generate_barnsley_fern_points(iterations)
                geo = extract_ifs_geometry(pts, max_points=max_react_points)
                label = f"Barnsley Fern — iter {iterations} — seed {seed}"
                params_js = build_threejs_params(
                    geo, react_colors,
                    animation_speed=animation_speed,
                    parallax_strength=parallax_strength,
                    particle_size=0.012,
                    glow_layers=glow_layers,
                    bg_color=react_bg_color,
                    label=label,
                )
                react_src = generate_react_component(params_js, label=label)
                iframe_html = generate_iframe_html(react_src, bg_color=react_bg_color)

                st.subheader("Live Preview")
                st.components.v1.html(iframe_html, height=500, scrolling=False)

                st.subheader("Component Source")
                st.code(react_src, language="jsx")

                st.download_button(
                    "Download .jsx",
                    data=react_src,
                    file_name="GeneratedBackground.jsx",
                    mime="text/plain",
                    key="ifs_react_download",
                )
                st.caption(
                    "Add to your React project. Run: `npm install three`. "
                    "Import and drop `<GeneratedBackground />` anywhere."
                )

st.markdown("---")
st.markdown("**Tips:** Increase iterations for more detail. Use the JSON rules editor for stochastic grammars. "
            "The 'Phyllotaxis' preset uses extended Julia-style commands with a custom `*` callback to draw colored discs.")