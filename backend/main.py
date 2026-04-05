"""
FastAPI backend for L-Systems & IFS Studio.

Run:
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""

import io
import json
import zipfile
import base64
import logging
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

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
)

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

app = FastAPI(title="L-Systems & IFS Studio API", version="1.0.0")

_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
_origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LSystemParams(BaseModel):
    axiom: str = "X"
    rules: Dict[str, list] = Field(default_factory=dict)
    angle: float = 25.0
    step: float = 5.0
    iterations: int = 6
    seed: Optional[int] = None
    stroke_width: float = 1.0
    stroke_color: str = "#003300"
    bg_color: str = "#ffffff"
    out_width: int = 1200
    out_height: int = 1200
    preset_name: Optional[str] = None


class ReactParams(BaseModel):
    colors: List[str] = ["#5227FF", "#FF9FFC", "#B19EEF"]
    animation_speed: float = 1.0
    parallax_strength: float = 0.4
    glow_layers: int = 2
    bg_color: str = "#0a0a0f"
    max_points: int = 4000
    particle_size: float = 0.008


class LSystemReactRequest(BaseModel):
    lsystem: LSystemParams = Field(default_factory=LSystemParams)
    react: ReactParams = Field(default_factory=ReactParams)


class IFSParams(BaseModel):
    width: int = 800
    height: int = 600
    iterations: int = 100000
    color: List[int] = [0, 120, 0]


class IFSReactRequest(BaseModel):
    iterations: int = 100000
    react: ReactParams = Field(default_factory=ReactParams)


class BatchParams(BaseModel):
    lsystem: LSystemParams = Field(default_factory=LSystemParams)
    count: int = 1


# ---------------------------------------------------------------------------
# Helper: normalize rules from JSON format
# ---------------------------------------------------------------------------

def _normalize_rules(rules_raw: Dict[str, list]) -> Dict[str, List[Tuple[str, float]]]:
    """Convert rules from JSON format to the internal tuple format."""
    rules = {}
    for k, v in rules_raw.items():
        normalized = []
        for item in v:
            if isinstance(item, (list, tuple)):
                rhs = str(item[0])
                w = float(item[1]) if len(item) > 1 else 1.0
            else:
                rhs = str(item)
                w = 1.0
            normalized.append((rhs, w))
        rules[str(k)] = normalized
    return rules


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.head("/api/presets")
def health_check():
    return JSONResponse(content={"status": "ok"})

@app.get("/api/presets")
def get_presets():
    """Return all L-system presets."""
    return JSONResponse(content=LSYS_PRESETS)


@app.post("/api/lsystem/svg")
def lsystem_svg(params: LSystemParams):
    """Generate L-system and return SVG string."""
    rules = _normalize_rules(params.rules)
    seed_val = None if params.seed == 0 or params.seed is None else params.seed

    program = generate_lsystem(params.axiom, rules, params.iterations, seed_val)
    segs = interpret_turtle(program, params.step, params.angle)

    if not segs:
        raise HTTPException(status_code=400, detail="No segments produced. Check axiom/rules.")

    svg = segments_to_svg(
        segs,
        stroke_width=params.stroke_width,
        stroke_color=params.stroke_color,
        bg_color=params.bg_color,
    )
    return JSONResponse(content={
        "svg": svg,
        "string_length": len(program),
        "segment_count": len(segs),
    })


@app.post("/api/lsystem/png")
def lsystem_png(params: LSystemParams):
    """Generate L-system PNG and return as base64 or stream."""
    rules = _normalize_rules(params.rules)
    seed_val = None if params.seed == 0 or params.seed is None else params.seed

    program = generate_lsystem(params.axiom, rules, params.iterations, seed_val)

    if params.preset_name == "Phyllotaxis":
        fx = make_phyllotaxis_asterisk(params.out_width, params.out_height)
        img = render_lsystem_advanced(
            program,
            width=params.out_width,
            height=params.out_height,
            angle_deg=params.angle,
            base_step=params.step,
            bg_color=params.bg_color,
            asterisk_func=fx,
        )
    else:
        segs = interpret_turtle(program, params.step, params.angle)
        img = segments_to_image(
            segs,
            img_size=(params.out_width, params.out_height),
            stroke_width=params.stroke_width,
            stroke_color=params.stroke_color,
            bg_color=params.bg_color,
        )
        img = img.transpose(Image.ROTATE_180)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Content-Disposition": "attachment; filename=lsystem.png"})


@app.post("/api/lsystem/geometry")
def lsystem_geometry(request: LSystemReactRequest):
    """Generate L-system and return normalized geometry JSON for preview."""
    p = request.lsystem
    rules = _normalize_rules(p.rules)
    seed_val = None if p.seed == 0 or p.seed is None else p.seed

    program = generate_lsystem(p.axiom, rules, p.iterations, seed_val)
    segs = interpret_turtle(program, p.step, p.angle)

    if not segs:
        raise HTTPException(status_code=400, detail="No segments produced. Check axiom/rules.")

    geo = extract_lsystem_geometry(segs, max_points=request.react.max_points)

    params_dict = {
        **geo,
        "colors": request.react.colors,
        "animationSpeed": request.react.animation_speed,
        "parallaxStrength": request.react.parallax_strength,
        "particleSize": request.react.particle_size,
        "lineWidth": 1,
        "glowLayers": request.react.glow_layers,
        "backgroundColor": request.react.bg_color,
    }

    return JSONResponse(content=params_dict)


@app.post("/api/lsystem/react")
def lsystem_react(request: LSystemReactRequest):
    """Generate React component JSX source for L-system."""
    p = request.lsystem
    rules = _normalize_rules(p.rules)
    seed_val = None if p.seed == 0 or p.seed is None else p.seed

    program = generate_lsystem(p.axiom, rules, p.iterations, seed_val)
    segs = interpret_turtle(program, p.step, p.angle)

    if not segs:
        raise HTTPException(status_code=400, detail="No segments produced. Check axiom/rules.")

    geo = extract_lsystem_geometry(segs, max_points=request.react.max_points)
    label = f"{p.preset_name or 'Custom'} — iter {p.iterations} — seed {p.seed or 0}"
    params_js = build_threejs_params(
        geo, request.react.colors,
        animation_speed=request.react.animation_speed,
        parallax_strength=request.react.parallax_strength,
        particle_size=request.react.particle_size,
        glow_layers=request.react.glow_layers,
        bg_color=request.react.bg_color,
        label=label,
    )
    react_src = generate_react_component(params_js, label=label)
    return JSONResponse(content={"jsx": react_src, "label": label})


@app.post("/api/lsystem/batch")
def lsystem_batch(params: BatchParams):
    """Generate batch of L-system PNGs and return as ZIP."""
    p = params.lsystem
    rules = _normalize_rules(p.rules)
    base_seed = None if p.seed == 0 or p.seed is None else p.seed

    out_zip_buf = io.BytesIO()
    with zipfile.ZipFile(out_zip_buf, mode="w") as zf:
        for i in range(params.count):
            s = None if base_seed is None else base_seed + i + 1
            program = generate_lsystem(p.axiom, rules, p.iterations, s)

            if p.preset_name == "Phyllotaxis":
                fx = make_phyllotaxis_asterisk(p.out_width, p.out_height)
                img = render_lsystem_advanced(
                    program,
                    width=p.out_width,
                    height=p.out_height,
                    angle_deg=p.angle,
                    base_step=p.step,
                    bg_color=p.bg_color,
                    asterisk_func=fx,
                )
            else:
                segs = interpret_turtle(program, p.step, p.angle)
                img = segments_to_image(
                    segs,
                    img_size=(p.out_width, p.out_height),
                    stroke_width=p.stroke_width,
                    stroke_color=p.stroke_color,
                    bg_color=p.bg_color,
                )

            b = io.BytesIO()
            img.save(b, format="PNG")
            b.seek(0)
            zf.writestr(f"lsystem_{i+1}.png", b.getvalue())

    out_zip_buf.seek(0)
    return StreamingResponse(
        out_zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=lsystem_batch.zip"},
    )


@app.post("/api/ifs/png")
def ifs_png(params: IFSParams):
    """Generate Barnsley Fern PNG."""
    color_tuple = tuple(params.color[:3]) if len(params.color) >= 3 else (0, 120, 0)
    img = render_barnsley_fern(params.width, params.height, params.iterations, color=color_tuple)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Content-Disposition": "attachment; filename=barnsley_fern.png"})


@app.post("/api/ifs/geometry")
def ifs_geometry(request: IFSReactRequest):
    """Generate IFS geometry JSON for preview."""
    pts = generate_barnsley_fern_points(request.iterations)
    geo = extract_ifs_geometry(pts, max_points=request.react.max_points)

    params_dict = {
        **geo,
        "colors": request.react.colors,
        "animationSpeed": request.react.animation_speed,
        "parallaxStrength": request.react.parallax_strength,
        "particleSize": 0.012,
        "lineWidth": 1,
        "glowLayers": request.react.glow_layers,
        "backgroundColor": request.react.bg_color,
    }

    return JSONResponse(content=params_dict)


@app.post("/api/ifs/react")
def ifs_react(request: IFSReactRequest):
    """Generate IFS React component JSX source."""
    pts = generate_barnsley_fern_points(request.iterations)
    geo = extract_ifs_geometry(pts, max_points=request.react.max_points)
    label = f"Barnsley Fern — iter {request.iterations}"
    params_js = build_threejs_params(
        geo, request.react.colors,
        animation_speed=request.react.animation_speed,
        parallax_strength=request.react.parallax_strength,
        particle_size=0.012,
        glow_layers=request.react.glow_layers,
        bg_color=request.react.bg_color,
        label=label,
    )
    react_src = generate_react_component(params_js, label=label)
    return JSONResponse(content={"jsx": react_src, "label": label})
