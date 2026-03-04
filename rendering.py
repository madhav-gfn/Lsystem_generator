"""
SVG, PNG, and advanced rendering utilities for L-systems.

Classes:
    TurtleState — dataclass holding turtle position, angle, pen, colour, etc.

Functions:
    hsl_to_rgb_tuple          — convert HSL to an (r, g, b) 0-255 tuple.
    segments_to_svg           — produce an SVG string from line segments.
    segments_to_image         — rasterize segments into a PIL Image.
    render_lsystem_advanced   — rich turtle renderer (pen up/down, colours, etc.).
    make_phyllotaxis_asterisk — factory for phyllotaxis disc-drawing callback.
"""

import math
import random
import colorsys
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageDraw

from lsystem_core import rescale


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
