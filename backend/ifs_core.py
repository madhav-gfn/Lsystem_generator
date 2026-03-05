"""
Iterated Function Systems — Barnsley Fern.

Functions:
    render_barnsley_fern         — generate a PIL Image of the fern.
    generate_barnsley_fern_points — return raw (x, y) points (no image).
"""

import random

import numpy as np
from PIL import Image, ImageDraw


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


def generate_barnsley_fern_points(iterations=100000):
    """Run the Barnsley Fern chaos game and return raw (x, y) points.

    This is a data-only version of render_barnsley_fern — no image is
    produced.  The points are in the original IFS coordinate space.

    Args:
        iterations: Number of chaos-game iterations.

    Returns:
        List of (x, y) tuples.
    """
    transforms = [
        (0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01),
        (0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85),
        (0.2, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07),
        (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07),
    ]
    probs = np.array([t[6] for t in transforms])
    probs = probs / probs.sum()
    x, y = 0.0, 0.0
    result = []
    for _ in range(iterations):
        r = random.random()
        acc = 0.0
        a11 = a12 = a21 = a22 = a13 = a23 = 0.0
        for p_idx, pr in enumerate(probs):
            acc += pr
            if r <= acc:
                a11, a12, a21, a22, a13, a23, _ = transforms[p_idx]
                break
        xn = a11 * x + a12 * y + a13
        yn = a21 * x + a22 * y + a23
        x, y = xn, yn
        result.append((x, y))
    return result
