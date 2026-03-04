"""
Core L-system grammar expansion and turtle interpretation.

Functions:
    weighted_choice  — pick from weighted alternatives.
    rescale          — linear remapping helper.
    generate_lsystem — expand an L-system grammar string.
    interpret_turtle — convert an L-system string into line segments.
"""

import math
import random
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


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
            logger.warning(f"String too large at iteration {i}; stopping further expansion.")
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
