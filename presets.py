"""
L-System preset definitions.

Each preset is a dict with keys: axiom, rules, angle, step, iterations.
"""

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
