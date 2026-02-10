# 🌿 L-Systems & IFS Studio

> A powerful procedural art generator combining Lindenmayer Systems with Iterated Function Systems for creating stunning fractal and organic visualizations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.15+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

##  Features

- **Stochastic L-Systems**: Generate organic, tree-like structures with controllable randomness
- ** Fractal Rendering**: Create intricate snowflakes, curves, and geometric patterns
- ** Advanced Turtle Graphics**: Extended command set with color, thickness, and shape controls
- ** Phyllotaxis Patterns**: Specialized renderer for nature-inspired spiral arrangements
- ** IFS Support**: Barnsley Fern and other Iterated Function System fractals
- ** Batch Generation**: Create multiple variations with different random seeds
- ** Multiple Export Formats**: SVG (vector) and PNG (raster) output options
- ** Real-time Preview**: Interactive web interface with instant visualization
- ** Preset Library**: Pre-configured examples including plants, snowflakes, and geometric patterns

##  Built-in Presets

### Plants & Trees
- **Fractal Plant** - Classic deterministic branching structure
- **Bushy Tree** - Dense, multi-branch vegetation
- **Sparse Tree** - Long trunk with minimal branching
- **Grass/Weed Patch** - Ground cover simulation

### Geometric Patterns
- **Koch Curve** - Classic space-filling curve
- **Sierpinski Triangle** - Recursive triangular fractal
- **Hilbert Curve** - Space-filling Hilbert pattern

### Special Effects
- **Snowflake Crystal** - Six-fold symmetric dendrite
- **Phyllotaxis** - Golden angle spiral with colored discs

### Fractals
- **Barnsley Fern** - IFS-based fern simulation

##  Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
pip install streamlit numpy pillow
```

### Clone & Run

```bash
git clone <repository-url>
cd lsystems-ifs-studio
streamlit run streamlit_lsystems_fractals.py
```

##  Usage

### Basic Workflow

1. **Launch the Application**
   ```bash
   streamlit run streamlit_lsystems_fractals.py
   ```

2. **Choose a Mode**
   - L-System (SVG/PNG) for grammar-based generation
   - IFS (Barnsley Fern) for fractal rendering

3. **Select a Preset or Configure Custom Rules**
   - Use the sidebar to choose from built-in presets
   - Or define your own axiom and production rules

4. **Adjust Parameters**
   - Iterations: Control complexity depth
   - Angle: Set branching angles (degrees)
   - Step: Define movement distance
   - Colors: Customize stroke and background

5. **Generate & Export**
   - Click "Generate" for preview
   - Download SVG (vector) or PNG (raster)
   - Use batch mode for variations

##  L-System Syntax

### Core Turtle Commands

| Symbol | Action |
|--------|--------|
| `F`, `G` | Move forward and draw line |
| `f` | Move forward without drawing |
| `+` | Turn right by angle |
| `-` | Turn left by angle |
| `[` | Push state (position, angle) onto stack |
| `]` | Pop state from stack |

### Advanced Commands (Julia-style)

| Symbol | Action |
|--------|--------|
| `B`, `V` | Move backward and draw |
| `b` | Turn 180° and move forward |
| `@` | Turn right 5° |
| `&` | Turn left 5° |
| `r` | Random turn (10-60°) |
| `U` | Pen up (stop drawing) |
| `D` | Pen down (start drawing) |
| `l` | Increase step length |
| `s` | Decrease step length |
| `1-9` | Set line width |
| `n` | Set line width to 0.5 |
| `T` | Random hue |
| `t` | Increment hue |
| `c` | Random saturation |
| `o` | Draw circle |
| `q` | Draw rectangle |
| `*` | Special callback (Phyllotaxis) |

### Example Grammar

**Simple Tree:**
```json
{
  "F": [["FF-[-F+F+F]+[+F-F-F]", 1.0]]
}
```

**Stochastic Branching:**
```json
{
  "F": [["F[+F]F", 0.5], ["F[-F]F", 0.5]]
}
```

**Weighted Rules:**
```json
{
  "X": [["F[+X]", 0.6], ["F[-X]", 0.4]]
}
```

## ⚙️ Configuration Options

### L-System Parameters

- **Axiom**: Starting string/seed
- **Rules**: JSON-formatted production rules with probabilities
- **Iterations**: Number of recursive expansions (0-300)
- **Angle**: Turning angle in degrees (0-180)
- **Step**: Movement distance (0.1-40 units)
- **Stroke Width**: Line thickness (0.1-10 pixels)
- **Random Seed**: For reproducible stochastic output (0 = random)

### Output Settings

- **PNG Resolution**: Width and height (200-4000 pixels)
- **Colors**: Stroke and background color pickers
- **Batch Mode**: Generate 1-200 variations as ZIP archive

### IFS Parameters

- **Dimensions**: Width and height (200-1600 pixels)
- **Iterations**: Point count (1,000-500,000)

## 🎓 Example Use Cases

### Creating Custom Plant

```python
Axiom: "X"
Rules: {
  "X": [["F-[[X]+X]+F[+FX]-X", 1.0]],
  "F": [["FF", 1.0]]
}
Angle: 25°
Iterations: 6
```

### Stochastic Variation

```python
Axiom: "F"
Rules: {
  "F": [["F[+F]F[-F]F", 0.5], ["F[+F]", 0.25], ["F[-F]", 0.25]]
}
Random Seed: 0 (different each time)
```

### Phyllotaxis Spiral

```python
Axiom: "A"
Rules: {
  "A": [["A+[UFD*]ll", 1.0]]
}
Angle: 137.5° (golden angle)
Iterations: 200
```

##  Performance Notes

- **Simple L-Systems** (5 iterations): ~0.2s generation time
- **Stochastic Trees** (7 iterations): ~0.8s generation time
- **IFS Fractals** (100k points): ~1.5s rendering time
- **Hybrid Outputs**: ~2-3s for complex compositions

##  Technical Architecture

### Core Components

1. **L-System Engine**: Stochastic grammar expansion with weighted rule selection
2. **Turtle Interpreter**: Stack-based state machine for vector path generation
3. **Advanced Renderer**: Extended command set with HSL colors and shapes
4. **IFS Generator**: Iterated Function System implementation
5. **Export Pipeline**: SVG and PNG output with scaling and normalization

### Technology Stack

- **Framework**: Streamlit (web UI)
- **Computation**: NumPy (numerical operations)
- **Imaging**: Pillow/PIL (raster graphics)
- **Data**: JSON (grammar serialization)

## 🔧 Troubleshooting

**Issue**: String too large during expansion
- **Solution**: Reduce iteration count or simplify grammar rules

**Issue**: No segments produced
- **Solution**: Verify axiom contains drawable symbols (F, G) and rules expand correctly

**Issue**: Slow rendering
- **Solution**: Decrease output resolution or iteration count

##  Tips & Best Practices

- Start with low iterations (3-5) when experimenting
- Use stochastic rules for organic variation
- Combine `[` and `]` for complex branching
- Set random seed ≠ 0 for reproducible results
- Generate batches to explore parameter space
- Export SVG for infinite scalability
- Use Phyllotaxis preset for nature-inspired patterns

##  Contributing

Contributions are welcome! Areas for enhancement:

- Additional L-system preset libraries
- New fractal algorithms (Julia sets, Mandelbrot)
- 3D turtle graphics support
- Animation/time-series generation
- Interactive grammar editor
- Parameter optimization tools

## Example Images
<img width="705" height="575" alt="image" src="https://github.com/user-attachments/assets/d0a6b9e6-60fb-4fe4-99bb-d0b01d4754f6" />


##  Acknowledgments

Based on the theoretical foundations of:
- Aristid Lindenmayer (L-systems, 1968)
- Michael Barnsley (IFS theory)
- Turtle graphics interpretation methods

## 📧 Contact

---
