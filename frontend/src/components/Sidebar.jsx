import { useCallback } from 'react';

export default function Sidebar({
    mode, onModeChange,
    presets, presetNames,
    lsystemParams, onLSystemChange,
    reactParams, onReactChange,
    ifsParams, onIFSChange,
}) {
    const handlePresetSelect = useCallback((name) => {
        const p = presets[name];
        if (!p) return;
        onLSystemChange({
            preset_name: name,
            axiom: p.axiom,
            rules: p.rules,
            angle: p.angle,
            step: p.step,
            iterations: p.iterations,
        });
    }, [presets, onLSystemChange]);

    // Parse rules into editable rows
    const ruleEntries = Object.entries(lsystemParams.rules || {});

    const handleRuleChange = (variable, newProduction) => {
        const newRules = { ...lsystemParams.rules };
        newRules[variable] = [[newProduction, 1.0]];
        onLSystemChange({ rules: newRules });
    };

    return (
        <aside className="sidebar">
            {/* Header */}
            <div className="sidebar-header">
                <h2>
                    <span className="material-symbols-outlined" style={{ fontSize: 18 }}>terminal</span>
                    Parameters
                </h2>
                <div className="sidebar-status">
                    <div className="dot active"></div>
                    <div className="dot inactive"></div>
                </div>
            </div>

            {/* Mode Tabs */}
            <div className="mode-toggle">
                <button
                    className={mode === 'lsystem' ? 'active' : ''}
                    onClick={() => onModeChange('lsystem')}
                >
                    L-System
                </button>
                <button
                    className={mode === 'ifs' ? 'active' : ''}
                    onClick={() => onModeChange('ifs')}
                >
                    IFS Matrix
                </button>
            </div>

            {/* Scrollable Controls */}
            <div className="sidebar-controls">
                {/* L-System Controls */}
                {mode === 'lsystem' && (
                    <>
                        {/* Preset */}
                        <div className="control-group">
                            <label>
                                <span>Preset</span>
                                <span className="tag">SELECT</span>
                            </label>
                            <select
                                value={lsystemParams.preset_name || ''}
                                onChange={(e) => handlePresetSelect(e.target.value)}
                            >
                                {presetNames.map(name => (
                                    <option key={name} value={name}>{name}</option>
                                ))}
                            </select>
                        </div>

                        {/* Axiom */}
                        <div className="control-group">
                            <label>
                                <span>Axiom</span>
                                <span className="tag">START</span>
                            </label>
                            <div className="input-decorated">
                                <input
                                    type="text"
                                    value={lsystemParams.axiom}
                                    onChange={(e) => onLSystemChange({ axiom: e.target.value })}
                                />
                            </div>
                        </div>

                        {/* Production Rules */}
                        <div className="control-group">
                            <label>
                                <span>Production Rules</span>
                                <button className="material-symbols-outlined" style={{
                                    fontSize: 16, color: 'rgba(230,25,25,0.7)', background: 'none',
                                    border: 'none', cursor: 'pointer'
                                }}>add_box</button>
                            </label>
                            <div className="rules-container">
                                {ruleEntries.map(([variable, productions]) => (
                                    <div className="rule-row" key={variable}>
                                        <span className="rule-var">{variable}</span>
                                        <span className="material-symbols-outlined rule-arrow">arrow_forward</span>
                                        <input
                                            type="text"
                                            value={Array.isArray(productions) ? productions.map(p => p[0]).join(' | ') : String(productions)}
                                            onChange={(e) => handleRuleChange(variable, e.target.value)}
                                        />
                                        <button className="rule-delete">
                                            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>delete</span>
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Numeric Params: Turn Angle + Iterations */}
                        <div className="control-row">
                            <div className="control-group">
                                <label>Turn Angle (θ)</label>
                                <div className="input-with-unit">
                                    <input
                                        type="number"
                                        value={lsystemParams.angle}
                                        onChange={(e) => onLSystemChange({ angle: parseFloat(e.target.value) })}
                                    />
                                    <span className="unit">°</span>
                                </div>
                            </div>
                            <div className="control-group">
                                <label>Iterations</label>
                                <div className="numeric-stepper">
                                    <button
                                        className="stepper-btn left"
                                        onClick={() => onLSystemChange({ iterations: Math.max(0, lsystemParams.iterations - 1) })}
                                    >−</button>
                                    <input
                                        type="number"
                                        value={lsystemParams.iterations}
                                        min="0"
                                        max="300"
                                        onChange={(e) => onLSystemChange({ iterations: parseInt(e.target.value) })}
                                    />
                                    <button
                                        className="stepper-btn right"
                                        onClick={() => onLSystemChange({ iterations: Math.min(300, lsystemParams.iterations + 1) })}
                                    >+</button>
                                </div>
                            </div>
                        </div>

                        {/* Step */}
                        <div className="control-group">
                            <label>
                                <span>Step Length</span>
                                <span className="tag">TURTLE</span>
                            </label>
                            <div className="input-with-unit">
                                <input
                                    type="number"
                                    value={lsystemParams.step}
                                    step="0.1"
                                    min="0.1"
                                    max="40"
                                    onChange={(e) => onLSystemChange({ step: parseFloat(e.target.value) })}
                                />
                                <span className="unit">px</span>
                            </div>
                        </div>

                        {/* Render Settings */}
                        <div className="section-divider">
                            <h3>Render Settings</h3>

                            <div className="color-swatch-row">
                                <label>Stroke Color</label>
                                <div className="swatch-group">
                                    <input
                                        className="swatch-hex"
                                        type="text"
                                        value={lsystemParams.stroke_color}
                                        readOnly
                                    />
                                    <input
                                        type="color"
                                        value={lsystemParams.stroke_color}
                                        onChange={(e) => onLSystemChange({ stroke_color: e.target.value })}
                                        style={{
                                            width: 24, height: 24,
                                            border: '1px solid rgba(230,25,25,0.5)',
                                            boxShadow: '0 0 10px rgba(230,25,25,0.4)',
                                            background: lsystemParams.stroke_color,
                                            cursor: 'pointer', padding: 0
                                        }}
                                    />
                                </div>
                            </div>

                            <div className="range-wrapper">
                                <div className="range-header">
                                    <label>Line Width</label>
                                    <span className="range-value">{lsystemParams.stroke_width}px</span>
                                </div>
                                <input
                                    type="range"
                                    min="0.1"
                                    max="10"
                                    step="0.1"
                                    value={lsystemParams.stroke_width}
                                    onChange={(e) => onLSystemChange({ stroke_width: parseFloat(e.target.value) })}
                                />
                            </div>

                            <div className="color-swatch-row">
                                <label>Background</label>
                                <div className="swatch-group">
                                    <input
                                        className="swatch-hex"
                                        type="text"
                                        value={lsystemParams.bg_color}
                                        readOnly
                                    />
                                    <input
                                        type="color"
                                        value={lsystemParams.bg_color}
                                        onChange={(e) => onLSystemChange({ bg_color: e.target.value })}
                                        style={{
                                            width: 24, height: 24,
                                            border: '1px solid rgba(230,25,25,0.5)',
                                            background: lsystemParams.bg_color,
                                            cursor: 'pointer', padding: 0
                                        }}
                                    />
                                </div>
                            </div>

                            <div className="control-row">
                                <div className="control-group">
                                    <label>Width</label>
                                    <div className="input-with-unit">
                                        <input
                                            type="number"
                                            min="200"
                                            max="4000"
                                            step="100"
                                            value={lsystemParams.out_width}
                                            onChange={(e) => onLSystemChange({ out_width: parseInt(e.target.value) })}
                                        />
                                        <span className="unit">px</span>
                                    </div>
                                </div>
                                <div className="control-group">
                                    <label>Height</label>
                                    <div className="input-with-unit">
                                        <input
                                            type="number"
                                            min="200"
                                            max="4000"
                                            step="100"
                                            value={lsystemParams.out_height}
                                            onChange={(e) => onLSystemChange({ out_height: parseInt(e.target.value) })}
                                        />
                                        <span className="unit">px</span>
                                    </div>
                                </div>
                            </div>

                            <div className="control-group">
                                <label>Seed (0 = random)</label>
                                <input
                                    type="number"
                                    min="0"
                                    step="1"
                                    value={lsystemParams.seed || 0}
                                    onChange={(e) => onLSystemChange({ seed: parseInt(e.target.value) || null })}
                                />
                            </div>
                        </div>
                    </>
                )}

                {/* IFS Controls */}
                {mode === 'ifs' && (
                    <div className="section-divider" style={{ borderTop: 'none', paddingTop: 0 }}>
                        <h3>Barnsley Fern</h3>
                        <div className="control-group">
                            <label>Width</label>
                            <div className="input-with-unit">
                                <input
                                    type="number"
                                    min="200"
                                    max="1600"
                                    step="50"
                                    value={ifsParams.width}
                                    onChange={(e) => onIFSChange({ width: parseInt(e.target.value) })}
                                />
                                <span className="unit">px</span>
                            </div>
                        </div>
                        <div className="control-group">
                            <label>Height</label>
                            <div className="input-with-unit">
                                <input
                                    type="number"
                                    min="200"
                                    max="1200"
                                    step="50"
                                    value={ifsParams.height}
                                    onChange={(e) => onIFSChange({ height: parseInt(e.target.value) })}
                                />
                                <span className="unit">px</span>
                            </div>
                        </div>
                        <div className="control-group">
                            <label>Iterations</label>
                            <div className="numeric-stepper">
                                <button
                                    className="stepper-btn left"
                                    onClick={() => onIFSChange({ iterations: Math.max(1000, ifsParams.iterations - 10000) })}
                                >−</button>
                                <input
                                    type="number"
                                    value={ifsParams.iterations}
                                    min="1000"
                                    max="500000"
                                    onChange={(e) => onIFSChange({ iterations: parseInt(e.target.value) })}
                                />
                                <button
                                    className="stepper-btn right"
                                    onClick={() => onIFSChange({ iterations: Math.min(500000, ifsParams.iterations + 10000) })}
                                >+</button>
                            </div>
                        </div>
                    </div>
                )}

                {/* React Component Settings */}
                <div className="section-divider">
                    <h3>React Component</h3>
                    <div className="range-wrapper">
                        <div className="range-header">
                            <label>Animation Speed</label>
                            <span className="range-value">{reactParams.animation_speed}</span>
                        </div>
                        <input type="range" min="0.1" max="2" step="0.1"
                            value={reactParams.animation_speed}
                            onChange={(e) => onReactChange({ animation_speed: parseFloat(e.target.value) })} />
                    </div>
                    <div className="range-wrapper">
                        <div className="range-header">
                            <label>Parallax Strength</label>
                            <span className="range-value">{reactParams.parallax_strength}</span>
                        </div>
                        <input type="range" min="0" max="1" step="0.05"
                            value={reactParams.parallax_strength}
                            onChange={(e) => onReactChange({ parallax_strength: parseFloat(e.target.value) })} />
                    </div>
                    <div className="range-wrapper">
                        <div className="range-header">
                            <label>Glow Layers</label>
                            <span className="range-value">{reactParams.glow_layers}</span>
                        </div>
                        <input type="range" min="1" max="3" step="1"
                            value={reactParams.glow_layers}
                            onChange={(e) => onReactChange({ glow_layers: parseInt(e.target.value) })} />
                    </div>
                    <div className="range-wrapper">
                        <div className="range-header">
                            <label>Max Points</label>
                            <span className="range-value">{reactParams.max_points}</span>
                        </div>
                        <input type="range" min="1000" max="8000" step="500"
                            value={reactParams.max_points}
                            onChange={(e) => onReactChange({ max_points: parseInt(e.target.value) })} />
                    </div>
                    <div className="color-swatch-row">
                        <label>Background</label>
                        <div className="swatch-group">
                            <input
                                type="color"
                                value={reactParams.bg_color}
                                onChange={(e) => onReactChange({ bg_color: e.target.value })}
                                style={{ width: 24, height: 24, cursor: 'pointer', border: '1px solid rgba(230,25,25,0.5)', padding: 0 }}
                            />
                        </div>
                    </div>
                    <div className="control-group">
                        <label>Color Palette (comma-separated)</label>
                        <input type="text" value={reactParams.colors.join(',')}
                            onChange={(e) => {
                                const colors = e.target.value.split(',').map(c => c.trim()).filter(Boolean);
                                if (colors.length > 0) onReactChange({ colors });
                            }} />
                    </div>
                </div>
            </div>

            {/* Action Area */}
            <div className="sidebar-actions">
                <button className="btn-execute" onClick={() => {
                    // Trigger generate on the active page via a custom event
                    window.dispatchEvent(new CustomEvent('execute-build'));
                }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 18 }}>play_arrow</span>
                    <span>Execute Build</span>
                </button>
                <div className="action-row">
                    <button className="btn-secondary">
                        <span className="material-symbols-outlined" style={{ fontSize: 16 }}>save</span>
                        Preset
                    </button>
                    <button className="btn-secondary">
                        <span className="material-symbols-outlined" style={{ fontSize: 16 }}>restart_alt</span>
                        Reset
                    </button>
                </div>
            </div>
        </aside>
    );
}
