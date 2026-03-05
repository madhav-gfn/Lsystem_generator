import { useState, useEffect } from 'react';
import { saveAs } from 'file-saver';
import {
    generateSVG,
    generatePNG,
    generateGeometry,
    generateReactComponent,
    generateBatch,
} from '../services/api';
import CanvasPreview from './CanvasPreview';
import CodeViewer from './CodeViewer';

const MIN_LOADING_MS = 5000;

export default function LSystemPage({ lsystemParams, reactParams }) {
    const [activeTab, setActiveTab] = useState('svg');
    const [loading, setLoading] = useState(false);
    const [showOverlay, setShowOverlay] = useState(false);
    const [loadingLabel, setLoadingLabel] = useState('');
    const [error, setError] = useState(null);

    // Tab data
    const [svgData, setSvgData] = useState(null);
    const [pngUrl, setPngUrl] = useState(null);
    const [pngBlob, setPngBlob] = useState(null);
    const [geometry, setGeometry] = useState(null);
    const [reactSource, setReactSource] = useState(null);
    const [batchCount, setBatchCount] = useState(1);

    const withMinDelay = async (label, fn) => {
        setLoading(true);
        setShowOverlay(true);
        setLoadingLabel(label);
        setError(null);
        const start = Date.now();
        try {
            await fn();
        } catch (e) {
            setError(e.message);
        }
        const elapsed = Date.now() - start;
        const remaining = MIN_LOADING_MS - elapsed;
        if (remaining > 0) await new Promise(r => setTimeout(r, remaining));
        setLoading(false);
        setShowOverlay(false);
    };

    const buildPayload = () => ({
        ...lsystemParams,
        seed: lsystemParams.seed || null,
    });

    const handleGenerateSVG = () => withMinDelay('Generating SVG', async () => {
        const data = await generateSVG(buildPayload());
        setSvgData(data);
    });

    const handleGeneratePNG = () => withMinDelay('Rendering PNG', async () => {
        const blob = await generatePNG(buildPayload());
        setPngBlob(blob);
        setPngUrl(URL.createObjectURL(blob));
    });

    const handleGeneratePreview = () => withMinDelay('Building Preview', async () => {
        const geo = await generateGeometry({ lsystem: buildPayload(), react: reactParams });
        setGeometry(geo);
    });

    const handleGenerateReact = () => withMinDelay('Compiling React Component', async () => {
        const data = await generateReactComponent({ lsystem: buildPayload(), react: reactParams });
        setReactSource(data);
    });

    const handleBatch = () => withMinDelay(`Generating ${batchCount} Variations`, async () => {
        const blob = await generateBatch({ lsystem: buildPayload(), count: batchCount });
        saveAs(blob, 'lsystem_batch.zip');
    });

    // Listen for "Execute Build" from sidebar
    useEffect(() => {
        const handler = () => {
            if (activeTab === 'svg') handleGenerateSVG();
            else if (activeTab === 'png') handleGeneratePNG();
            else if (activeTab === 'preview') handleGeneratePreview();
            else if (activeTab === 'react') handleGenerateReact();
            else if (activeTab === 'batch') handleBatch();
        };
        window.addEventListener('execute-build', handler);
        return () => window.removeEventListener('execute-build', handler);
    });

    const tabs = [
        { id: 'svg', label: 'SVG Preview', icon: 'image' },
        { id: 'png', label: 'PNG', icon: 'photo_camera' },
        { id: 'preview', label: 'Live Preview', icon: 'play_circle' },
        { id: 'react', label: 'React', icon: 'code' },
        { id: 'batch', label: 'Batch', icon: 'collections' },
    ];

    return (
        <div className="main-content">
            <div className="main-header">
                <h2>
                    <span className="material-symbols-outlined" style={{ fontSize: 18, marginRight: 8, verticalAlign: 'middle' }}>data_object</span>
                    L-System Generator
                </h2>
                <div className="tabs">
                    {tabs.map(t => (
                        <button
                            key={t.id}
                            className={`tab-btn ${activeTab === t.id ? 'active' : ''}`}
                            onClick={() => setActiveTab(t.id)}
                        >
                            {t.label}
                        </button>
                    ))}
                </div>
            </div>

            <div className="tab-content">
                {error && <div className="error-message">⚠ {error}</div>}

                {/* SVG Tab */}
                {activeTab === 'svg' && (
                    <div>
                        <button className="btn btn-primary" onClick={handleGenerateSVG} disabled={loading}>
                            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
                                {loading ? 'hourglass_empty' : 'eco'}
                            </span>
                            <span>{loading ? 'Generating...' : 'Generate SVG'}</span>
                        </button>
                        {svgData && (
                            <div className="section-gap" style={{ marginTop: 20 }}>
                                <div className="info-badge" style={{ marginBottom: 12 }}>
                                    STR: {svgData.string_length.toLocaleString()} · SEG: {svgData.segment_count.toLocaleString()}
                                </div>
                                <div className="svg-preview">
                                    <div className="corner tl"></div>
                                    <div className="corner tr"></div>
                                    <div className="corner bl"></div>
                                    <div className="corner br"></div>
                                    <div dangerouslySetInnerHTML={{ __html: svgData.svg }} />
                                </div>
                                <div className="actions-bar">
                                    <button className="btn btn-secondary btn-sm" onClick={() => {
                                        const blob = new Blob([svgData.svg], { type: 'image/svg+xml' });
                                        saveAs(blob, 'lsystem.svg');
                                    }}>
                                        <span className="material-symbols-outlined" style={{ fontSize: 14 }}>download</span>
                                        Download SVG
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* PNG Tab */}
                {activeTab === 'png' && (
                    <div>
                        <button className="btn btn-primary" onClick={handleGeneratePNG} disabled={loading}>
                            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
                                {loading ? 'hourglass_empty' : 'photo_camera'}
                            </span>
                            <span>{loading ? 'Rendering...' : 'Generate PNG'}</span>
                        </button>
                        {pngUrl && (
                            <div className="section-gap" style={{ marginTop: 20 }}>
                                <div className="image-preview">
                                    <div className="corner tl"></div>
                                    <div className="corner tr"></div>
                                    <div className="corner bl"></div>
                                    <div className="corner br"></div>
                                    <img src={pngUrl} alt="L-System Render" />
                                </div>
                                <div className="actions-bar">
                                    <button className="btn btn-secondary btn-sm" onClick={() => saveAs(pngBlob, 'lsystem.png')}>
                                        <span className="material-symbols-outlined" style={{ fontSize: 14 }}>download</span>
                                        Download PNG
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Live Preview Tab */}
                {activeTab === 'preview' && (
                    <div>
                        <button className="btn btn-primary" onClick={handleGeneratePreview} disabled={loading} style={{ marginBottom: 16 }}>
                            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
                                {loading ? 'hourglass_empty' : 'play_arrow'}
                            </span>
                            <span>{loading ? 'Generating...' : 'Generate Preview'}</span>
                        </button>
                        <CanvasPreview geometry={geometry} />
                    </div>
                )}

                {/* React Component Tab */}
                {activeTab === 'react' && (
                    <div>
                        <button className="btn btn-primary" onClick={handleGenerateReact} disabled={loading}>
                            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
                                {loading ? 'hourglass_empty' : 'code'}
                            </span>
                            <span>{loading ? 'Generating...' : 'Generate React Component'}</span>
                        </button>
                        {reactSource && (
                            <div className="section-gap" style={{ marginTop: 20 }}>
                                <div className="info-badge" style={{ marginBottom: 12 }}>
                                    {reactSource.label}
                                </div>
                                <CodeViewer code={reactSource.jsx} />
                                <p style={{ marginTop: 12, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                                    Add to your React project. Run: <code style={{ color: 'var(--primary)' }}>npm install three</code>.
                                    Import and drop <code style={{ color: 'var(--primary)' }}>{'<GeneratedBackground />'}</code> anywhere.
                                </p>
                            </div>
                        )}
                    </div>
                )}

                {/* Batch Tab */}
                {activeTab === 'batch' && (
                    <div>
                        <div className="control-group" style={{ maxWidth: 300, marginBottom: 16 }}>
                            <label>Number of variations</label>
                            <div className="numeric-stepper">
                                <button className="stepper-btn left" onClick={() => setBatchCount(Math.max(1, batchCount - 1))}>−</button>
                                <input type="number" min="1" max="200" value={batchCount}
                                    onChange={(e) => setBatchCount(Math.max(1, parseInt(e.target.value) || 1))} />
                                <button className="stepper-btn right" onClick={() => setBatchCount(Math.min(200, batchCount + 1))}>+</button>
                            </div>
                        </div>
                        <button className="btn btn-primary" onClick={handleBatch} disabled={loading}>
                            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
                                {loading ? 'hourglass_empty' : 'collections'}
                            </span>
                            <span>{loading ? 'Generating...' : `Generate ${batchCount} Variations`}</span>
                        </button>
                        <p style={{ marginTop: 12, fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                            Downloads as a ZIP archive containing {batchCount} PNG images.
                        </p>
                    </div>
                )}
            </div>

            {showOverlay && (
                <div className="loading-overlay">
                    <div className="loading-overlay-content">
                        <div className="loading-fractal">
                            <div className="loading-fractal-ring" />
                            <div className="loading-fractal-ring ring-2" />
                            <div className="loading-fractal-ring ring-3" />
                            <div className="loading-fractal-core" />
                        </div>
                        <h3 className="loading-title">{loadingLabel}</h3>
                        <p className="loading-subtitle">Processing L-System pipeline...</p>
                        <div className="loading-progress-track">
                            <div className="loading-progress-bar" />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
