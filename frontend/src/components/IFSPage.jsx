import { useState, useEffect } from 'react';
import { saveAs } from 'file-saver';
import {
    generateIFSPng,
    generateIFSGeometry,
    generateIFSReact,
} from '../services/api';
import CanvasPreview from './CanvasPreview';
import CodeViewer from './CodeViewer';

const MIN_LOADING_MS = 5000;

export default function IFSPage({ ifsParams, reactParams }) {
    const [activeTab, setActiveTab] = useState('image');
    const [loading, setLoading] = useState(false);
    const [showOverlay, setShowOverlay] = useState(false);
    const [loadingLabel, setLoadingLabel] = useState('');
    const [error, setError] = useState(null);

    const [pngUrl, setPngUrl] = useState(null);
    const [pngBlob, setPngBlob] = useState(null);
    const [geometry, setGeometry] = useState(null);
    const [reactSource, setReactSource] = useState(null);

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

    const handleGeneratePNG = () => withMinDelay('Rendering Fern', async () => {
        const blob = await generateIFSPng(ifsParams);
        setPngBlob(blob);
        setPngUrl(URL.createObjectURL(blob));
    });

    const handleGeneratePreview = () => withMinDelay('Building Preview', async () => {
        const geo = await generateIFSGeometry({ iterations: ifsParams.iterations, react: reactParams });
        setGeometry(geo);
    });

    const handleGenerateReact = () => withMinDelay('Compiling React Component', async () => {
        const data = await generateIFSReact({ iterations: ifsParams.iterations, react: reactParams });
        setReactSource(data);
    });

    // Listen for "Execute Build" from sidebar
    useEffect(() => {
        const handler = () => {
            if (activeTab === 'image') handleGeneratePNG();
            else if (activeTab === 'preview') handleGeneratePreview();
            else if (activeTab === 'react') handleGenerateReact();
        };
        window.addEventListener('execute-build', handler);
        return () => window.removeEventListener('execute-build', handler);
    });

    const tabs = [
        { id: 'image', label: 'Fern Image', icon: 'eco' },
        { id: 'preview', label: 'Live Preview', icon: 'play_circle' },
        { id: 'react', label: 'React', icon: 'code' },
    ];

    return (
        <div className="main-content">
            <div className="main-header">
                <h2>
                    <span className="material-symbols-outlined" style={{ fontSize: 18, marginRight: 8, verticalAlign: 'middle' }}>grid_on</span>
                    Barnsley Fern (IFS)
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

                {activeTab === 'image' && (
                    <div>
                        <button className="btn btn-primary" onClick={handleGeneratePNG} disabled={loading}>
                            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
                                {loading ? 'hourglass_empty' : 'eco'}
                            </span>
                            <span>{loading ? 'Generating...' : 'Generate Fern'}</span>
                        </button>
                        {pngUrl && (
                            <div className="section-gap" style={{ marginTop: 20 }}>
                                <div className="image-preview">
                                    <div className="corner tl"></div>
                                    <div className="corner tr"></div>
                                    <div className="corner bl"></div>
                                    <div className="corner br"></div>
                                    <img src={pngUrl} alt="Barnsley Fern" />
                                </div>
                                <div className="actions-bar">
                                    <button className="btn btn-secondary btn-sm" onClick={() => saveAs(pngBlob, 'barnsley_fern.png')}>
                                        <span className="material-symbols-outlined" style={{ fontSize: 14 }}>download</span>
                                        Download PNG
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                )}

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
                        <p className="loading-subtitle">Processing IFS pipeline...</p>
                        <div className="loading-progress-track">
                            <div className="loading-progress-bar" />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
