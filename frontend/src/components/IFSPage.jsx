import { useState, useEffect } from 'react';
import { saveAs } from 'file-saver';
import {
    generateIFSPng,
    generateIFSGeometry,
    generateIFSReact,
} from '../services/api';
import CanvasPreview from './CanvasPreview';
import CodeViewer from './CodeViewer';

export default function IFSPage({ ifsParams, reactParams }) {
    const [activeTab, setActiveTab] = useState('image');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const [pngUrl, setPngUrl] = useState(null);
    const [pngBlob, setPngBlob] = useState(null);
    const [geometry, setGeometry] = useState(null);
    const [reactSource, setReactSource] = useState(null);

    const handleGeneratePNG = async () => {
        setLoading(true);
        setError(null);
        try {
            const blob = await generateIFSPng(ifsParams);
            setPngBlob(blob);
            setPngUrl(URL.createObjectURL(blob));
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    const handleGeneratePreview = async () => {
        setLoading(true);
        setError(null);
        try {
            const geo = await generateIFSGeometry({ iterations: ifsParams.iterations, react: reactParams });
            setGeometry(geo);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    const handleGenerateReact = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await generateIFSReact({ iterations: ifsParams.iterations, react: reactParams });
            setReactSource(data);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

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

            {loading && (
                <div className="status-bar">
                    <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
                    Processing build...
                </div>
            )}
        </div>
    );
}
