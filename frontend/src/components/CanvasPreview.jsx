import { useRef, useEffect, useCallback } from 'react';

export default function CanvasPreview({ geometry }) {
    const containerRef = useRef(null);
    const canvasRef = useRef(null);
    const animationRef = useRef(null);
    const drawRef = useRef(null);
    const stateRef = useRef({ angle: 0, parallaxX: 0, parallaxY: 0, mx: 0, my: 0 });

    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas || !geometry) return;
        const ctx = canvas.getContext('2d');
        const P = geometry;
        const s = stateRef.current;
        const W = canvas.width;
        const H = canvas.height;

        const elapsed = performance.now() / 1000;

        // Clear
        ctx.fillStyle = P.backgroundColor || '#150a0a';
        ctx.fillRect(0, 0, W, H);

        ctx.save();
        ctx.translate(W / 2, H / 2);

        // Parallax
        s.parallaxX += (s.mx * 30 * (P.parallaxStrength || 0.4) - s.parallaxX) * 0.05;
        s.parallaxY += (s.my * 30 * (P.parallaxStrength || 0.4) - s.parallaxY) * 0.05;
        ctx.translate(s.parallaxX, s.parallaxY);

        // Breathe
        const breathe = 1.0 + Math.sin(elapsed * (P.animationSpeed || 1.0) * 0.4) * 0.04;
        ctx.scale(breathe, breathe);

        // Rotation
        s.angle += (P.animationSpeed || 1.0) * 0.002;
        ctx.rotate(s.angle);

        const scale = Math.min(W, H) * 0.42;
        ctx.scale(scale, -scale);
        ctx.lineCap = 'round';

        const colors = P.colors || ['#e61919'];
        const colorCount = colors.length;

        if (P.type === 'lsystem') {
            const lw = (P.lineWidth || 1) / scale;
            const ps = (P.particleSize || 0.008) * 2;
            const edges = P.edges || [];
            const verts = P.vertices || [];

            // Glow
            ctx.globalAlpha = 0.15;
            ctx.lineWidth = lw * 4;
            for (let ci = 0; ci < colorCount; ci++) {
                ctx.strokeStyle = colors[ci];
                ctx.beginPath();
                for (let i = ci; i < edges.length; i += colorCount) {
                    const e = edges[i];
                    ctx.moveTo(e[0], e[1]);
                    ctx.lineTo(e[2], e[3]);
                }
                ctx.stroke();
            }

            // Main
            ctx.globalAlpha = 1.0;
            ctx.lineWidth = lw;
            for (let ci = 0; ci < colorCount; ci++) {
                ctx.strokeStyle = colors[ci];
                ctx.beginPath();
                for (let i = ci; i < edges.length; i += colorCount) {
                    const e = edges[i];
                    ctx.moveTo(e[0], e[1]);
                    ctx.lineTo(e[2], e[3]);
                }
                ctx.stroke();

                // Vertex dots
                ctx.fillStyle = colors[ci];
                ctx.beginPath();
                for (let i = ci; i < verts.length; i += colorCount) {
                    const v = verts[i];
                    ctx.moveTo(v[0] + ps, v[1]);
                    ctx.arc(v[0], v[1], ps, 0, Math.PI * 2);
                }
                ctx.fill();
            }
        } else {
            const ps = (P.particleSize || 0.012) * 1.5;
            const points = P.points || [];

            // Glow
            ctx.globalAlpha = 0.12;
            for (let ci = 0; ci < colorCount; ci++) {
                ctx.fillStyle = colors[ci];
                ctx.beginPath();
                for (let i = ci; i < points.length; i += colorCount) {
                    const p = points[i];
                    ctx.moveTo(p[0] + ps * 2, p[1]);
                    ctx.arc(p[0], p[1], ps * 2, 0, Math.PI * 2);
                }
                ctx.fill();
            }

            // Main
            ctx.globalAlpha = 0.85;
            for (let ci = 0; ci < colorCount; ci++) {
                ctx.fillStyle = colors[ci];
                ctx.beginPath();
                for (let i = ci; i < points.length; i += colorCount) {
                    const p = points[i];
                    ctx.moveTo(p[0] + ps, p[1]);
                    ctx.arc(p[0], p[1], ps, 0, Math.PI * 2);
                }
                ctx.fill();
            }
            ctx.globalAlpha = 1.0;
        }

        ctx.restore();
        animationRef.current = requestAnimationFrame(() => drawRef.current && drawRef.current());
    }, [geometry]);

    useEffect(() => {
        drawRef.current = draw;
    }, [draw]);

    useEffect(() => {
        const container = containerRef.current;
        const canvas = canvasRef.current;
        if (!container || !canvas) return;

        const resize = () => {
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
        };
        resize();

        const onMouseMove = (e) => {
            const rect = canvas.getBoundingClientRect();
            stateRef.current.mx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            stateRef.current.my = ((e.clientY - rect.top) / rect.height) * 2 - 1;
        };

        window.addEventListener('resize', resize);
        canvas.addEventListener('mousemove', onMouseMove);

        if (geometry) {
            stateRef.current.angle = 0;
            animationRef.current = requestAnimationFrame(draw);
        }

        return () => {
            cancelAnimationFrame(animationRef.current);
            window.removeEventListener('resize', resize);
            canvas.removeEventListener('mousemove', onMouseMove);
        };
    }, [geometry, draw]);

    // Compute stats
    const stats = geometry ? {
        points: geometry.type === 'lsystem'
            ? (geometry.vertices?.length || 0)
            : (geometry.points?.length || 0),
        edges: geometry.edges?.length || 0,
    } : null;

    return (
        <div className="canvas-container" ref={containerRef}>
            {/* Corner Brackets */}
            <div className="corner tl"></div>
            <div className="corner tr"></div>
            <div className="corner bl"></div>
            <div className="corner br"></div>

            <canvas ref={canvasRef} />

            {/* LIVE indicator */}
            {geometry && (
                <div className="live-indicator">
                    <div className="dot"></div>
                    LIVE
                </div>
            )}

            {/* HUD Stats */}
            {stats && (
                <div className="canvas-hud">
                    <span>VERTICES: {stats.points.toLocaleString()}</span>
                    {stats.edges > 0 && <span>EDGES: {stats.edges.toLocaleString()}</span>}
                    <span>SCALE: 1.0x</span>
                </div>
            )}

            {/* Empty state */}
            {!geometry && (
                <div className="preview-empty">
                    <div>
                        <div className="icon">
                            <span className="material-symbols-outlined" style={{ fontSize: 48, color: 'var(--primary)', opacity: 0.4 }}>
                                science
                            </span>
                        </div>
                        <p>Execute build to generate live preview</p>
                    </div>
                </div>
            )}
        </div>
    );
}
