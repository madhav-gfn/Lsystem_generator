import { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { saveAs } from 'file-saver';

export default function CodeViewer({ code, filename = 'GeneratedBackground.jsx' }) {
    const [copied, setCopied] = useState(false);

    const handleDownload = () => {
        const blob = new Blob([code], { type: 'text/plain;charset=utf-8' });
        saveAs(blob, filename);
    };

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            // Fallback
            const textarea = document.createElement('textarea');
            textarea.value = code;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    return (
        <div className="code-viewer">
            {/* Corner brackets */}
            <div className="corner tl"></div>
            <div className="corner tr"></div>
            <div className="corner bl"></div>
            <div className="corner br"></div>

            <div className="code-viewer-header">
                <span>{filename}</span>
                <div className="code-actions">
                    <button
                        className="btn btn-secondary btn-sm"
                        onClick={handleCopy}
                        style={{ minWidth: 80 }}
                    >
                        <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
                            {copied ? 'check' : 'content_copy'}
                        </span>
                        {copied ? 'Copied!' : 'Copy'}
                    </button>
                    <button className="btn btn-secondary btn-sm" onClick={handleDownload}>
                        <span className="material-symbols-outlined" style={{ fontSize: 14 }}>download</span>
                        Download
                    </button>
                </div>
            </div>
            <SyntaxHighlighter
                language="jsx"
                style={vscDarkPlus}
                customStyle={{
                    margin: 0,
                    background: 'transparent',
                    padding: '16px',
                    fontSize: '0.78rem',
                }}
                showLineNumbers
            >
                {code}
            </SyntaxHighlighter>
        </div>
    );
}
