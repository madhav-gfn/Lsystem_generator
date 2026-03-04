const API_BASE = 'http://localhost:8000';

export async function fetchPresets() {
  const res = await fetch(`${API_BASE}/api/presets`);
  if (!res.ok) throw new Error('Failed to fetch presets');
  return res.json();
}

export async function generateSVG(params) {
  const res = await fetch(`${API_BASE}/api/lsystem/svg`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'SVG generation failed');
  }
  return res.json();
}

export async function generatePNG(params) {
  const res = await fetch(`${API_BASE}/api/lsystem/png`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error('PNG generation failed');
  return res.blob();
}

export async function generateGeometry(params) {
  const res = await fetch(`${API_BASE}/api/lsystem/geometry`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Geometry generation failed');
  }
  return res.json();
}

export async function generateReactComponent(params) {
  const res = await fetch(`${API_BASE}/api/lsystem/react`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'React component generation failed');
  }
  return res.json();
}

export async function generateBatch(params) {
  const res = await fetch(`${API_BASE}/api/lsystem/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error('Batch generation failed');
  return res.blob();
}

export async function generateIFSPng(params) {
  const res = await fetch(`${API_BASE}/api/ifs/png`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error('IFS PNG generation failed');
  return res.blob();
}

export async function generateIFSGeometry(params) {
  const res = await fetch(`${API_BASE}/api/ifs/geometry`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error('IFS geometry generation failed');
  return res.json();
}

export async function generateIFSReact(params) {
  const res = await fetch(`${API_BASE}/api/ifs/react`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error('IFS React generation failed');
  return res.json();
}
