import { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import LSystemPage from './components/LSystemPage';
import IFSPage from './components/IFSPage';
import { fetchPresets } from './services/api';
import './index.css';

const DEFAULT_LSYSTEM = {
  preset_name: 'Fractal Plant (Deterministic)',
  axiom: 'X',
  rules: {
    X: [['F-[[X]+X]+F[+FX]-X', 1.0]],
    F: [['FF', 1.0]],
  },
  angle: 25.0,
  step: 5.0,
  iterations: 6,
  seed: null,
  stroke_width: 1.0,
  stroke_color: '#e61919',
  bg_color: '#211111',
  out_width: 1200,
  out_height: 1200,
};

const DEFAULT_REACT = {
  colors: ['#e61919', '#ff4444', '#ff6b6b'],
  animation_speed: 1.0,
  parallax_strength: 0.4,
  glow_layers: 2,
  bg_color: '#150a0a',
  max_points: 4000,
  particle_size: 0.008,
};

const DEFAULT_IFS = {
  width: 800,
  height: 600,
  iterations: 100000,
  color: [230, 25, 25],
};

export default function App() {
  const [mode, setMode] = useState('lsystem');
  const [presets, setPresets] = useState({});
  const [presetNames, setPresetNames] = useState([]);
  const [lsystemParams, setLSystemParams] = useState(DEFAULT_LSYSTEM);
  const [reactParams, setReactParams] = useState(DEFAULT_REACT);
  const [ifsParams, setIFSParams] = useState(DEFAULT_IFS);

  // Load presets on mount
  useEffect(() => {
    fetchPresets()
      .then(data => {
        setPresets(data);
        const names = Object.keys(data);
        setPresetNames(names);
        // Apply first preset
        if (names.length > 0) {
          const first = data[names[0]];
          setLSystemParams(prev => ({
            ...prev,
            preset_name: names[0],
            axiom: first.axiom,
            rules: first.rules,
            angle: first.angle,
            step: first.step,
            iterations: first.iterations,
          }));
        }
      })
      .catch(err => console.error('Failed to load presets:', err));
  }, []);

  const handleLSystemChange = useCallback((updates) => {
    setLSystemParams(prev => ({ ...prev, ...updates }));
  }, []);

  const handleReactChange = useCallback((updates) => {
    setReactParams(prev => ({ ...prev, ...updates }));
  }, []);

  const handleIFSChange = useCallback((updates) => {
    setIFSParams(prev => ({ ...prev, ...updates }));
  }, []);

  return (
    <div className="app-layout">
      {/* Top Header */}
      <header className="app-header">
        <div className="app-header-logo">
          <div className="app-header-icon">
            <span className="material-symbols-outlined">account_tree</span>
          </div>
          <h1>L-Systems & IFS Studio</h1>
          <span className="app-header-version">V 2.0.4</span>
        </div>
        <nav className="app-header-nav">
          <a href="#">
            <span className="material-symbols-outlined">grid_view</span> Gallery
          </a>
          <a href="#">
            <span className="material-symbols-outlined">menu_book</span> Docs
          </a>
          <a href="#">
            <span className="material-symbols-outlined">output</span> Export
          </a>
          <a href="#">
            <span className="material-symbols-outlined">settings</span> Config
          </a>
        </nav>
      </header>

      {/* Main Workspace */}
      <div className="app-workspace">
        {/* Left: Canvas / Content */}
        {mode === 'lsystem' ? (
          <LSystemPage lsystemParams={lsystemParams} reactParams={reactParams} />
        ) : (
          <IFSPage ifsParams={ifsParams} reactParams={reactParams} />
        )}

        {/* Right: Sidebar */}
        <Sidebar
          mode={mode}
          onModeChange={setMode}
          presets={presets}
          presetNames={presetNames}
          lsystemParams={lsystemParams}
          onLSystemChange={handleLSystemChange}
          reactParams={reactParams}
          onReactChange={handleReactChange}
          ifsParams={ifsParams}
          onIFSChange={handleIFSChange}
        />
      </div>
    </div>
  );
}
