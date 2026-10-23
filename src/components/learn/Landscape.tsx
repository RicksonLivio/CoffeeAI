import { useEffect, useMemo, useRef, useState } from 'react';
import { fx, useLang } from '../../i18n';
import { basinOf, descend, DOMAIN, GLOBAL, LOCAL, surface, type DescentPoint } from '../../lib/landscape';

export interface Runner {
  eta: number;
  color: string;
  label: string;
}

interface Props {
  runners: Runner[];
  start: [number, number];
  onStart: (p: [number, number]) => void;
  steps: number;
}

const [LO, HI] = DOMAIN;
const ZMIN = 0.05;
const ZMAX = 1.95;

/** Height → colour: gold in the valleys, cyan on the slopes, navy on the hills (theme palette). */
function heightColor(z: number, alpha = 1): string {
  const t = Math.max(0, Math.min(1, (z - ZMIN) / (ZMAX - ZMIN)));
  const stops: [number, [number, number, number]][] = [
    [0, [255, 210, 94]], [0.18, [245, 181, 42]], [0.38, [41, 211, 232]], [0.62, [30, 91, 216]], [1, [13, 27, 42]],
  ];
  let i = 0;
  while (i < stops.length - 2 && t > stops[i + 1][0]) i++;
  const [t0, c0] = stops[i];
  const [t1, c1] = stops[i + 1];
  const k = (t - t0) / (t1 - t0);
  const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

function useCanvas(draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void, deps: unknown[]) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    draw(ctx, w, h);
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  return ref;
}

/** Marching squares: line segments of the level set L = level on an n×n grid. */
function isoSegments(grid: number[][], level: number, n: number): [number, number, number, number][] {
  const seg: [number, number, number, number][] = [];
  const lerp = (a: number, b: number) => (level - a) / (b - a);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = [grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]];
      const pts: [number, number][] = [];
      const edges: [number, number, number, number][] = [[0, 1, 0, 0], [1, 2, 1, 0], [2, 3, 0, 1], [3, 0, 0, 0]];
      // corners: 0 (i,j) 1 (i+1,j) 2 (i+1,j+1) 3 (i,j+1)
      const corner = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
      for (const [a, b] of edges) {
        if ((v[a] < level) !== (v[b] < level)) {
          const t = lerp(v[a], v[b]);
          pts.push([corner[a][0] + (corner[b][0] - corner[a][0]) * t, corner[a][1] + (corner[b][1] - corner[a][1]) * t]);
        }
      }
      if (pts.length >= 2) seg.push([pts[0][0], pts[0][1], pts[1][0], pts[1][1]]);
      if (pts.length === 4) seg.push([pts[2][0], pts[2][1], pts[3][0], pts[3][1]]);
    }
  }
  return seg;
}

export function useDescents(runners: Runner[], start: [number, number], steps: number): DescentPoint[][] {
  return useMemo(() => runners.map((r) => descend(start[0], start[1], r.eta, steps)), [runners, start, steps]);
}

/** Animated gradient descent over a toy error surface: contour map (click to choose the start) and 3D view. */
export default function Landscape({ runners, start, onStart, steps }: Props) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const paths = useDescents(runners, start, steps);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [az, setAz] = useState(-0.75);
  // elevation of the camera: 0.12 ≈ seen from the side, 1.45 ≈ seen from above
  const [el, setEl] = useState(0.8);
  const [spin, setSpin] = useState(false);
  const raf = useRef<number | null>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => { setT(0); }, [start, runners]);

  useEffect(() => {
    if (!playing && !spin) return;
    let last = performance.now();
    let acc = 0;
    const tick = (now: number) => {
      const dt = now - last;
      last = now;
      if (spin) setAz((a) => a + dt * 0.00035);
      if (playing) {
        acc += dt;
        if (acc > 90) {
          acc = 0;
          setT((x) => {
            if (x >= steps) { setPlaying(false); return x; }
            return x + 1;
          });
        }
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => { if (raf.current !== null) cancelAnimationFrame(raf.current); };
  }, [playing, spin, steps]);

  // grid of heights, shared by both views
  const N = 60;
  const grid = useMemo(() => Array.from({ length: N + 1 }, (_, i) => Array.from({ length: N + 1 }, (_, j) =>
    surface(LO + ((HI - LO) * i) / N, LO + ((HI - LO) * j) / N))), []);
  const levels = useMemo(() => Array.from({ length: 16 }, (_, k) => 0.15 + k * 0.1), []);
  const iso = useMemo(() => levels.map((lv) => isoSegments(grid, lv, N)), [grid, levels]);

  const contourRef = useCanvas((ctx, w, h) => {
    const size = Math.min(w, h);
    const ox = (w - size) / 2;
    const sx = (a: number) => ox + ((a - LO) / (HI - LO)) * size;
    const sy = (b: number) => size - ((b - LO) / (HI - LO)) * size;
    const cell = size / N;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      ctx.fillStyle = heightColor((grid[i][j] + grid[i + 1][j + 1]) / 2, 0.55);
      ctx.fillRect(ox + i * cell, size - (j + 1) * cell, cell + 0.6, cell + 0.6);
    }
    ctx.lineWidth = 1;
    iso.forEach((segs, k) => {
      ctx.strokeStyle = k % 4 === 0 ? 'rgba(230,236,244,0.55)' : 'rgba(10,20,36,0.55)';
      ctx.beginPath();
      for (const [i0, j0, i1, j1] of segs) {
        ctx.moveTo(ox + i0 * cell, size - j0 * cell);
        ctx.lineTo(ox + i1 * cell, size - j1 * cell);
      }
      ctx.stroke();
    });
    const mark = (a: number, b: number, txt: string) => {
      ctx.fillStyle = '#05080f';
      ctx.beginPath(); ctx.arc(sx(a), sy(b), 5, 0, 2 * Math.PI); ctx.fill();
      ctx.font = '600 12px "JetBrains Mono", monospace';
      ctx.fillText(txt, sx(a) + 8, sy(b) - 8);
    };
    mark(GLOBAL.a, GLOBAL.b, pt ? 'mínimo global' : 'global minimum');
    mark(LOCAL.a, LOCAL.b, pt ? 'mínimo local' : 'local minimum');
    paths.forEach((p, r) => {
      const upto = p.slice(0, Math.min(p.length, t + 1));
      ctx.strokeStyle = runners[r].color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      upto.forEach((q, k) => (k ? ctx.lineTo(sx(q.a), sy(q.b)) : ctx.moveTo(sx(q.a), sy(q.b))));
      ctx.stroke();
      const q = upto[upto.length - 1];
      ctx.fillStyle = runners[r].color;
      ctx.shadowColor = runners[r].color; ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.arc(sx(Math.max(LO, Math.min(HI, q.a))), sy(Math.max(LO, Math.min(HI, q.b))), 6, 0, 2 * Math.PI); ctx.fill();
      ctx.shadowBlur = 0;
    });
    ctx.strokeStyle = '#e6ecf4'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sx(start[0]), sy(start[1]), 8, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = '#a9b6c8'; ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText(pt ? 'peso a →' : 'weight a →', ox + size - 86, size - 6);
    ctx.save(); ctx.translate(ox + 12, 80); ctx.rotate(-Math.PI / 2); ctx.fillText(pt ? 'peso b →' : 'weight b →', 0, 0); ctx.restore();
  }, [paths, t, grid, iso, start, lang]);

  const surfaceRef = useCanvas((ctx, w, h) => {
    const scale = Math.min(w, h * 1.25) / 9.2;
    const zs = 1.7;
    const cx = w / 2;
    const cy = h * (0.62 - 0.12 * (el - 0.8));
    const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
    const proj = (a: number, b: number, z: number) => {
      const xr = a * ca - b * sa;
      const yr = a * sa + b * ca;
      // camera direction (0, cos el, sin el): d = distance toward the camera, larger = closer
      return { x: cx + xr * scale, y: cy - (z * zs * ce - yr * se) * scale * 0.9, d: yr * ce + z * zs * se };
    };
    const M = 40;
    const step = (HI - LO) / M;
    const quads: { pts: { x: number; y: number }[]; d: number; z: number }[] = [];
    for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) {
      const a0 = LO + i * step, b0 = LO + j * step;
      const c = [[a0, b0], [a0 + step, b0], [a0 + step, b0 + step], [a0, b0 + step]].map(([a, b]) => ({ a, b, z: surface(a, b) }));
      const p = c.map((q) => proj(q.a, q.b, q.z));
      quads.push({ pts: p, d: p.reduce((s, q) => s + q.d, 0) / 4, z: c.reduce((s, q) => s + q.z, 0) / 4 });
    }
    // painter's algorithm: far tiles first, close tiles on top
    quads.sort((q1, q2) => q1.d - q2.d);
    for (const q of quads) {
      ctx.fillStyle = heightColor(q.z, 0.92);
      ctx.strokeStyle = 'rgba(5,8,15,0.45)';
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      q.pts.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    paths.forEach((p, r) => {
      const upto = p.slice(0, Math.min(p.length, t + 1)).filter((q) => q.a >= LO && q.a <= HI && q.b >= LO && q.b <= HI);
      if (!upto.length) return;
      ctx.strokeStyle = runners[r].color;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      upto.forEach((q, k) => { const s = proj(q.a, q.b, q.loss + 0.03); if (k) ctx.lineTo(s.x, s.y); else ctx.moveTo(s.x, s.y); });
      ctx.stroke();
      const q = upto[upto.length - 1];
      const s = proj(q.a, q.b, q.loss + 0.05);
      ctx.fillStyle = runners[r].color; ctx.shadowColor = runners[r].color; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(s.x, s.y, 6, 0, 2 * Math.PI); ctx.fill(); ctx.shadowBlur = 0;
    });
    ctx.fillStyle = '#a9b6c8'; ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText(pt ? 'altura = erro L(a, b) · arraste: ↔ gira, ↕ inclina' : 'height = loss L(a, b) · drag: ↔ spin, ↕ tilt', 10, h - 10);
  }, [paths, t, az, el, lang]);

  function clickContour(e: React.MouseEvent<HTMLCanvasElement>) {
    const cv = e.currentTarget;
    const r = cv.getBoundingClientRect();
    const size = Math.min(r.width, r.height);
    const ox = (r.width - size) / 2;
    const a = LO + ((e.clientX - r.left - ox) / size) * (HI - LO);
    const b = LO + ((size - (e.clientY - r.top)) / size) * (HI - LO);
    if (a < LO || a > HI || b < LO || b > HI) return;
    onStart([Math.round(a * 10) / 10, Math.round(b * 10) / 10]);
    setPlaying(true);
  }

  return (
    <div>
      <div className="player">
        <button className="btn btn--gold btn--sm" onClick={() => { if (t >= steps) setT(0); setPlaying((p) => !p); }}>
          {playing ? (pt ? '❚❚ Pausar' : '❚❚ Pause') : (pt ? '▶ Descer' : '▶ Descend')}
        </button>
        <button className="btn btn--ghost btn--sm" onClick={() => { setPlaying(false); setT((x) => Math.min(steps, x + 1)); }}>{pt ? 'Um passo' : 'One step'} ▶</button>
        <button className="btn btn--ghost btn--sm" onClick={() => { setPlaying(false); setT(0); }}>{pt ? 'Recomeçar' : 'Restart'}</button>
        <button className={`btn btn--sm ${spin ? '' : 'btn--ghost'}`} onClick={() => setSpin((s) => !s)}>{pt ? 'Girar 3D' : 'Spin 3D'}</button>
        <span className="mono small">{pt ? 'passo' : 'step'} {t}/{steps} · {pt ? 'início' : 'start'} ({fx(start[0], 1, lang)}, {fx(start[1], 1, lang)})</span>
      </div>
      <div className="grid grid-2">
        <div>
          <canvas ref={contourRef} onClick={clickContour} style={{ width: '100%', height: 380, cursor: 'crosshair', display: 'block' }}
            aria-label={pt ? 'Curvas de nível do erro; clique para escolher o ponto de partida' : 'Contour map of the loss; click to choose the starting point'} />
          <p className="small muted" style={{ margin: '6px 0 0' }}>{pt ? 'Curvas de nível (vista de cima). Clique no mapa para mudar o ponto de partida.' : 'Contour lines (top view). Click the map to change the starting point.'}</p>
        </div>
        <div>
          <canvas ref={surfaceRef} style={{ width: '100%', height: 380, cursor: 'grab', display: 'block', touchAction: 'none' }}
            onPointerDown={(e) => { drag.current = { x: e.clientX, y: e.clientY }; e.currentTarget.setPointerCapture(e.pointerId); }}
            onPointerMove={(e) => {
              const d = drag.current;
              if (!d) return;
              setAz((a) => a + (e.clientX - d.x) * 0.01);
              setEl((v) => Math.max(0.12, Math.min(1.45, v + (e.clientY - d.y) * 0.008)));
              drag.current = { x: e.clientX, y: e.clientY };
            }}
            onPointerUp={() => { drag.current = null; }}
            aria-label={pt ? 'Superfície 3D do erro' : '3D loss surface'} />
          <div className="range-row" style={{ marginTop: 6 }}>
            <span className="label">{pt ? 'Girar' : 'Spin'}</span>
            <input type="range" min={-3.14} max={3.14} step={0.01} value={((az + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI}
              onChange={(e) => setAz(Number(e.target.value))} aria-label={pt ? 'Girar na horizontal' : 'Rotate horizontally'} />
            <span className="label">{pt ? 'Inclinar' : 'Tilt'}</span>
            <input type="range" min={0.12} max={1.45} step={0.01} value={el} onChange={(e) => setEl(Number(e.target.value))}
              aria-label={pt ? 'Girar na vertical' : 'Rotate vertically'} />
          </div>
          <p className="small muted" style={{ margin: '6px 0 0' }}>{pt ? 'Superfície 3D: a mesma função. Arraste ou use os controles para girar e inclinar.' : '3D surface: the same function. Drag or use the sliders to spin and tilt.'}</p>
        </div>
      </div>
      <div className="table-wrap" style={{ marginTop: 10 }}>
        <table className="hud">
          <thead><tr><th style={{ textTransform: 'none' }}>η</th><th>{pt ? 'erro no passo atual' : 'loss at this step'}</th><th>{pt ? 'onde terminou' : 'where it ended'}</th><th>{pt ? 'erro final' : 'final loss'}</th></tr></thead>
          <tbody>
            {paths.map((p, r) => {
              const now = p[Math.min(p.length - 1, t)];
              const end = p[p.length - 1];
              const bs = basinOf(end);
              const last2 = p.slice(-2);
              const moving = last2.length === 2 && Math.hypot(last2[0].a - last2[1].a, last2[0].b - last2[1].b) > 0.05;
              const where = bs === 'global' ? (pt ? 'mínimo global ✓' : 'global minimum ✓')
                : bs === 'local' && !moving ? (pt ? 'presa no mínimo local' : 'stuck in the local minimum')
                : moving ? (pt ? 'não para: pula de um lado para o outro' : 'never settles: bounces back and forth')
                : (pt ? 'parou em outro ponto' : 'stopped elsewhere');
              return (
                <tr key={r}>
                  <td><span style={{ color: runners[r].color }}>●</span> {runners[r].label}</td>
                  <td>{fx(now.loss, 3, lang)}</td><td>{t >= steps ? where : '…'}</td><td>{t >= steps ? fx(end.loss, 3, lang) : '…'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
