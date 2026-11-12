import { useEffect, useRef } from 'react';
import { num, useLang } from '../../../i18n';
import { CARDNET_SHAPES, type Tensor } from '../../../lib/cnn';

/** A small numeric grid drawn as squares; brighter = larger value (red tone for negatives when `signed`). */
export function BlockGrid({
  grid, hi, pad = 0, signed = false, cell = 34, caption, group, digits = 2, onPick, title,
}: {
  grid: number[][];
  /** cells to outline in gold, as "y,x" */
  hi?: Set<string>;
  /** outer rings drawn as padding (shown as dashed zero cells) */
  pad?: number;
  signed?: boolean;
  cell?: number;
  caption?: React.ReactNode;
  /** colour band per cell (pooling blocks) */
  group?: (y: number, x: number) => number;
  digits?: number;
  onPick?: (y: number, x: number) => void;
  title?: string;
}) {
  const { lang } = useLang();
  const flat = grid.flat();
  const max = Math.max(1e-9, ...flat.map(Math.abs));
  const GROUP = ['rgba(41,211,232,.18)', 'rgba(245,181,42,.18)', 'rgba(76,196,106,.18)', 'rgba(255,122,47,.18)'];
  return (
    <figure className="blk" style={{ margin: 0 }} aria-label={title}>
      <div className="blk-grid" style={{ gridTemplateColumns: `repeat(${grid[0].length}, ${cell}px)` }}>
        {grid.map((row, y) => row.map((v, x) => {
          const isPad = pad > 0 && (y < pad || x < pad || y >= grid.length - pad || x >= row.length - pad);
          const a = Math.abs(v) / max;
          const bg = isPad ? 'transparent'
            : group ? GROUP[group(y, x) % GROUP.length]
            : v < 0 && signed ? `rgba(232,49,47,${0.12 + 0.6 * a})` : `rgba(41,211,232,${0.08 + 0.55 * a})`;
          const on = hi?.has(`${y},${x}`);
          return (
            <span key={`${y}-${x}`} className={`blk-cell ${on ? 'on' : ''} ${isPad ? 'pad' : ''} ${onPick ? 'pick' : ''}`}
              style={{ width: cell, height: cell, background: bg, fontSize: cell < 30 ? '.62rem' : '.74rem' }}
              onClick={onPick ? () => onPick(y, x) : undefined}>
              {num(Number(v.toFixed(digits)), lang)}
            </span>
          );
        }))}
      </div>
      {caption && <figcaption className="small" style={{ marginTop: 6 }}>{caption}</figcaption>}
    </figure>
  );
}

/** One feature map (h×w numbers) painted on a canvas, min → dark, max → bright cyan. */
export function MapCanvas({ data, h, w, px = 56, tint = 'cyan', title }: { data: ArrayLike<number>; h: number; w: number; px?: number; tint?: 'cyan' | 'r' | 'g' | 'b' | 'gold'; title?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < h * w; i++) { mn = Math.min(mn, data[i]); mx = Math.max(mx, data[i]); }
    const img = ctx.createImageData(w, h), span = mx - mn || 1;
    const T: Record<string, [number, number, number]> = { cyan: [125, 240, 255], r: [255, 80, 70], g: [90, 230, 120], b: [80, 140, 255], gold: [255, 210, 94] };
    const [r, g, b] = T[tint];
    for (let i = 0; i < h * w; i++) {
      const t = (data[i] - mn) / span;
      img.data[i * 4] = 10 + t * (r - 10); img.data[i * 4 + 1] = 20 + t * (g - 20); img.data[i * 4 + 2] = 36 + t * (b - 36); img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data, h, w, tint]);
  return <canvas ref={ref} width={w} height={h} className="map" style={{ width: px, height: px * (h / w) }} title={title} aria-label={title} />;
}

/** RGB tensor (3×h×w, values centered on 0) as a colour picture. */
export function RgbCanvas({ t, px = 96, title }: { t: Tensor; px?: number; title?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d'); if (!ctx) return;
    const n = t.h * t.w, img = ctx.createImageData(t.w, t.h);
    for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) img.data[i * 4 + ch] = Math.round((t.d[ch * n + i] + 0.5) * 255);
    for (let i = 0; i < n; i++) img.data[i * 4 + 3] = 255;
    ctx.putImageData(img, 0, 0);
  }, [t]);
  return <canvas ref={ref} width={t.w} height={t.h} className="map" style={{ width: px, height: px * (t.h / t.w) }} title={title} aria-label={title} />;
}

/** Channel c of a tensor as a flat array. */
export const channel = (t: Tensor, c: number) => t.d.slice(c * t.h * t.w, (c + 1) * t.h * t.w);

/** The architecture as squares that shrink from the input to the output. */
export function ArchDiagram({ active, onPick }: { active?: string; onPick?: (key: string) => void }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const side = (s: number) => 24 + s * 3.4;
  const LABEL: Record<string, [string, string]> = {
    input: ['entrada', 'input'], conv1: ['conv 5×5 · 6 filtros', 'conv 5×5 · 6 filters'], pool1: ['max pool 2×2', 'max pool 2×2'],
    conv2: ['conv 3×3 · 8 filtros', 'conv 3×3 · 8 filters'], pool2: ['max pool 2×2', 'max pool 2×2'],
    flat: ['achatar', 'flatten'], out: ['densa + softmax', 'dense + softmax'],
  };
  let x = 10;
  const items = CARDNET_SHAPES.map((s) => {
    const isVec = s.key === 'flat' || s.key === 'out';
    const w = isVec ? 76 : side(s.s) + Math.min(s.c, 8) * 5;
    const it = { ...s, x, w, isVec };
    x += w + 30;
    return it;
  });
  const H = 200, mid = 92;
  return (
    <svg viewBox={`0 0 ${x} ${H}`} className="graph arch" role="img" aria-label={pt ? 'Arquitetura da rede de cartas' : 'Card network architecture'}>
      {items.map((it, i) => {
        const on = active === it.key;
        const stroke = on ? '#ffd25e' : '#29d3e8';
        let shape: React.ReactNode;
        if (it.key === 'flat') {
          shape = <rect x={it.x + it.w / 2 - 7} y={mid - 70} width={14} height={140} fill="rgba(41,211,232,.12)" stroke={stroke} />;
        } else if (it.key === 'out') {
          shape = (['#ff7a2f', '#2f7bff', '#4cc46a'] as const).map((c, j) => <circle key={c} cx={it.x + it.w / 2} cy={mid - 34 + j * 34} r={10} fill={c} stroke={stroke} />);
        } else {
          const s = side(it.s), n = Math.min(it.c, 8);
          shape = Array.from({ length: n }, (_, j) => (
            <rect key={j} x={it.x + (n - 1 - j) * 5} y={mid - s / 2 - (n - 1 - j) * 5 + (n - 1) * 2.5} width={s} height={s}
              fill={it.key === 'input' ? ['rgba(255,80,70,.35)', 'rgba(90,230,120,.35)', 'rgba(80,140,255,.4)'][j % 3] : `rgba(41,211,232,${0.07 + j * 0.03})`}
              stroke={stroke} strokeWidth={on ? 2 : 1} />
          ));
        }
        const dims = it.key === 'flat' ? '288' : it.key === 'out' ? '3' : `${it.s}×${it.s}×${it.c}`;
        return (
          <g key={it.key} style={{ cursor: onPick ? 'pointer' : undefined }} onClick={() => onPick?.(it.key)}>
            {shape}
            <text x={it.x + it.w / 2} y={H - 26} textAnchor="middle" fontSize={11} fill={on ? '#ffd25e' : '#e6ecf4'} className="node-label">{dims}</text>
            <text x={it.x + it.w / 2} y={H - 10} textAnchor="middle" fontSize={9.5} fill="#a9b6c8">{LABEL[it.key][pt ? 0 : 1]}</text>
            {i < items.length - 1 && (
              <path d={`M ${it.x + it.w + 6} ${mid} l 20 0 m -6 -5 l 6 5 l -6 5`} stroke="#f5b52a" fill="none" strokeWidth={1.6} />
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Probability bars for Fire / Water / Grass. */
export function ProbBars({ p, names }: { p: number[]; names: string[] }) {
  const { lang } = useLang();
  const best = p.indexOf(Math.max(...p));
  const cls = ['fire', 'water', 'grass'];
  return (
    <div className="probs">
      {p.map((v, j) => (
        <div key={j} className={`prob ${j === best ? 'best' : ''}`}>
          <span className={`t-${cls[j]}`}>{names[j]}</span>
          <span className="bar"><i className={cls[j]} style={{ width: `${Math.max(1, v * 100)}%` }} /></span>
          <output>{num(Number((v * 100).toFixed(1)), lang)} %</output>
        </div>
      ))}
    </div>
  );
}
