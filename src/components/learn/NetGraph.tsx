import { useEffect, useRef, useState } from 'react';
import { num, useLang } from '../../i18n';
import type { DeepNet } from '../../lib/deep';

export type Pulse = 'forward' | 'backward';

interface Props {
  net: DeepNet;
  /** raw inputs shown in the input nodes (Speed, Defense) */
  x: [number, number];
  /** acts[l] = values shown in the neurons of layer l (hidden layers, then the outputs) */
  acts: number[][];
  winner: number | null;
  labels?: boolean;
  digits?: number;
  /** changing `id` replays the animation */
  play?: { id: number; kind: Pulse };
  /** layer drawn with a gold ring: −1 inputs, 0.. layers */
  focus?: number | null;
  /** weights that just changed are drawn in gold */
  changed?: boolean;
  /** names of the hidden layers' neurons, e.g. [['h1','h2'], ['g1','g2']] */
  names?: string[][];
}

const H = 400;
const R = 28;
const C = {
  pos: '#29d3e8', neg: '#7b4dff', zero: '#5b6b82', text: '#e6ecf4', dim: '#a9b6c8',
  navy: '#0d1b2a', gold: '#f5b52a', gold4: '#ffd25e', red: '#e8312f',
};
const OUT_COLOR = ['#ff7a2f', '#2f7bff', '#4cc46a'];
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

/** y positions of a column with n neurons (+ an optional bias node at the bottom). */
function ys(n: number, bias: boolean): number[] {
  const top = 78;
  const bottom = bias ? 300 : 320;
  const out = n === 1 ? [(top + bottom) / 2] : Array.from({ length: n }, (_, i) => lerp(top, bottom, i / (n - 1)));
  if (bias) out.push(362);
  return out;
}

/**
 * Layered drawing of a network: x1, x2, −1 → hidden layer(s) → Fire, Water, Grass.
 * Same conventions as lesson 1: gold balls go forward, red balls carry the error back;
 * cyan edges are positive weights, violet negative, dashed zero.
 */
export default function NetGraph({ net, x, acts, winner, labels = true, digits = 1, play, focus = null, changed = false, names }: Props) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [u, setU] = useState<number | null>(null);
  const raf = useRef<number | null>(null);
  const L = net.layers.length;
  const W = 260 + L * 230;
  const colX = Array.from({ length: L + 1 }, (_, c) => lerp(120, W - 130, c / L));
  const colY = [ys(2, true), ...net.layers.map((layer, l) => ys(layer.length, l < L - 1))];

  useEffect(() => {
    if (!play || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const start = performance.now();
    const dur = 700 * L;
    const tick = (t: number) => {
      const k = Math.min(1, (t - start) / dur);
      setU(k);
      if (k < 1) raf.current = requestAnimationFrame(tick);
      else setU(null);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
    };
  }, [play?.id, play?.kind, L]); // eslint-disable-line react-hooks/exhaustive-deps

  interface Edge { x1: number; y1: number; x2: number; y2: number; w: number; layer: number }
  const edges: Edge[] = [];
  net.layers.forEach((layer, l) => layer.forEach((n, j) => {
    [...n.w, n.theta].forEach((w, i) => edges.push({ x1: colX[l], y1: colY[l][i], x2: colX[l + 1], y2: colY[l + 1][j], w, layer: l }));
  }));
  const maxW = net.layers.map((_, l) => Math.max(1e-9, ...edges.filter((e) => e.layer === l).map((e) => Math.abs(e.w))));

  const balls: { x: number; y: number; c: string }[] = [];
  if (u !== null && play) {
    const seg = Math.min(L - 1, Math.floor(u * L));
    const k = ease(u * L - seg);
    const fwd = play.kind === 'forward';
    const layer = fwd ? seg : L - 1 - seg;
    for (const e of edges.filter((q) => q.layer === layer)) {
      const t = fwd ? k : 1 - k;
      balls.push({ x: lerp(e.x1 + R, e.x2 - R, t), y: lerp(e.y1, e.y2, t), c: fwd ? C.gold4 : C.red });
    }
  }

  const inNames = pt ? ['Velocidade', 'Defesa', 'bias'] : ['Speed', 'Defense', 'bias'];
  const outNames = pt ? ['Fogo', 'Água', 'Planta'] : ['Fire', 'Water', 'Grass'];
  const header = (c: number) => {
    if (c === 0) return pt ? 'ENTRADA' : 'INPUT';
    if (c === L) return pt ? 'SAÍDA' : 'OUTPUT';
    return L > 2 ? (pt ? `OCULTA ${c}` : `HIDDEN ${c}`) : (pt ? 'CAMADA OCULTA' : 'HIDDEN LAYER');
  };
  const ring = (layer: number) => (focus === layer ? C.gold : C.pos);
  // label position along the edge: alternate so labels of the same column do not overlap
  const tFor = (l: number) => (l % 2 === 0 ? 0.3 : 0.7);

  return (
    <svg className="graph" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={pt ? `Rede com ${L - 1} camada(s) oculta(s) e 3 saídas` : `Network with ${L - 1} hidden layer(s) and 3 outputs`}>
      <defs>
        <filter id="glow3" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3.5" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {colX.map((cx, c) => (
        <text key={c} className="node-label" x={cx} y={26} textAnchor="middle" fontSize={11} fill={focus === c - 1 ? C.gold4 : C.dim}>{header(c)}</text>
      ))}

      {edges.map((e, i) => (
        <line key={i} x1={e.x1 + R} y1={e.y1} x2={e.x2 - R} y2={e.y2}
          stroke={e.w === 0 ? C.zero : changed ? C.gold : e.w > 0 ? C.pos : C.neg}
          strokeOpacity={e.w === 0 ? 0.35 : 0.72} strokeDasharray={e.w === 0 ? '3 5' : undefined}
          strokeWidth={e.w === 0 ? 1 : 1.2 + 4 * (Math.abs(e.w) / maxW[e.layer])} />
      ))}
      {labels && edges.map((e, i) => {
        if (e.w === 0) return null;
        const t = tFor(e.layer);
        const mx = lerp(e.x1 + R, e.x2 - R, t);
        const my = lerp(e.y1, e.y2, t);
        const txt = num(Number(e.w.toFixed(digits)), lang);
        return (
          <g key={`l${i}`} transform={`translate(${mx}, ${my})`}>
            <rect x={-6 - txt.length * 3.6} y={-10} width={12 + txt.length * 7.2} height={18} fill={C.navy} stroke={changed ? C.gold : '#2a3547'} />
            <text textAnchor="middle" y={4} fontSize={11.5} fill={changed ? C.gold4 : e.w > 0 ? '#7df0ff' : '#b49bff'}>{txt}</text>
          </g>
        );
      })}

      {[x[0], x[1], -1].map((v, i) => (
        <g key={`in${i}`}>
          <circle cx={colX[0]} cy={colY[0][i]} r={R} fill={C.navy} stroke={ring(-1)} strokeWidth={2} strokeDasharray={i === 2 ? '5 4' : undefined} />
          <text x={colX[0]} y={colY[0][i] + 5} textAnchor="middle" fontSize={15} fill={C.text} fontWeight={600}>{num(v, lang)}</text>
          <text className="node-label" x={colX[0] - R - 8} y={colY[0][i] + 4} textAnchor="end" fontSize={10.5} fill={C.dim}>{inNames[i].toUpperCase()}</text>
        </g>
      ))}

      {net.layers.map((layer, l) => {
        const isOut = l === L - 1;
        return (
          <g key={`L${l}`}>
            {layer.map((_, j) => {
              const cx = colX[l + 1];
              const cy = colY[l + 1][j];
              const val = acts[l]?.[j] ?? 0;
              if (isOut) {
                return (
                  <g key={j}>
                    <circle cx={cx} cy={cy} r={R + 2} fill={C.navy} strokeWidth={winner === j ? 3.5 : 2}
                      stroke={winner === j ? OUT_COLOR[j] : focus === l ? C.gold : C.zero} filter={winner === j ? 'url(#glow3)' : undefined} />
                    <text x={cx} y={cy + 5} textAnchor="middle" fontSize={14} fill={C.text} fontWeight={600}>{num(Number(val.toFixed(Math.max(digits, 2))), lang)}</text>
                    <text className="node-label" x={cx + R + 12} y={cy + 4} fontSize={12} fill={OUT_COLOR[j]}>{outNames[j].toUpperCase()}</text>
                  </g>
                );
              }
              return (
                <g key={j}>
                  <circle cx={cx} cy={cy} r={R + 3} fill={C.navy} stroke={ring(l)} strokeWidth={2.5} filter={focus === l ? 'url(#glow3)' : undefined} />
                  <text x={cx} y={cy - 4} textAnchor="middle" fontSize={11} fill={C.dim}>{names?.[l]?.[j] ?? `h${j + 1}`}</text>
                  <text x={cx} y={cy + 13} textAnchor="middle" fontSize={14} fill={C.gold4} fontWeight={600}>{num(Number(val.toFixed(Math.max(digits, 2))), lang)}</text>
                </g>
              );
            })}
            {!isOut && (
              <g>
                <circle cx={colX[l + 1]} cy={colY[l + 1][layer.length]} r={R - 8} fill={C.navy} stroke={C.pos} strokeWidth={1.5} strokeDasharray="5 4" />
                <text x={colX[l + 1]} y={colY[l + 1][layer.length] + 5} textAnchor="middle" fontSize={13} fill={C.text}>−1</text>
              </g>
            )}
          </g>
        );
      })}

      {balls.map((b, i) => <circle key={i} cx={b.x} cy={b.y} r={6} fill={b.c} filter="url(#glow3)" />)}
    </svg>
  );
}
