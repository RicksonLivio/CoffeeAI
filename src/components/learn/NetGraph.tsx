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
  /** error δ of each neuron (same shape as acts), drawn in red under the node */
  deltas?: (number[] | null)[];
  /** edges ("l:j:i" = layer l, neuron j, input i) and nodes ("l:j", −1 = inputs) to emphasize */
  highlight?: { edges?: string[]; nodes?: string[] };
  /** weights before the update: when given, the labels and widths slide from `from` to `net` */
  from?: DeepNet;
  /** activation of the output layer, shown in its header (softmax, step…) */
  outAct?: string;
}

const H = 412;
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
  if (bias) out.push(386);
  return out;
}

/**
 * Layered drawing of a network: x1, x2, −1 → hidden layer(s) → Fire, Water, Grass.
 * Same conventions as lesson 1: gold balls go forward, red balls carry the error back;
 * cyan edges are positive weights, violet negative, dashed zero.
 */
export default function NetGraph({
  net: target, x, acts, winner, labels = true, digits = 1, play, focus = null, changed = false, names, deltas, highlight, from, outAct,
}: Props) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [u, setU] = useState<number | null>(null);
  const raf = useRef<number | null>(null);
  // tween of the weights during an update (from → target), so the change is visible on the graph
  const [k, setK] = useState(1);
  useEffect(() => {
    if (!from || window.matchMedia('(prefers-reduced-motion: reduce)').matches) { setK(1); return; }
    const start = performance.now();
    let id = 0;
    const tick = (t: number) => {
      const q = Math.min(1, (t - start) / 900);
      setK(q);
      if (q < 1) id = requestAnimationFrame(tick);
    };
    setK(0);
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [from, target]);
  const net: DeepNet = from && k < 1
    ? { layers: target.layers.map((layer, l) => layer.map((n, j) => ({
      w: n.w.map((w, i) => from.layers[l][j].w[i] + (w - from.layers[l][j].w[i]) * k),
      theta: from.layers[l][j].theta + (n.theta - from.layers[l][j].theta) * k,
    }))) }
    : target;
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

  interface Edge { x1: number; y1: number; x2: number; y2: number; w: number; layer: number; key: string }
  const edges: Edge[] = [];
  net.layers.forEach((layer, l) => layer.forEach((n, j) => {
    [...n.w, n.theta].forEach((w, i) => edges.push({ x1: colX[l], y1: colY[l][i], x2: colX[l + 1], y2: colY[l + 1][j], w, layer: l, key: `${l}:${j}:${i}` }));
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

  // while the animation runs, values appear layer by layer: forward from the input, errors from the output
  const animSeg = u !== null && play ? Math.min(L, Math.floor(u * L)) : L;
  const valueShown = (l: number) => !(play?.kind === 'forward' && u !== null) || l < animSeg;
  const deltaShown = (l: number) => !(play?.kind === 'backward' && u !== null) || l >= L - 1 - animSeg;
  const hiE = new Set(highlight?.edges ?? []);
  const hiN = new Set(highlight?.nodes ?? []);
  const anyHi = hiE.size + hiN.size > 0;

  const inNames = pt ? ['Velocidade', 'Defesa', 'bias'] : ['Speed', 'Defense', 'bias'];
  const outNames = pt ? ['Fogo', 'Água', 'Planta'] : ['Fire', 'Water', 'Grass'];
  const header = (c: number) => {
    if (c === 0) return pt ? 'ENTRADA' : 'INPUT';
    if (c === L) return `${pt ? 'SAÍDA' : 'OUTPUT'}${outAct ? ` · ${outAct}` : ''}`;
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

      {edges.map((e, i) => {
        const hi = hiE.has(e.key);
        return (
          <line key={i} x1={e.x1 + R} y1={e.y1} x2={e.x2 - R} y2={e.y2}
            stroke={hi ? C.red : e.w === 0 ? C.zero : changed ? C.gold : e.w > 0 ? C.pos : C.neg}
            strokeOpacity={hi ? 1 : anyHi ? 0.18 : e.w === 0 ? 0.35 : 0.72} strokeDasharray={e.w === 0 && !hi ? '3 5' : undefined}
            strokeWidth={hi ? 4 : e.w === 0 ? 1 : 1.2 + 4 * (Math.abs(e.w) / maxW[e.layer])} filter={hi ? 'url(#glow3)' : undefined} />
        );
      })}
      {labels && edges.map((e, i) => {
        if (e.w === 0 || (anyHi && !hiE.has(e.key))) return null;
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
          <circle cx={colX[0]} cy={colY[0][i]} r={R} fill={C.navy} stroke={hiN.has(`-1:${i}`) ? C.red : ring(-1)} strokeWidth={hiN.has(`-1:${i}`) ? 3.5 : 2} strokeDasharray={i === 2 ? '5 4' : undefined} />
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
                    <circle cx={cx} cy={cy} r={R + 2} fill={C.navy} strokeWidth={hiN.has(`${l}:${j}`) || winner === j ? 3.5 : 2}
                      stroke={hiN.has(`${l}:${j}`) ? C.red : winner === j && valueShown(l) ? OUT_COLOR[j] : focus === l ? C.gold : C.zero}
                      filter={winner === j || hiN.has(`${l}:${j}`) ? 'url(#glow3)' : undefined} />
                    <text x={cx} y={cy + 5} textAnchor="middle" fontSize={14} fill={C.text} fontWeight={600}>{valueShown(l) ? num(Number(val.toFixed(Math.max(digits, 2))), lang) : '·'}</text>
                    <text className="node-label" x={cx + R + 12} y={cy + 4} fontSize={12} fill={OUT_COLOR[j]}>{outNames[j].toUpperCase()}</text>
                    {deltas?.[l] && deltaShown(l) && (
                      <text x={cx + R + 12} y={cy + 20} fontSize={11.5} fill={C.red} fontWeight={600}>δ {num(Number(deltas[l]![j].toFixed(3)), lang)}</text>
                    )}
                  </g>
                );
              }
              return (
                <g key={j}>
                  <circle cx={cx} cy={cy} r={R + 3} fill={C.navy} stroke={hiN.has(`${l}:${j}`) ? C.red : ring(l)} strokeWidth={hiN.has(`${l}:${j}`) ? 3.5 : 2.5}
                    filter={focus === l || hiN.has(`${l}:${j}`) ? 'url(#glow3)' : undefined} />
                  <text x={cx} y={cy - 4} textAnchor="middle" fontSize={11} fill={C.dim}>{names?.[l]?.[j] ?? `h${j + 1}`}</text>
                  <text x={cx} y={cy + 13} textAnchor="middle" fontSize={14} fill={C.gold4} fontWeight={600}>{valueShown(l) ? num(Number(val.toFixed(Math.max(digits, 2))), lang) : '·'}</text>
                  {deltas?.[l] && deltaShown(l) && (
                    <g>
                      <rect x={cx - 34} y={cy + R + 6} width={68} height={17} fill={C.navy} stroke={C.red} strokeOpacity={0.6} />
                      <text x={cx} y={cy + R + 19} textAnchor="middle" fontSize={11.5} fill={C.red} fontWeight={600}>δ {num(Number(deltas[l]![j].toFixed(3)), lang)}</text>
                    </g>
                  )}
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
