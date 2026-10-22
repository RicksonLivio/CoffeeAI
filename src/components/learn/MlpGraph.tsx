import { useEffect, useRef, useState } from 'react';
import { num, useLang } from '../../i18n';
import type { Net } from '../../lib/mlp';

export type Pulse = 'forward' | 'backward';

interface Props {
  net: Net;
  /** inputs shown in the input nodes (raw Speed and Defense) */
  x: [number, number];
  /** value shown in each hidden node (h1, h2) */
  h: number[];
  /** value shown in each output node */
  o: number[];
  /** index of the winning output, highlighted */
  winner: number | null;
  /** show weight values on the edges */
  labels?: boolean;
  /** decimals for edge labels and node values */
  digits?: number;
  /** changing this number replays the animation */
  play?: { id: number; kind: Pulse };
  /** highlight the layer being computed: 0 inputs, 1 hidden, 2 output */
  focus?: number | null;
  hiddenNames?: string[];
}

const W = 780;
const H = 400;
const COL = [118, 400, 660];
const IN_Y = [92, 200, 318];
const HID_Y = [128, 262, 352];
const OUT_Y = [96, 200, 304];
const R = 30;
const C = {
  pos: '#29d3e8', neg: '#7b4dff', zero: '#5b6b82', text: '#e6ecf4', dim: '#a9b6c8',
  navy: '#0d1b2a', gold: '#f5b52a', gold4: '#ffd25e', red: '#e8312f',
  fire: '#ff7a2f', water: '#2f7bff', grass: '#4cc46a',
};
const OUT_COLOR = [C.fire, C.water, C.grass];
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

interface Edge { x1: number; y1: number; x2: number; y2: number; w: number; t: number; layer: 0 | 1 }

/** Layered drawing of the 2-2-3 network: x1, x2, −1 → h1, h2, −1 → Fire, Water, Grass. */
export default function MlpGraph({ net, x, h, o, winner, labels = true, digits = 1, play, focus = null, hiddenNames }: Props) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [u, setU] = useState<number | null>(null);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    if (!play || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const start = performance.now();
    const dur = 1600;
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
  }, [play?.id, play?.kind]); // eslint-disable-line react-hooks/exhaustive-deps

  const edges: Edge[] = [];
  net.hidden.forEach((n, j) => {
    [...n.w, n.theta].forEach((w, i) => edges.push({ x1: COL[0], y1: IN_Y[i], x2: COL[1], y2: HID_Y[j], w, t: 0.3, layer: 0 }));
  });
  net.output.forEach((n, k) => {
    [...n.w, n.theta].forEach((w, j) => edges.push({ x1: COL[1], y1: HID_Y[j], x2: COL[2], y2: OUT_Y[k], w, t: 0.7, layer: 1 }));
  });
  const maxW = [0, 1].map((L) => Math.max(1e-9, ...edges.filter((e) => e.layer === L).map((e) => Math.abs(e.w))));

  const balls: { x: number; y: number; c: string }[] = [];
  if (u !== null && play) {
    const fwd = play.kind === 'forward';
    const phase = fwd ? (u < 0.5 ? 0 : 1) : u < 0.5 ? 1 : 0;
    const k = ease((u % 0.5) / 0.5);
    for (const e of edges.filter((q) => q.layer === phase)) {
      const t = fwd ? k : 1 - k;
      balls.push({ x: lerp(e.x1 + R, e.x2 - R, t), y: lerp(e.y1, e.y2, t), c: fwd ? C.gold4 : C.red });
    }
  }

  const inNames = pt ? ['Velocidade', 'Defesa', 'bias'] : ['Speed', 'Defense', 'bias'];
  const outNames = pt ? ['Fogo', 'Água', 'Planta'] : ['Fire', 'Water', 'Grass'];
  const hidNames = hiddenNames ?? ['h1', 'h2'];
  const ring = (layer: number) => (focus === layer ? C.gold : C.pos);

  return (
    <svg className="graph" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={pt ? 'MLP com 2 entradas, 2 neurônios ocultos e 3 saídas' : 'MLP with 2 inputs, 2 hidden neurons and 3 outputs'}>
      <defs>
        <filter id="glow2" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3.5" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {['entrada', 'oculta', 'saída'].map((_, i) => (
        <text key={i} className="node-label" x={COL[i]} y={26} textAnchor="middle" fontSize={11} fill={C.dim}>
          {(pt ? ['ENTRADA', 'CAMADA OCULTA', 'SAÍDA'] : ['INPUT', 'HIDDEN LAYER', 'OUTPUT'])[i]}
        </text>
      ))}

      {edges.map((e, i) => (
        <line key={i} x1={e.x1 + R} y1={e.y1} x2={e.x2 - R} y2={e.y2}
          stroke={e.w === 0 ? C.zero : e.w > 0 ? C.pos : C.neg}
          strokeOpacity={e.w === 0 ? 0.35 : 0.75}
          strokeDasharray={e.w === 0 ? '3 5' : undefined}
          strokeWidth={e.w === 0 ? 1 : 1.2 + 4 * (Math.abs(e.w) / maxW[e.layer])} />
      ))}
      {labels && edges.map((e, i) => {
        if (e.w === 0) return null;
        const mx = lerp(e.x1 + R, e.x2 - R, e.t);
        const my = lerp(e.y1, e.y2, e.t);
        const txt = num(Number(e.w.toFixed(digits)), lang);
        return (
          <g key={`l${i}`} transform={`translate(${mx}, ${my})`}>
            <rect x={-6 - txt.length * 3.6} y={-10} width={12 + txt.length * 7.2} height={18} fill={C.navy} stroke="#2a3547" />
            <text textAnchor="middle" y={4} fontSize={11.5} fill={e.w > 0 ? '#7df0ff' : '#b49bff'}>{txt}</text>
          </g>
        );
      })}

      {/* input nodes */}
      {[x[0], x[1], -1].map((v, i) => (
        <g key={`in${i}`}>
          <circle cx={COL[0]} cy={IN_Y[i]} r={R} fill={C.navy} stroke={ring(0)} strokeWidth={2} strokeDasharray={i === 2 ? '5 4' : undefined} />
          <text x={COL[0]} y={IN_Y[i] + 5} textAnchor="middle" fontSize={15} fill={C.text} fontWeight={600}>{num(v, lang)}</text>
          <text className="node-label" x={COL[0] - R - 8} y={IN_Y[i] + 4} textAnchor="end" fontSize={10.5} fill={C.dim}>
            {inNames[i].toUpperCase()}
          </text>
        </g>
      ))}

      {/* hidden nodes */}
      {[0, 1].map((j) => (
        <g key={`h${j}`}>
          <circle cx={COL[1]} cy={HID_Y[j]} r={R + 4} fill={C.navy} stroke={ring(1)} strokeWidth={2.5}
            filter={focus === 1 ? 'url(#glow2)' : undefined} />
          <text x={COL[1]} y={HID_Y[j] - 4} textAnchor="middle" fontSize={11} fill={C.dim}>{hidNames[j]}</text>
          <text x={COL[1]} y={HID_Y[j] + 13} textAnchor="middle" fontSize={14} fill={C.gold4} fontWeight={600}>
            {num(Number(h[j].toFixed(Math.max(digits, 2))), lang)}
          </text>
        </g>
      ))}
      <circle cx={COL[1]} cy={HID_Y[2]} r={R - 8} fill={C.navy} stroke={C.pos} strokeWidth={1.5} strokeDasharray="5 4" />
      <text x={COL[1]} y={HID_Y[2] + 5} textAnchor="middle" fontSize={13} fill={C.text}>−1</text>

      {/* output nodes */}
      {[0, 1, 2].map((k) => (
        <g key={`o${k}`}>
          <circle cx={COL[2]} cy={OUT_Y[k]} r={R + 2} fill={C.navy} strokeWidth={winner === k ? 3.5 : 2}
            stroke={winner === k ? OUT_COLOR[k] : focus === 2 ? C.gold : C.zero} filter={winner === k ? 'url(#glow2)' : undefined} />
          <text x={COL[2]} y={OUT_Y[k] + 5} textAnchor="middle" fontSize={14} fill={C.text} fontWeight={600}>
            {num(Number(o[k].toFixed(Math.max(digits, 2))), lang)}
          </text>
          <text className="node-label" x={COL[2] + R + 12} y={OUT_Y[k] + 4} fontSize={12} fill={OUT_COLOR[k]}>
            {outNames[k].toUpperCase()}
          </text>
        </g>
      ))}

      {balls.map((b, i) => <circle key={i} cx={b.x} cy={b.y} r={6} fill={b.c} filter="url(#glow2)" />)}
    </svg>
  );
}
