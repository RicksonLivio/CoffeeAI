import { num, useLang } from '../../i18n';
import type { TrainStep } from '../../lib/perceptron';

/**
 * Phases of one training step, in the order the lesson explains them.
 * The graph follows the NFH app convention: gold balls carry data forward,
 * red balls carry the error back to the weights.
 */
export const PHASES = ['inputs', 'net', 'activation', 'error', 'update'] as const;
export type Phase = (typeof PHASES)[number];

interface Props {
  step: TrainStep;
  /** current phase (0..4) */
  phase: number;
  /** progress inside the phase, 0..1 */
  u: number;
  motion: boolean;
}

const W = 760;
const H = 340;
const IN_X = 124;
const SUM = { x: 420, y: 170, r: 44 };
const STEP = { x: 572, y: 170 };
const OUT = { x: 690, y: 170 };
const INPUTS = [
  { key: 'x1', y: 66, w: 'w1' as const },
  { key: 'x2', y: 170, w: 'w2' as const },
  { key: 'x0', y: 274, w: 'theta' as const },
];

const C = {
  edge: '#5b6b82',
  cyan: '#29d3e8',
  gold: '#f5b52a',
  gold4: '#ffd25e',
  red: '#e8312f',
  green: '#3ddc84',
  text: '#e6ecf4',
  dim: '#a9b6c8',
  navy: '#0d1b2a',
  fire: '#ff7a2f',
  water: '#2f7bff',
};

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

export default function PerceptronGraph({ step, phase, u, motion }: Props) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const s = step.sample;
  const xs = [s.x1, s.x2, -1];
  const inputNames = pt ? ['Velocidade', 'Defesa', 'Bias fixo'] : ['Speed', 'Defense', 'Fixed bias'];
  const updated = phase === 4 && u > 0.55;
  const weights = updated ? step.after : step.before;
  const changed = step.e !== 0;
  const isFire = step.y === 1;

  // balls
  const balls: { x: number; y: number; color: string }[] = [];
  if (motion) {
    if (phase === 0) {
      const t = ease(u);
      INPUTS.forEach((n) => balls.push({ x: lerp(IN_X + 34, SUM.x - SUM.r, t), y: lerp(n.y, SUM.y, t), color: C.gold4 }));
    } else if (phase === 1 && u < 0.5) {
      const t = ease(u / 0.5);
      balls.push({ x: lerp(SUM.x + SUM.r, STEP.x - 34, t), y: SUM.y, color: C.gold4 });
    } else if (phase === 2 && u < 0.6) {
      const t = ease(u / 0.6);
      balls.push({ x: lerp(STEP.x + 34, OUT.x - 32, t), y: OUT.y, color: C.gold4 });
    } else if (phase === 4 && changed && u < 0.6) {
      const t = ease(u / 0.6);
      if (t < 0.4) {
        const k = t / 0.4;
        balls.push({ x: lerp(OUT.x - 32, SUM.x + SUM.r, k), y: OUT.y + 30 * Math.sin(Math.PI * k), color: C.red });
      } else {
        const k = (t - 0.4) / 0.6;
        INPUTS.forEach((n) => balls.push({ x: lerp(SUM.x - SUM.r, IN_X + 34, k), y: lerp(SUM.y, n.y, k), color: C.red }));
      }
    }
  }

  const sumLit = phase >= 1;
  const stepLit = phase >= 2;
  const errLit = phase >= 3;

  return (
    <svg className="graph" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={pt ? 'Perceptron: entradas, pesos, soma, função degrau e saída' : 'Perceptron: inputs, weights, sum, step function and output'}>
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3.5" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <linearGradient id="sumfill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#12284a" /><stop offset="1" stopColor="#0a1424" />
        </linearGradient>
      </defs>

      {/* edges input → Σ, with weight and product labels */}
      {INPUTS.map((n, i) => {
        const wv = weights[n.w];
        const flash = phase === 4 && changed && u > 0.45 && u < 0.85;
        const mx = lerp(IN_X + 34, SUM.x - SUM.r, 0.42);
        const my = lerp(n.y, SUM.y, 0.42);
        const prod = step.products[i];
        return (
          <g key={n.key}>
            <line x1={IN_X + 34} y1={n.y} x2={SUM.x - SUM.r} y2={SUM.y}
              stroke={flash ? C.red : phase === 0 ? C.gold : C.cyan} strokeOpacity={phase === 0 || flash ? 0.95 : 0.55}
              strokeWidth={1.5 + Math.min(5, Math.log10(1 + Math.abs(wv)) * 1.6)} strokeDasharray={n.key === 'x0' ? '6 5' : undefined} />
            <g transform={`translate(${mx}, ${my - 14})`}>
              <rect x={-58} y={-13} width={116} height={22} fill={C.navy} stroke={flash ? C.red : updated && changed ? C.gold : '#2a3547'} />
              <text textAnchor="middle" y={3} fontSize={12.5} fill={updated && changed ? C.gold4 : C.text}>
                {n.w === 'theta' ? 'θ' : n.w} = {num(wv, lang)}
              </text>
            </g>
            {sumLit && (
              <text x={lerp(IN_X + 34, SUM.x - SUM.r, 0.8)} y={lerp(n.y, SUM.y, 0.8) + (i === 1 ? 20 : i === 0 ? -10 : 22)}
                textAnchor="middle" fontSize={11.5} fill={C.gold4}>
                {num(prod, lang)}
              </text>
            )}
          </g>
        );
      })}

      {/* inputs */}
      {INPUTS.map((n, i) => (
        <g key={n.key}>
          <circle cx={IN_X} cy={n.y} r={32} fill={C.navy} stroke={phase === 0 ? C.gold : C.cyan} strokeWidth={2}
            filter={phase === 0 ? 'url(#glow)' : undefined} />
          <text x={IN_X} y={n.y - 2} textAnchor="middle" fontSize={16} fill={C.text} fontWeight={600}>{num(xs[i], lang)}</text>
          <text x={IN_X} y={n.y + 15} textAnchor="middle" fontSize={10.5} fill={C.dim}>{n.key}</text>
          <text className="node-label" x={IN_X - 40} y={n.y + 4} textAnchor="end" fontSize={11} fill={C.dim}>
            {inputNames[i].toUpperCase()}
          </text>
        </g>
      ))}

      {/* Σ → step → output */}
      <line x1={SUM.x + SUM.r} y1={SUM.y} x2={STEP.x - 34} y2={STEP.y} stroke={sumLit ? C.cyan : C.edge} strokeWidth={2} />
      <line x1={STEP.x + 34} y1={STEP.y} x2={OUT.x - 32} y2={OUT.y} stroke={stepLit ? C.cyan : C.edge} strokeWidth={2} />

      <circle cx={SUM.x} cy={SUM.y} r={SUM.r} fill="url(#sumfill)" stroke={phase === 1 ? C.gold : C.cyan} strokeWidth={2.5}
        filter={phase === 1 ? 'url(#glow)' : undefined} />
      <text x={SUM.x} y={SUM.y - 6} textAnchor="middle" fontSize={22} fill={C.text}>Σ</text>
      <text x={SUM.x} y={SUM.y + 17} textAnchor="middle" fontSize={12.5} fill={sumLit ? C.gold4 : C.dim}>
        {sumLit ? `net = ${num(step.net, lang)}` : 'net'}
      </text>

      <rect x={STEP.x - 34} y={STEP.y - 30} width={68} height={60} fill={C.navy} stroke={phase === 2 ? C.gold : C.cyan} strokeWidth={2}
        filter={phase === 2 ? 'url(#glow)' : undefined} />
      <polyline points={`${STEP.x - 22},${STEP.y + 14} ${STEP.x},${STEP.y + 14} ${STEP.x},${STEP.y - 14} ${STEP.x + 22},${STEP.y - 14}`}
        fill="none" stroke={stepLit ? C.gold4 : C.dim} strokeWidth={2.5} />
      <text className="node-label" x={STEP.x} y={STEP.y + 48} textAnchor="middle" fontSize={10.5} fill={C.dim}>
        {pt ? 'DEGRAU' : 'STEP'}
      </text>

      <circle cx={OUT.x} cy={OUT.y} r={32} fill={C.navy} strokeWidth={3}
        stroke={!stepLit ? C.edge : isFire ? C.fire : C.water} filter={stepLit ? 'url(#glow)' : undefined} />
      <text x={OUT.x} y={OUT.y + 2} textAnchor="middle" fontSize={17} fill={C.text} fontWeight={600}>
        {stepLit ? `y = ${step.y}` : 'y'}
      </text>
      {stepLit && (
        <text className="node-label" x={OUT.x} y={OUT.y - 44} textAnchor="middle" fontSize={12} fill={isFire ? C.fire : C.water}>
          {isFire ? (pt ? 'FOGO' : 'FIRE') : (pt ? 'ÁGUA' : 'WATER')}
        </text>
      )}
      <text x={OUT.x} y={OUT.y + 52} textAnchor="middle" fontSize={11.5} fill={C.dim}>d = {s.d}</text>
      {errLit && (
        <text x={OUT.x} y={OUT.y + 72} textAnchor="middle" fontSize={13} fontWeight={600} fill={step.e === 0 ? C.green : C.red}>
          e = {num(step.e, lang)} {step.e === 0 ? '✓' : '✗'}
        </text>
      )}

      {/* formula strip */}
      <text x={SUM.x} y={H - 16} textAnchor="middle" fontSize={12} fill={C.dim}>
        net = w1·x1 + w2·x2 + θ·(−1)
      </text>

      {balls.map((b, i) => (
        <circle key={i} cx={b.x} cy={b.y} r={6.5} fill={b.color} filter="url(#glow)" />
      ))}
    </svg>
  );
}
