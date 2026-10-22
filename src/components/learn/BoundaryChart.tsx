import {
  CartesianGrid, Customized, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts';
import { num, useLang } from '../../i18n';
import { boundarySegment, classify, fireRegion, type Box, type Weights } from '../../lib/perceptron';
import { pokeName } from './CalcPanel';

export interface ChartPoint {
  id: string;
  x1: number;
  x2: number;
  /** real type, used for the marker colour */
  kind: 'fire' | 'water' | 'grass';
}

const BOX: Box = { min: 40, max: 140 };
const COLOR = { fire: '#ff7a2f', water: '#2f7bff', grass: '#4cc46a' };

interface Props {
  w: Weights;
  points: ChartPoint[];
  height?: number;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
/** Draws the Fire half-plane and the boundary w1·x1 + w2·x2 = θ using the chart's own scales. */
function RegionLayer(props: any & { w: Weights }) {
  const xa = Object.values(props.xAxisMap ?? {})[0] as any;
  const ya = Object.values(props.yAxisMap ?? {})[0] as any;
  if (!xa?.scale || !ya?.scale) return null;
  const sx = xa.scale;
  const sy = ya.scale;
  const w: Weights = props.w;
  const toPts = (pts: [number, number][]) => pts.map(([a, b]) => `${sx(a)},${sy(b)}`).join(' ');
  const square: [number, number][] = [[BOX.min, BOX.min], [BOX.max, BOX.min], [BOX.max, BOX.max], [BOX.min, BOX.max]];
  const fire = fireRegion(w, BOX);
  const seg = boundarySegment(w, BOX);
  return (
    <g>
      <polygon points={toPts(square)} fill={COLOR.water} fillOpacity={0.13} />
      {fire.length >= 3 && <polygon points={toPts(fire)} fill={COLOR.fire} fillOpacity={0.17} />}
      {seg && (
        <line x1={sx(seg[0][0])} y1={sy(seg[0][1])} x2={sx(seg[1][0])} y2={sy(seg[1][1])}
          stroke="#ffd25e" strokeWidth={2.5} strokeDasharray="7 5" />
      )}
    </g>
  );
}

function Tip({ active, payload, w }: any) {
  const { lang } = useLang();
  if (!active || !payload?.length) return null;
  const p: ChartPoint = payload[0].payload;
  const r = classify(w, p.x1, p.x2);
  return (
    <div className="panel panel--tight" style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', fontSize: '.78rem' }}>
      <b>{pokeName(p.id)}</b> · x1 = {p.x1}, x2 = {p.x2}
      <br />net = {num(r.net, lang)} → y = {r.y}
    </div>
  );
}
/* eslint-enable @typescript-eslint/no-explicit-any */

/** Decision boundary over the Speed × Defense plane (same Recharts stack as the NFH app). */
export default function BoundaryChart({ w, points, height = 320 }: Props) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const seg = boundarySegment(w, BOX);
  const allFire = fireRegion(w, BOX).length === 4 && !seg;

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 10, right: 16, bottom: 28, left: 4 }}>
          <CartesianGrid stroke="#2a3547" strokeDasharray="3 3" />
          <XAxis type="number" dataKey="x1" domain={[BOX.min, BOX.max]} ticks={[40, 60, 80, 100, 120, 140]} stroke="#a9b6c8" fontSize={11}
            allowDataOverflow label={{ value: pt ? 'x1 · Velocidade' : 'x1 · Speed', position: 'insideBottom', offset: -14, fill: '#a9b6c8', fontSize: 11 }} />
          <YAxis type="number" dataKey="x2" domain={[BOX.min, BOX.max]} ticks={[40, 60, 80, 100, 120, 140]} stroke="#a9b6c8" fontSize={11}
            allowDataOverflow width={44} label={{ value: pt ? 'x2 · Defesa' : 'x2 · Defense', angle: -90, position: 'insideLeft', offset: 10, fill: '#a9b6c8', fontSize: 11 }} />
          <ZAxis range={[160, 160]} />
          <Customized component={(p: object) => <RegionLayer {...p} w={w} />} />
          <Tooltip content={<Tip w={w} />} cursor={false} />
          {(['fire', 'water', 'grass'] as const).map((k) => {
            const data = points.filter((p) => p.kind === k);
            return data.length ? (
              <Scatter key={k} data={data} fill={COLOR[k]} stroke="#e6ecf4" strokeWidth={1.5} isAnimationActive={false}
                shape={k === 'water' ? 'square' : k === 'grass' ? 'triangle' : 'circle'} />
            ) : null;
          })}
        </ScatterChart>
      </ResponsiveContainer>
      <div className="legend">
        <span><i style={{ background: 'rgba(255,122,47,.45)' }} />{pt ? 'região Fogo (net ≥ 0)' : 'Fire region (net ≥ 0)'}</span>
        <span><i style={{ background: 'rgba(47,123,255,.4)' }} />{pt ? 'região Água (net < 0)' : 'Water region (net < 0)'}</span>
        <span><i style={{ background: 'transparent', borderTop: '2px dashed #ffd25e', height: 0, marginBottom: 3 }} />
          {pt ? 'fronteira' : 'boundary'}: {num(w.w1, lang)}·x1 + {num(w.w2, lang)}·x2 = {num(w.theta, lang)}</span>
      </div>
      {!seg && (
        <p className="small muted" style={{ marginTop: 6 }}>
          {w.w1 === 0 && w.w2 === 0
            ? pt ? 'Com w1 = w2 = 0 não existe fronteira: net = −θ para todo ponto.' : 'With w1 = w2 = 0 there is no boundary: net = −θ everywhere.'
            : allFire
              ? pt ? 'A fronteira passa fora do gráfico: todos os pontos caem do lado Fogo.' : 'The boundary is outside the chart: every point falls on the Fire side.'
              : pt ? 'A fronteira passa fora do gráfico: todos os pontos caem do lado Água.' : 'The boundary is outside the chart: every point falls on the Water side.'}
        </p>
      )}
    </div>
  );
}
