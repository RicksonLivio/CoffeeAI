import {
  CartesianGrid, Customized, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts';
import type { PokeType } from '../../lib/mlp';

export interface RegionPoint {
  id: string;
  name: string;
  x: number;
  y: number;
  type: PokeType;
  /** extra line for the tooltip */
  info?: string;
}

export interface Segment {
  a: [number, number];
  b: [number, number];
  color?: string;
  label?: string;
}

interface Props {
  x: [number, number];
  y: [number, number];
  xLabel: string;
  yLabel: string;
  points: RegionPoint[];
  /** colours each cell of the plane by the predicted class */
  region?: (x: number, y: number) => PokeType;
  segments?: Segment[];
  polygon?: [number, number][];
  height?: number;
  cells?: number;
  ticks?: number[];
}

export const TYPE_COLOR: Record<PokeType, string> = { fire: '#ff7a2f', water: '#2f7bff', grass: '#4cc46a' };
const SHAPE = { fire: 'circle', water: 'square', grass: 'triangle' } as const;

/* eslint-disable @typescript-eslint/no-explicit-any */
function Layer(props: any) {
  const xa = Object.values(props.xAxisMap ?? {})[0] as any;
  const ya = Object.values(props.yAxisMap ?? {})[0] as any;
  if (!xa?.scale || !ya?.scale) return null;
  const sx = xa.scale;
  const sy = ya.scale;
  const { region, segments, polygon, x, y, cells } = props.opts as Props & { cells: number };
  const rects: JSX.Element[] = [];
  if (region) {
    const dx = (x[1] - x[0]) / cells;
    const dy = (y[1] - y[0]) / cells;
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const cx = x[0] + (i + 0.5) * dx;
        const cy = y[0] + (j + 0.5) * dy;
        const x0 = sx(x[0] + i * dx);
        const y0 = sy(y[0] + (j + 1) * dy);
        rects.push(
          <rect key={`${i}-${j}`} x={x0} y={y0} width={sx(x[0] + (i + 1) * dx) - x0 + 0.5} height={sy(y[0] + j * dy) - y0 + 0.5}
            fill={TYPE_COLOR[region(cx, cy)]} fillOpacity={0.16} />,
        );
      }
    }
  }
  return (
    <g>
      {rects}
      {polygon && (
        <polygon points={polygon.map(([a, b]) => `${sx(a)},${sy(b)}`).join(' ')} fill="#a9b6c8" fillOpacity={0.08}
          stroke="#a9b6c8" strokeWidth={1.5} strokeDasharray="4 4" />
      )}
      {segments?.map((s, i) => (
        <g key={i}>
          <line x1={sx(s.a[0])} y1={sy(s.a[1])} x2={sx(s.b[0])} y2={sy(s.b[1])} stroke={s.color ?? '#ffd25e'} strokeWidth={2.5} strokeDasharray="7 5" />
          {s.label && (
            <text x={sx(s.b[0]) - 6} y={sy(s.b[1]) + 14} textAnchor="end" fontSize={11} fill={s.color ?? '#ffd25e'} fontFamily="var(--font-mono)">
              {s.label}
            </text>
          )}
        </g>
      ))}
    </g>
  );
}

function Tip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const p: RegionPoint = payload[0].payload;
  return (
    <div className="panel panel--tight" style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', fontSize: '.78rem' }}>
      <b className={`t-${p.type}`}>{p.name}</b>
      {p.info && <><br />{p.info}</>}
    </div>
  );
}
/* eslint-enable @typescript-eslint/no-explicit-any */

/** Scatter of Pokémon over a plane, with optional class regions, lines and a shaded polygon. */
export default function RegionChart(props: Props) {
  const { x, y, xLabel, yLabel, points, height = 320, cells = 48, ticks } = props;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 10, right: 16, bottom: 28, left: 4 }}>
        <CartesianGrid stroke="#2a3547" strokeDasharray="3 3" />
        <XAxis type="number" dataKey="x" domain={x} ticks={ticks} stroke="#a9b6c8" fontSize={11} allowDataOverflow
          label={{ value: xLabel, position: 'insideBottom', offset: -14, fill: '#a9b6c8', fontSize: 11 }} />
        <YAxis type="number" dataKey="y" domain={y} ticks={ticks} stroke="#a9b6c8" fontSize={11} allowDataOverflow width={44}
          label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 10, fill: '#a9b6c8', fontSize: 11 }} />
        <ZAxis range={[130, 130]} />
        <Customized component={(p: object) => <Layer {...p} opts={{ ...props, cells }} />} />
        <Tooltip content={<Tip />} cursor={false} />
        {(['fire', 'water', 'grass'] as const).map((k) => {
          const data = points.filter((p) => p.type === k);
          return data.length ? (
            <Scatter key={k} data={data} fill={TYPE_COLOR[k]} stroke="#e6ecf4" strokeWidth={1.5} isAnimationActive={false} shape={SHAPE[k]} />
          ) : null;
        })}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
