import { useMemo, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ReferenceDot, ResponsiveContainer, XAxis, YAxis } from 'recharts';
import { fx, num, useLang } from '../../i18n';
import { sigmoid, STARTERS } from '../../lib/mlp';

/**
 * One input (Speed − Defense), two sigmoid hidden neurons, one linear output:
 *   y(x) = v1·σ(k·(x − c1)) + v2·σ(k·(x − c2))
 * With c1 = −11, c2 = 11, v1 = 1, v2 = −1 the sum is a "bump": ≈ 1 inside the band, ≈ 0 outside.
 */
export default function CurvesDemo() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [k, setK] = useState(0.4);
  const [c1, setC1] = useState(-11);
  const [c2, setC2] = useState(11);
  const [v1, setV1] = useState(1);
  const [v2, setV2] = useState(-1);

  const data = useMemo(() => Array.from({ length: 161 }, (_, i) => {
    const x = -40 + i * 0.5;
    const h1 = v1 * sigmoid(k * (x - c1));
    const h2 = v2 * sigmoid(k * (x - c2));
    return { x, h1, h2, y: h1 + h2 };
  }), [k, c1, c2, v1, v2]);
  const y = (x: number) => v1 * sigmoid(k * (x - c1)) + v2 * sigmoid(k * (x - c2));
  const dots = STARTERS.map((s) => ({ id: s.id, x: s.x1 - s.x2, type: s.type }));
  const color = { fire: '#ff7a2f', water: '#2f7bff', grass: '#4cc46a' } as const;

  const slider = (label: string, v: number, set: (n: number) => void, min: number, max: number, step: number) => (
    <label className="field">
      <span className="row"><span className="label">{label}</span><output>{num(v, lang)}</output></span>
      <input type="range" min={min} max={max} step={step} value={v} onChange={(e) => set(Number(e.target.value))} />
    </label>
  );

  return (
    <div className="grid grid-side">
      <div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 10, right: 16, bottom: 22, left: 4 }}>
            <CartesianGrid stroke="#2a3547" strokeDasharray="3 3" />
            <XAxis dataKey="x" type="number" domain={[-40, 40]} ticks={[-40, -22, -11, 0, 11, 22, 40]} stroke="#a9b6c8" fontSize={11}
              label={{ value: pt ? 'x = Velocidade − Defesa' : 'x = Speed − Defense', position: 'insideBottom', offset: -10, fill: '#a9b6c8', fontSize: 11 }} />
            <YAxis domain={[-1.2, 1.2]} ticks={[-1, -0.5, 0, 0.5, 1]} stroke="#a9b6c8" fontSize={11} width={40} tickFormatter={(v) => fx(v, 1, lang)} />
            <Legend formatter={(v) => (v === 'y' ? (pt ? 'saída y = soma' : 'output y = sum') : v === 'h1' ? 'v1·σ(k(x − c1))' : 'v2·σ(k(x − c2))')} />
            <Line dataKey="h1" stroke="#29d3e8" strokeDasharray="5 4" dot={false} isAnimationActive={false} />
            <Line dataKey="h2" stroke="#7b4dff" strokeDasharray="5 4" dot={false} isAnimationActive={false} />
            <Line dataKey="y" stroke="#ffd25e" strokeWidth={3} dot={false} isAnimationActive={false} />
            {dots.map((d) => <ReferenceDot key={d.id} x={d.x} y={y(d.x)} r={5} fill={color[d.type]} stroke="#e6ecf4" />)}
          </LineChart>
        </ResponsiveContainer>
        <p className="small muted" style={{ margin: 0 }}>
          {pt
            ? `Pontos: os nove iniciais na curva amarela. Com os valores iniciais, Planta (x = −3, −4) fica em y ≈ ${fx(y(-3), 2, lang)}, Fogo (x = 22) em ${fx(y(22), 2, lang)} e Água (x = −22) em ${fx(y(-22), 2, lang)}.`
            : `Dots: the nine starters on the yellow curve. With the initial values, Grass (x = −3, −4) sits at y ≈ ${fx(y(-3), 2, lang)}, Fire (x = 22) at ${fx(y(22), 2, lang)} and Water (x = −22) at ${fx(y(-22), 2, lang)}.`}
        </p>
      </div>
      <div>
        {slider(pt ? 'k · inclinação' : 'k · steepness', k, setK, 0.05, 2, 0.05)}
        {slider('c1', c1, setC1, -30, 30, 1)}
        {slider('c2', c2, setC2, -30, 30, 1)}
        {slider('v1', v1, setV1, -1, 1, 0.1)}
        {slider('v2', v2, setV2, -1, 1, 0.1)}
        <button className="btn btn--ghost btn--sm" onClick={() => { setK(0.4); setC1(-11); setC2(11); setV1(1); setV2(-1); }}>{pt ? 'Valores iniciais' : 'Initial values'}</button>
      </div>
    </div>
  );
}
