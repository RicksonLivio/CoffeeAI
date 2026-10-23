import { useMemo, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ReferenceDot, ResponsiveContainer, XAxis, YAxis } from 'recharts';
import { fx, num, useLang } from '../../i18n';
import { curveY, INITIAL, isolated, PRESETS, type CurveParams } from '../../lib/curves';
import { sigmoid, STARTERS, type PokeType } from '../../lib/mlp';
import Note from './Note';

const COLOR = { fire: '#ff7a2f', water: '#2f7bff', grass: '#4cc46a' } as const;

export default function CurvesDemo() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const i = pt ? 0 : 1;
  const T = { fire: pt ? 'Fogo' : 'Fire', water: pt ? 'Água' : 'Water', grass: pt ? 'Planta' : 'Grass' };
  const [p, setP] = useState<CurveParams>(INITIAL);
  const [preset, setPreset] = useState('start');
  const set = (k: keyof CurveParams) => (v: number) => { setP((q) => ({ ...q, [k]: v })); setPreset(''); };

  const data = useMemo(() => Array.from({ length: 161 }, (_, j) => {
    const x = -40 + j * 0.5;
    const h1 = p.v1 * sigmoid(p.k * (x - p.c1));
    const h2 = p.v2 * sigmoid(p.k * (x - p.c2));
    return { x, h1, h2, y: h1 + h2 };
  }), [p]);
  const verdict = isolated(p);
  const current = PRESETS.find((q) => q.id === preset);

  const slider = (label: string, v: number, onSet: (n: number) => void, min: number, max: number, step: number, hint: string) => (
    <label className="field">
      <span className="row"><span className="label">{label}</span><output>{num(v, lang)}</output></span>
      <input type="range" min={min} max={max} step={step} value={v} onChange={(e) => onSet(Number(e.target.value))} />
      <span className="small muted">{hint}</span>
    </label>
  );

  return (
    <>
      <div className="grid grid-2">
        <div>
          <p style={{ marginTop: 0 }}>
            <b className="gold">{pt ? 'O que é esta simulação.' : 'What this simulation is.'}</b>{' '}
            {pt
              ? 'Uma mini rede neural com uma entrada só, x = Velocidade − Defesa, dois neurônios ocultos com sigmoide e uma saída que soma as respostas deles, cada uma multiplicada por um peso v. É uma versão em uma dimensão do que a saída Planta da parte 3 faz com h1 e h2.'
              : 'A mini neural network with a single input, x = Speed − Defense, two sigmoid hidden neurons and one output that adds their answers, each multiplied by a weight v. It is a one-dimensional version of what the Grass output in part 3 does with h1 and h2.'}
          </p>
          <p>
            <b className="gold">{pt ? 'O objetivo.' : 'The goal.'}</b>{' '}
            {pt
              ? 'Fazer a saída y ficar alta só para o tipo Planta (x = −3 e −4) e baixa para Fogo (x = +22) e Água (x = −22), usando apenas degraus suaves. Isso mostra, num gráfico que dá para ver inteiro, por que juntar neurônios permite desenhar formas que um neurônio sozinho não desenha.'
              : 'Make the output y high only for the Grass type (x = −3 and −4) and low for Fire (x = +22) and Water (x = −22), using only smooth steps. It shows, in a chart you can see all at once, why combining neurons draws shapes a single neuron cannot.'}
          </p>
        </div>
        <div>
          <div className="formula">{pt
            ? `x = Velocidade − Defesa
neurônio 1:  a1 = σ(k·(x − c1))     (tracejado azul)
neurônio 2:  a2 = σ(k·(x − c2))     (tracejado roxo)
saída:        y = v1·a1 + v2·a2      (linha amarela)

na notação da rede:  σ(w·x − θ)  com  w = k,  θ = k·c`
            : `x = Speed − Defense
neuron 1:  a1 = σ(k·(x − c1))     (dashed blue)
neuron 2:  a2 = σ(k·(x − c2))     (dashed violet)
output:     y = v1·a1 + v2·a2      (yellow line)

in network notation:  σ(w·x − θ)  with  w = k,  θ = k·c`}</div>
          <p className="small mono" style={{ margin: 0 }}>
            {pt ? 'Agora' : 'Now'}: {pt ? 'neurônio 1' : 'neuron 1'} w = {num(p.k, lang)}, θ = {fx(p.k * p.c1, 2, lang)} · {pt ? 'neurônio 2' : 'neuron 2'} w = {num(p.k, lang)}, θ = {fx(p.k * p.c2, 2, lang)}
          </p>
        </div>
      </div>

      <div className="presets" style={{ marginTop: 14 }}>
        <span className="label">{pt ? 'Experimente' : 'Try'}</span>
        {PRESETS.map((q) => (
          <button key={q.id} className={`btn btn--sm ${preset === q.id ? '' : 'btn--ghost'}`} onClick={() => { setP(q.p); setPreset(q.id); }}>{q.name[i]}</button>
        ))}
      </div>
      {current && <p className="small" style={{ marginTop: 0 }}><b className="gold">{current.name[i]}:</b> {current.look[i]}</p>}

      <div className="grid grid-side">
        <div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data} margin={{ top: 10, right: 16, bottom: 22, left: 4 }}>
              <CartesianGrid stroke="#2a3547" strokeDasharray="3 3" />
              <XAxis dataKey="x" type="number" domain={[-40, 40]} ticks={[-40, -22, -11, 0, 11, 22, 40]} stroke="#a9b6c8" fontSize={11}
                label={{ value: pt ? 'x = Velocidade − Defesa' : 'x = Speed − Defense', position: 'insideBottom', offset: -10, fill: '#a9b6c8', fontSize: 11 }} />
              <YAxis domain={[-1.2, 1.2]} ticks={[-1, -0.5, 0, 0.5, 1]} stroke="#a9b6c8" fontSize={11} width={40} tickFormatter={(v) => fx(v, 1, lang)} />
              <Legend verticalAlign="top" height={30} formatter={(v) => (v === 'y' ? (pt ? 'saída y = soma' : 'output y = sum') : v === 'h1' ? 'v1·a1' : 'v2·a2')} />
              <Line dataKey="h1" stroke="#29d3e8" strokeDasharray="5 4" dot={false} isAnimationActive={false} />
              <Line dataKey="h2" stroke="#7b4dff" strokeDasharray="5 4" dot={false} isAnimationActive={false} />
              <Line dataKey="y" stroke="#ffd25e" strokeWidth={3} dot={false} isAnimationActive={false} />
              {STARTERS.map((s) => <ReferenceDot key={s.id} x={s.x1 - s.x2} y={curveY(p, s.x1 - s.x2)} r={5} fill={COLOR[s.type]} stroke="#e6ecf4" />)}
            </LineChart>
          </ResponsiveContainer>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>{pt ? 'Tipo' : 'Type'}</th><th>x</th><th>y</th></tr></thead>
              <tbody>
                {(['water', 'grass', 'fire'] as PokeType[]).map((t) => {
                  const xs = [...new Set(STARTERS.filter((s) => s.type === t).map((s) => s.x1 - s.x2))];
                  const [lo, hi] = verdict.byType[t];
                  return <tr key={t}><td className={`t-${t}`}>{T[t]}</td><td>{xs.map((x) => num(x, lang)).join(pt ? ' e ' : ' and ')}</td><td>{Math.abs(lo - hi) < 0.005 ? fx(lo, 2, lang) : `${fx(lo, 2, lang)} ${pt ? 'a' : 'to'} ${fx(hi, 2, lang)}`}</td></tr>;
                })}
              </tbody>
            </table>
          </div>
          <p className={`verdict small ${verdict.type === 'grass' ? 'ok' : verdict.type ? 'gold' : 'bad'}`} style={{ marginTop: 8 }}>
            {verdict.type === 'grass'
              ? (pt ? `✓ Planta isolada: fica ${fx(verdict.gap, 2, lang)} acima dos outros dois tipos.` : `✓ Grass isolated: it sits ${fx(verdict.gap, 2, lang)} above the other two types.`)
              : verdict.type
                ? (pt ? `A curva isola ${T[verdict.type]}, não Planta.` : `The curve isolates ${T[verdict.type]}, not Grass.`)
                : (pt ? '✗ Nenhum tipo fica sozinho no alto: a curva não separa Planta.' : '✗ No type sits alone at the top: the curve does not separate Grass.')}
          </p>
        </div>
        <div>
          {slider(pt ? 'k · inclinação' : 'k · steepness', p.k, set('k'), 0.05, 2, 0.05,
            pt ? 'Quão abrupto é cada degrau. Na rede, é o tamanho do peso w.' : 'How abrupt each step is. In the network, it is the size of the weight w.')}
          {slider('c1', p.c1, set('c1'), -30, 30, 1, pt ? 'Onde o neurônio 1 muda de 0 para 1 (θ/w).' : 'Where neuron 1 switches from 0 to 1 (θ/w).')}
          {slider('c2', p.c2, set('c2'), -30, 30, 1, pt ? 'Onde o neurônio 2 muda de 0 para 1.' : 'Where neuron 2 switches from 0 to 1.')}
          {slider('v1', p.v1, set('v1'), -1, 1, 0.1, pt ? 'Peso de saída do neurônio 1: quanto ele vota e com que sinal.' : 'Output weight of neuron 1: how much it votes and with which sign.')}
          {slider('v2', p.v2, set('v2'), -1, 1, 0.1, pt ? 'Peso de saída do neurônio 2. Negativo = subtrai o degrau.' : 'Output weight of neuron 2. Negative = subtracts the step.')}
          <button className="btn btn--ghost btn--sm" onClick={() => { setP(INITIAL); setPreset('start'); }}>{pt ? 'Valores iniciais' : 'Initial values'}</button>
        </div>
      </div>

      <Note title={pt ? 'O que levar desta simulação' : 'What to take from this simulation'}>
        {pt
          ? 'Um neurônio sozinho só sabe dizer "acima ou abaixo de um ponto". Somando dois, com um peso positivo e outro negativo, aparece um morro, que é alto só numa faixa. Com mais neurônios dá para empilhar morros e aproximar qualquer curva. Na rede de verdade, o treino ajusta k, c e v sozinho (são os pesos w, θ e v), e em duas dimensões os degraus viram retas no plano Velocidade × Defesa, como nos mapas de regiões das partes 2 e 3.'
          : 'A single neuron can only say "above or below a point". Adding two, one with a positive and one with a negative weight, makes a bump that is high only in a band. With more neurons you can stack bumps and approximate any curve. In the real network, training adjusts k, c and v by itself (they are the weights w, θ and v), and in two dimensions the steps become lines on the Speed × Defense plane, as in the region maps of parts 2 and 3.'}
      </Note>
    </>
  );
}
