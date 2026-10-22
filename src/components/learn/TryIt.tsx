import { useMemo, useState } from 'react';
import { num, useLang } from '../../i18n';
import { BLASTOISE, CHARIZARD, LESSON_INIT, classify, train, type Sample, type Weights } from '../../lib/perceptron';
import BoundaryChart, { type ChartPoint } from './BoundaryChart';
import { pokeName, sumExpr } from './CalcPanel';

/** Venusaur (base Speed 80, Defense 83) is not in the training set. */
const POINTS: ChartPoint[] = [
  { id: 'charizard', x1: 100, x2: 78, kind: 'fire' },
  { id: 'blastoise', x1: 78, x2: 100, kind: 'water' },
  { id: 'venusaur', x1: 80, x2: 83, kind: 'grass' },
];
const ETAS = [1, 0.1, 0.01];

export default function TryIt({ trained }: { trained: Weights }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [w, setW] = useState<Weights>(trained);
  const [target, setTarget] = useState('venusaur');
  const [eta, setEta] = useState(1);
  const [order, setOrder] = useState<'cb' | 'bc'>('cb');

  const p = POINTS.find((q) => q.id === target)!;
  const r = classify(w, p.x1, p.x2);
  const products = [w.w1 * p.x1, w.w2 * p.x2, -w.theta];

  const retrain = useMemo(() => {
    const samples: Sample[] = order === 'cb' ? [CHARIZARD, BLASTOISE] : [BLASTOISE, CHARIZARD];
    return train(samples, LESSON_INIT, eta, 50);
  }, [eta, order]);
  const updates = retrain.steps.filter((s) => s.e !== 0).length;

  const slider = (key: keyof Weights, label: string, min: number, max: number, step: number) => (
    <label className="field">
      <span className="row"><span className="label">{label}</span><output>{num(w[key], lang)}</output></span>
      <input type="range" min={min} max={max} step={step} value={w[key]}
        onChange={(e) => setW((cur) => ({ ...cur, [key]: Number(e.target.value) }))} />
    </label>
  );

  return (
    <div className="grid grid-2">
      <div className="panel">
        <h3>{pt ? 'Mexa nos pesos' : 'Move the weights'}</h3>
        <div className="presets">
          <button className="btn btn--ghost btn--sm" onClick={() => setW(LESSON_INIT)}>{pt ? 'Iniciais' : 'Initial'}</button>
          <button className="btn btn--ghost btn--sm" onClick={() => setW({ w1: 100, w2: 78, theta: -0.5 })}>{pt ? 'Após Charizard' : 'After Charizard'}</button>
          <button className="btn btn--sm" onClick={() => setW(trained)}>{pt ? 'Treinados' : 'Trained'}</button>
        </div>
        {slider('w1', 'w1 · ' + (pt ? 'Velocidade' : 'Speed'), -120, 120, 1)}
        {slider('w2', 'w2 · ' + (pt ? 'Defesa' : 'Defense'), -120, 120, 1)}
        {slider('theta', 'θ · bias', -500, 500, 0.5)}

        <label className="field">
          <span className="label">Pokémon</span>
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            {POINTS.map((q) => <option key={q.id} value={q.id}>{pokeName(q.id)} (x1 = {q.x1}, x2 = {q.x2})</option>)}
          </select>
        </label>

        <div className="calc">
          <div className="block on">
            <div className="line">net = ({num(w.w1, lang)} · {p.x1}) + ({num(w.w2, lang)} · {p.x2}) + ({num(w.theta, lang)} · −1)</div>
            <div className="line">net = {sumExpr(products, lang)} = <span className="res">{num(r.net, lang)}</span></div>
            <div className="line">
              ⟹ y = {r.y}{' '}
              <span className="verdict" style={{ color: r.y === 1 ? '#ff7a2f' : '#2f7bff' }}>
                {r.y === 1 ? (pt ? '🔥 Fogo' : '🔥 Fire') : (pt ? '💧 Água' : '💧 Water')}
              </span>
            </div>
          </div>
        </div>
        {target === 'venusaur' && (
          <p className="small muted" style={{ margin: 0 }}>
            {pt
              ? 'Venusaur é do tipo Planta e não estava no treino. Com uma única saída 0/1, o perceptron só conhece dois lados: todo Pokémon cai em Fogo ou em Água. Reconhecer um terceiro tipo exige mais neurônios de saída.'
              : 'Venusaur is a Grass type and was not in the training data. With a single 0/1 output, the perceptron only knows two sides: every Pokémon lands on Fire or Water. Recognizing a third type needs more output neurons.'}
          </p>
        )}
      </div>

      <div className="panel">
        <h3>{pt ? 'Fronteira de decisão' : 'Decision boundary'}</h3>
        <BoundaryChart w={w} points={POINTS} />
      </div>

      <div className="panel panel--gold" style={{ gridColumn: '1 / -1' }}>
        <h3>{pt ? 'Treine de novo, com outras escolhas' : 'Train again with other choices'}</h3>
        <div className="presets" style={{ alignItems: 'center' }}>
          <span className="label">η</span>
          {ETAS.map((v) => (
            <button key={v} className={`btn btn--sm ${eta === v ? '' : 'btn--ghost'}`} onClick={() => setEta(v)}>{num(v, lang)}</button>
          ))}
          <span className="label" style={{ marginLeft: 12 }}>{pt ? 'Ordem' : 'Order'}</span>
          <button className={`btn btn--sm ${order === 'cb' ? '' : 'btn--ghost'}`} onClick={() => setOrder('cb')}>Charizard → Blastoise</button>
          <button className={`btn btn--sm ${order === 'bc' ? '' : 'btn--ghost'}`} onClick={() => setOrder('bc')}>Blastoise → Charizard</button>
        </div>
        <p style={{ margin: '0 0 10px' }} className="mono small">
          {pt ? 'Início' : 'Start'}: w1 = 0, w2 = 0, θ = 0,5 → {retrain.converged
            ? (pt ? `convergiu em ${retrain.epochs.length} épocas, com ${updates} ajustes` : `converged in ${retrain.epochs.length} epochs, with ${updates} updates`)
            : (pt ? 'não convergiu em 50 épocas' : 'did not converge in 50 epochs')}
          {' '}→ w1 = {num(retrain.final.w1, lang)}, w2 = {num(retrain.final.w2, lang)}, θ = {num(retrain.final.theta, lang)}
        </p>
        <div className="table-wrap">
          <table className="hud">
            <thead><tr><th>{pt ? 'Época' : 'Epoch'}</th><th>Pokémon</th><th>net</th><th>y</th><th>e</th><th>{pt ? 'pesos depois' : 'weights after'}</th></tr></thead>
            <tbody>
              {retrain.steps.map((s, i) => (
                <tr key={i}>
                  <td>{s.epoch}</td><td>{pokeName(s.sample.id)}</td><td>{num(s.net, lang)}</td><td>{s.y}</td>
                  <td className={s.e === 0 ? 'ok' : 'bad'}>{num(s.e, lang)}</td>
                  <td>({num(s.after.w1, lang)}, {num(s.after.w2, lang)}, {num(s.after.theta, lang)})</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="presets" style={{ marginTop: 12, marginBottom: 0 }}>
          <button className="btn btn--sm" onClick={() => setW(retrain.final)}>{pt ? 'Usar estes pesos no gráfico' : 'Use these weights in the chart'}</button>
        </div>
        <p className="small muted" style={{ margin: '10px 0 0' }}>
          {pt
            ? `Treinado agora, no seu navegador, com o mesmo código do modelo em cache. Observe a fronteira: com η menor os pesos mudam menos, mas a reta continua separando os dois.`
            : `Trained now, in your browser, with the same code as the cached model. Watch the boundary: with a smaller η the weights move less, but the line still separates the two.`}
        </p>
      </div>
    </div>
  );
}
