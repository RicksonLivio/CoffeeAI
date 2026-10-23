import { useEffect, useMemo, useRef, useState } from 'react';
import { fx, pfx, useLang } from '../../i18n';
import { deepEpoch, deepHistory, flatDeep, type DeepNet } from '../../lib/deep';
import { POKE_NAME, STARTERS, TYPES } from '../../lib/mlp';
import NetGraph, { type Pulse } from './NetGraph';
import Note from './Note';

const PHASE_MS = 1700;

interface Props {
  init: DeepNet;
  eta: number;
  epochs: number;
  /** id prefix for anchors */
  id: string;
  /** neuron names per hidden layer */
  names: string[][];
}

/** Letters for the weights of each layer, as in the formulas: w/θ (first hidden), u/β (second hidden), v/φ (output). */
function letters(nLayers: number): [string, string][] {
  return nLayers === 2 ? [['w', 'θ'], ['v', 'φ']] : [['w', 'θ'], ['u', 'β'], ['v', 'φ']];
}

/** Lesson-1-style player over the epochs: forward → output error → error back → mean gradient → update. */
export default function EpochPlayer({ init, eta, epochs, id, names }: Props) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const history = useMemo(() => deepHistory(init, STARTERS, eta, epochs), [init, eta, epochs]);
  const [epoch, setEpoch] = useState(1);
  const [phase, setPhase] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [pid, setPid] = useState('venusaur');
  const [pulse, setPulse] = useState<{ id: number; kind: Pulse }>({ id: 0, kind: 'forward' });
  const timer = useRef<number | null>(null);

  const st = useMemo(() => deepEpoch(history[epoch - 1], STARTERS, eta, epoch), [history, epoch, eta]);
  const nL = init.layers.length;
  const LET = letters(nL);
  const outNames = pt ? ['Fogo', 'Água', 'Planta'] : ['Fire', 'Water', 'Grass'];
  const si = STARTERS.findIndex((s) => s.id === pid);
  const s = STARTERS[si];
  const b = st.samples[si];
  const f = b.forward;
  const d4 = (v: number) => fx(v, 4, lang);
  const p4 = (v: number) => pfx(v, 4, lang);
  const updated = phase === 4;
  const shownNet = updated ? st.after : st.before;
  const shownF = updated ? deepEpoch(st.after, STARTERS, eta, epoch + 1).samples[si].forward : f;

  const PHASES = pt
    ? ['Para a frente', 'Erro na saída', 'Erro para trás', 'Gradiente médio', 'Atualização']
    : ['Forward', 'Output error', 'Error backward', 'Mean gradient', 'Update'];

  // replay the ball animation when the phase that moves data starts
  useEffect(() => {
    if (phase === 0) setPulse((p) => ({ id: p.id + 1, kind: 'forward' }));
    if (phase === 2) setPulse((p) => ({ id: p.id + 1, kind: 'backward' }));
  }, [phase, epoch]);

  useEffect(() => {
    if (!playing) return;
    timer.current = window.setTimeout(() => {
      if (phase < 4) setPhase(phase + 1);
      else if (epoch < epochs) { setEpoch(epoch + 1); setPhase(0); }
      else setPlaying(false);
    }, PHASE_MS / speed);
    return () => { if (timer.current !== null) window.clearTimeout(timer.current); };
  }, [playing, phase, epoch, epochs, speed]);

  const goEpoch = (e: number) => { setPlaying(false); setEpoch(Math.max(1, Math.min(epochs, e))); setPhase(0); };
  const next = () => { setPlaying(false); if (phase < 4) setPhase(phase + 1); else if (epoch < epochs) { setEpoch(epoch + 1); setPhase(0); } };
  const back = () => { setPlaying(false); if (phase > 0) setPhase(phase - 1); else if (epoch > 1) { setEpoch(epoch - 1); setPhase(4); } };

  // flat list of weights with names, for the gradient and update tables
  const rows = st.before.layers.flatMap((layer, l) => layer.flatMap((n, j) => [...n.w, n.theta].map((w, i) => {
    const isBias = i === n.w.length;
    const name = isBias ? `${LET[l][1]}${j + 1}` : `${LET[l][0]}${j + 1},${i + 1}`;
    const g = isBias ? st.grad.layers[l][j].theta : st.grad.layers[l][j].w[i];
    const a = isBias ? st.after.layers[l][j].theta : st.after.layers[l][j].w[i];
    const mine = isBias ? b.grad.layers[l][j].theta : b.grad.layers[l][j].w[i];
    return { l, name, w, g, a, mine };
  })));
  const maxStep = Math.max(...rows.map((r) => Math.abs(r.a - r.w)));
  const inputName = (l: number, i: number) => (l === 0 ? `x${i + 1}` : `${names[l - 1][i]}`);
  const layerTitle = (l: number) => (l === nL - 1 ? (pt ? 'saída' : 'output') : nL > 2 ? (pt ? `oculta ${l + 1}` : `hidden ${l + 1}`) : (pt ? 'oculta' : 'hidden'));
  const layerLabel = (l: number, cap: boolean) => {
    const t = l === nL - 1 ? (pt ? 'saída' : 'output') : pt ? `camada oculta${nL > 2 ? ` ${l + 1}` : ''}` : `hidden layer${nL > 2 ? ` ${l + 1}` : ''}`;
    return cap ? t.charAt(0).toUpperCase() + t.slice(1) : t;
  };
  const neuronName = (l: number, j: number) => (l === nL - 1 ? outNames[j] : names[l][j]);
  const recent = Array.from({ length: Math.min(6, epoch) }, (_, k) => epoch - k).reverse();

  return (
    <div className="panel" id={id}>
      <div className="player">
        <button className="btn btn--gold btn--sm" onClick={() => setPlaying((p) => !p)}>
          {playing ? (pt ? '❚❚ Pausar' : '❚❚ Pause') : (pt ? '▶ Treinar' : '▶ Train')}
        </button>
        <button className="btn btn--ghost btn--sm" onClick={back} disabled={epoch === 1 && phase === 0}>◀ {pt ? 'Voltar' : 'Back'}</button>
        <button className="btn btn--ghost btn--sm" onClick={next} disabled={epoch === epochs && phase === 4}>{pt ? 'Próximo' : 'Next'} ▶</button>
        <button className="btn btn--ghost btn--sm" onClick={() => goEpoch(1)}>{pt ? 'Época 1' : 'Epoch 1'}</button>
        <button className="btn btn--ghost btn--sm" onClick={() => goEpoch(epoch + 10)}>+10</button>
        <button className="btn btn--ghost btn--sm" onClick={() => goEpoch(epoch + 100)}>+100</button>
        <label className="speed label">
          {pt ? 'Velocidade' : 'Speed'}
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            <option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option><option value={4}>4×</option><option value={8}>8×</option><option value={16}>16×</option>
          </select>
        </label>
      </div>
      <div className="range-row" style={{ marginBottom: 10 }}>
        <span className="label">{pt ? 'Época' : 'Epoch'}</span>
        <input type="range" min={1} max={epochs} value={epoch} onChange={(e) => goEpoch(Number(e.target.value))} aria-label={pt ? 'Época' : 'Epoch'} />
        <span className="mono gold">{epoch} / {epochs}</span>
        <span className="mono small">L = {d4(st.loss)} · {pt ? 'acertos' : 'correct'} {Math.round(st.accuracy * 9)}/9</span>
      </div>
      <div className="steps">
        {PHASES.map((p, i) => (
          <button key={p} className={`step-pill ${i === phase ? 'active' : i < phase ? 'done' : ''}`} onClick={() => { setPlaying(false); setPhase(i); }}>
            {i + 1}. {p}
          </button>
        ))}
      </div>
      <div className="poke-pick" role="radiogroup" aria-label="Pokémon">
        {STARTERS.map((q) => (
          <button key={q.id} role="radio" aria-checked={pid === q.id} className={`${q.type} ${pid === q.id ? 'active' : ''}`} onClick={() => setPid(q.id)}>
            {POKE_NAME[q.id]}
          </button>
        ))}
      </div>

      <div className="grid grid-side" style={{ alignItems: 'start' }}>
        <div>
          <NetGraph net={shownNet} x={[s.x1, s.x2]} acts={shownF.a.slice(1)} winner={shownF.p.indexOf(Math.max(...shownF.p))}
            play={pulse} digits={2} changed={updated} names={names} from={updated ? st.before : undefined} outAct="softmax"
            deltas={phase === 0 || updated ? undefined : b.delta.map((d, l) => (l === nL - 1 || phase >= 2 ? d : null))}
            focus={phase === 0 ? null : phase === 1 ? nL - 1 : phase === 2 ? 0 : null} />
          <p className="small muted" style={{ textAlign: 'center', margin: 0 }}>
            {updated
              ? (pt ? `Pesos novos (em dourado), já usados pela época ${epoch + 1}. Maior mudança nesta época: ${d4(maxStep)}.` : `New weights (gold), used from epoch ${epoch + 1}. Largest change this epoch: ${d4(maxStep)}.`)
              : phase === 0
                ? (pt ? `Época ${epoch}: os valores acendem camada por camada, da entrada para a saída. Pesos do início da época.` : `Epoch ${epoch}: values light up layer by layer, from input to output. Weights from the start of the epoch.`)
                : (pt ? `Em vermelho, o erro δ de cada neurônio${phase === 1 ? ' da saída' : ', voltando da saída para a entrada'}. Os pesos só mudam na fase 5.` : `In red, the error δ of each neuron${phase === 1 ? ' at the output' : ', going back from the output to the input'}. Weights only change in phase 5.`)}
          </p>
        </div>

        <div className="calc">
          {phase === 0 && (
            <div className="block on">
              <div className="title">1 · {PHASES[0]} · {POKE_NAME[s.id]} · x = ({fx(f.a[0][0], 2, lang)}, {fx(f.a[0][1], 2, lang)})</div>
              {st.before.layers.map((layer, l) => layer.map((n, j) => (
                <div key={`${l}${j}`} className="line">
                  z_{neuronName(l, j)} = {n.w.map((w, i) => `${pfx(w, 4, lang)}·${l === 0 ? fx(f.a[0][i], 2, lang) : d4(f.a[l][i])}`).join(' + ')} + {pfx(n.theta, 4, lang)}·(−1) = {d4(f.z[l][j])}
                  {l < nL - 1 && <> ⟹ {neuronName(l, j)} = σ = <span className="res">{d4(f.a[l + 1][j])}</span></>}
                </div>
              )))}
              <div className="line">p = softmax = ({f.p.map(d4).join(' ; ')})</div>
              <div className="line">L = −ln p_{outNames[TYPES.indexOf(s.type)]} = <span className={b.loss < 0.1 ? 'ok' : 'bad'}>{d4(b.loss)}</span></div>
              <div className="line muted">{pt ? `Os nove passam pela rede com os mesmos pesos. Erro médio da época: ${d4(st.loss)}.` : `All nine go through the network with the same weights. Mean loss of the epoch: ${d4(st.loss)}.`}</div>
            </div>
          )}
          {phase === 1 && (
            <div className="block on">
              <div className="title">2 · {PHASES[1]} · d = ({b.d.join(', ')})</div>
              {b.delta[nL - 1].map((v, k) => (
                <div key={k} className="line">δ_{outNames[k]} = p − d = {d4(f.p[k])} − {b.d[k]} = <span className="res">{d4(v)}</span></div>
              ))}
              <div className="line muted">
                {pt ? 'Positivo: a rede deu probabilidade demais a esse tipo. Negativo: deu de menos.' : 'Positive: the network gave that type too much probability. Negative: too little.'}
              </div>
            </div>
          )}
          {phase === 2 && (
            <div className="block on">
              <div className="title">3 · {PHASES[2]} · {pt ? 'com os pesos antigos' : 'with the old weights'}</div>
              {Array.from({ length: nL - 1 }, (_, k) => nL - 2 - k).map((l) => (
                <div key={l}>
                  <div className="line muted">{pt ? `${layerLabel(l, true)} ← erro vindo da ${layerLabel(l + 1, false)}:` : `${layerLabel(l, true)} ← error coming from the ${layerLabel(l + 1, false)}:`}</div>
                  {b.delta[l].map((v, j) => (
                    <div key={j} className="line">
                      δ_{neuronName(l, j)} = ({st.before.layers[l + 1].map((n, k) => `${pfx(n.w[j], 4, lang)}·${p4(b.delta[l + 1][k])}`).join(' + ')}) · {d4(f.a[l + 1][j])}·(1 − {d4(f.a[l + 1][j])}) = <span className="res">{d4(v)}</span>
                    </div>
                  ))}
                </div>
              ))}
              <div className="line muted">{pt ? 'Nenhum peso mudou ainda: os δ de trás usam os pesos que geraram a previsão.' : 'No weight has changed yet: the earlier δ use the weights that produced the prediction.'}</div>
            </div>
          )}
          {phase === 3 && (
            <div className="block on">
              <div className="title">4 · {PHASES[3]}</div>
              <div className="line">∂L/∂peso = {pt ? 'média, nos 9 Pokémon, de' : 'mean, over the 9 Pokémon, of'} δ · ({pt ? 'entrada do peso' : 'the weight’s input'})</div>
              <div className="line">{pt ? 'Ex.' : 'E.g.'} ∂L/∂{rows[0].name}: {POKE_NAME[s.id]} {pt ? 'contribui' : 'adds'} δ_{neuronName(0, 0)}·{inputName(0, 0)} = {p4(b.delta[0][0])}·{fx(f.a[0][0], 2, lang)} = {d4(rows[0].mine)}; {pt ? 'média dos 9' : 'mean of the 9'} = <span className="res">{d4(rows[0].g)}</span></div>
              <div className="line">‖∇L‖ = {d4(st.gradNorm)} {pt ? '(tamanho do vetor gradiente)' : '(length of the gradient vector)'}</div>
            </div>
          )}
          {phase === 4 && (
            <div className="block on">
              <div className="title">5 · {PHASES[4]} · <span style={{ textTransform: 'none' }}>η = {fx(eta, 1, lang)}</span></div>
              <div className="line">{pt ? 'peso novo = peso − η · ∂L/∂peso, para todos ao mesmo tempo' : 'new weight = weight − η · ∂L/∂weight, for all at the same time'}</div>
              {rows.slice(0, 3).map((r) => (
                <div key={r.name} className="line">{r.name} = {d4(r.w)} − {fx(eta, 1, lang)}·{p4(r.g)} = <span className="res">{d4(r.a)}</span></div>
              ))}
              <div className="line muted">{pt ? `… e os outros ${rows.length - 3} na tabela abaixo.` : `… and the other ${rows.length - 3} in the table below.`}</div>
            </div>
          )}
        </div>
      </div>

      {(phase === 3 || phase === 4) && (
        <div className="table-wrap" style={{ marginTop: 12 }}>
          <table className="hud">
            <thead>
              <tr>
                <th>{pt ? 'Peso' : 'Weight'}</th><th>{pt ? 'camada' : 'layer'}</th><th>{pt ? 'antes' : 'before'}</th>
                <th>∂L/∂{pt ? 'peso' : 'weight'} ({POKE_NAME[s.id]})</th><th>∂L/∂{pt ? 'peso' : 'weight'} ({pt ? 'média' : 'mean'})</th>
                {phase === 4 && <th>{pt ? 'depois' : 'after'}</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name}>
                  <td>{r.name}</td><td className="muted">{layerTitle(r.l)}</td><td>{d4(r.w)}</td><td className="muted">{d4(r.mine)}</td><td>{d4(r.g)}</td>
                  {phase === 4 && <td className="gold">{d4(r.a)}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="table-wrap" style={{ marginTop: 12 }}>
        <table className="hud">
          <thead><tr><th>{pt ? 'Época' : 'Epoch'}</th><th>{pt ? 'erro médio L' : 'mean loss L'}</th><th>{pt ? 'acertos' : 'correct'}</th><th>‖∇L‖</th><th>{pt ? 'maior mudança de peso' : 'largest weight change'}</th></tr></thead>
          <tbody>
            {recent.map((e) => {
              const q = e === epoch ? st : deepEpoch(history[e - 1], STARTERS, eta, e);
              const dmax = Math.max(...flatDeep(q.after).map((v, i) => Math.abs(v - flatDeep(q.before)[i])));
              return (
                <tr key={e} className={e === epoch ? 'current' : ''}>
                  <td>{e}</td><td>{d4(q.loss)}</td><td>{Math.round(q.accuracy * 9)}/9</td><td>{d4(q.gradNorm)}</td>
                  <td>{e < epoch || phase === 4 ? d4(dmax) : '·'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Note title={pt ? 'Ordem do backpropagation' : 'Order of backpropagation'}>
        {pt
          ? 'O cálculo dos erros anda de trás para a frente (saída → oculta), mas a atualização não segue essa ordem: primeiro se calculam todos os δ e todos os gradientes com os pesos antigos, e só então todos os pesos mudam juntos, na fase 5. Se os pesos da saída mudassem antes, o erro da camada oculta seria calculado com pesos que não geraram a previsão, e a conta ficaria errada.'
          : 'Errors are computed from back to front (output → hidden), but the update does not follow that order: first every δ and every gradient is computed with the old weights, and only then all weights change together, in phase 5. If the output weights changed first, the hidden-layer error would be computed with weights that did not produce the prediction, and the math would be wrong.'}
      </Note>
    </div>
  );
}
