import { useEffect, useRef, useState } from 'react';
import { assets, CACHED_MODEL_PATH, loadCachedModel, type ModelSource } from '../api';
import { num, useLang } from '../i18n';
import type { CachedModel } from '../lib/model';
import BoundaryChart, { type ChartPoint } from './learn/BoundaryChart';
import CalcPanel, { pokeName } from './learn/CalcPanel';
import PerceptronGraph, { PHASES } from './learn/PerceptronGraph';
import TryIt from './learn/TryIt';

const PHASE_MS = 1300;
/** Manual stepping shows the end state of a phase (e.g. the updated weights in phase 5). */
const END = 0.999;
const LESSON_POINTS: ChartPoint[] = [
  { id: 'charizard', x1: 100, x2: 78, kind: 'fire' },
  { id: 'blastoise', x1: 78, x2: 100, kind: 'water' },
];

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const on = () => setReduced(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return reduced;
}

export default function LearnSheet() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [cached, setCached] = useState<{ model: CachedModel; source: ModelSource } | null>(null);

  useEffect(() => {
    loadCachedModel().then(setCached);
  }, []);

  return (
    <div className="sheet">
      <section className="section" style={{ marginTop: 0 }}>
        <span className="kicker">&gt; LESSON_01.EXE · {pt ? 'APRENDA COMIGO' : 'LEARN WITH ME'}</span>
        <h1 className="wordmark" style={{ fontSize: 'clamp(1.8rem, 5vw, 3.4rem)' }}>{pt ? 'O Perceptron' : 'The Perceptron'}</h1>
        <p className="lead">
          {pt
            ? 'Um neurônio artificial aprende a separar Fogo de Água. Charizard é rápido e tem menos defesa; Blastoise é o contrário. Vamos ver cada conta que o perceptron faz até acertar os dois.'
            : 'One artificial neuron learns to tell Fire from Water. Charizard is fast with less defense; Blastoise is the opposite. We will follow every calculation the perceptron makes until it gets both right.'}
        </p>
        <div className="parts">
          <a href="#inicio" onClick={(e) => { e.preventDefault(); document.getElementById('inicio')?.scrollIntoView(); }}>
            <b>1</b>{pt ? 'Início: o problema' : 'Start: the problem'}
          </a>
          <a href="#treino" onClick={(e) => { e.preventDefault(); document.getElementById('treino')?.scrollIntoView(); }}>
            <b>2</b>{pt ? 'Funcionamento e treinamento' : 'How it works and trains'}
          </a>
          <a href="#experimente" onClick={(e) => { e.preventDefault(); document.getElementById('experimente')?.scrollIntoView(); }}>
            <b>3</b>{pt ? 'Experimente' : 'Try it yourself'}
          </a>
        </div>
      </section>

      {/* ---------- 1. Start ---------- */}
      <section className="section" id="inicio" aria-labelledby="h-inicio">
        <div className="section-head">
          <h2 id="h-inicio">1 · {pt ? 'Início: montando o problema' : 'Start: setting up the problem'}</h2>
        </div>
        <div className="grid grid-2">
          <article className="panel poke fire">
            <img src={assets.card('charizard')} alt="Charizard card" width={128} loading="lazy" />
            <div>
              <h3>Charizard</h3>
              <span className="type fire">{pt ? 'Fogo' : 'Fire'}</span>
              <p className="statline">x1 · {pt ? 'Velocidade' : 'Speed'} <b>100</b></p>
              <p className="statline">x2 · {pt ? 'Defesa' : 'Defense'} <b>78</b></p>
              <p className="statline">d = <b>1</b> ({pt ? 'Fogo' : 'Fire'})</p>
            </div>
          </article>
          <article className="panel poke water">
            <img src={assets.card('blastoise')} alt="Blastoise card" width={128} loading="lazy" />
            <div>
              <h3>Blastoise</h3>
              <span className="type water">{pt ? 'Água' : 'Water'}</span>
              <p className="statline">x1 · {pt ? 'Velocidade' : 'Speed'} <b>78</b></p>
              <p className="statline">x2 · {pt ? 'Defesa' : 'Defense'} <b>100</b></p>
              <p className="statline">d = <b>0</b> ({pt ? 'Água' : 'Water'})</p>
            </div>
          </article>
        </div>

        <div className="grid grid-side" style={{ marginTop: 20 }}>
          <article className="panel">
            <h3>{pt ? 'O desenho do perceptron' : 'The shape of the perceptron'}</h3>
            <p>
              {pt
                ? 'O neurônio recebe as entradas, multiplica cada uma pelo seu peso, soma com o viés (bias) e passa o resultado por uma função de ativação:'
                : 'The neuron takes the inputs, multiplies each one by its weight, adds the bias and passes the result through an activation function:'}
            </p>
            <div className="formula">{pt
              ? `Velocidade (x1 = 100) ──[ w1 ]──╮
Defesa     (x2 = 78)  ──[ w2 ]──┼──▶ [ Σ ] ──▶ [ degrau ] ──▶ y
Bias fixo  (x0 = −1)  ──[ θ  ]──╯
          net = w1·x1 + w2·x2 + θ·(−1)`
              : `Speed      (x1 = 100) ──[ w1 ]──╮
Defense    (x2 = 78)  ──[ w2 ]──┼──▶ [ Σ ] ──▶ [ step ] ──▶ y
Fixed bias (x0 = −1)  ──[ θ  ]──╯
          net = w1·x1 + w2·x2 + θ·(−1)`}</div>
          </article>
          <article className="panel">
            <h3>{pt ? 'Convenções e regras' : 'Conventions and rules'}</h3>
            <ul className="arrows rules">
              <li>{pt ? 'Entradas' : 'Inputs'}: <span className="var">x1</span> = {pt ? 'Velocidade' : 'Speed'}, <span className="var">x2</span> = {pt ? 'Defesa' : 'Defense'}, <span className="var">x0 = −1</span> ({pt ? 'entrada fixa do bias' : 'fixed input for the bias'})</li>
              <li>{pt ? 'Pesos iniciais' : 'Initial weights'}: <span className="var">w1 = 0</span>, <span className="var">w2 = 0</span>, <span className="var">θ = {num(0.5, lang)}</span></li>
              <li>{pt ? 'Taxa de aprendizado' : 'Learning rate'}: <span className="var">η = 1</span> {pt ? '(para simplificar as contas)' : '(keeps the arithmetic simple)'}</li>
              <li>{pt ? 'Função degrau' : 'Step function'}: net ≥ 0 ⟹ y = 1 ({pt ? 'Fogo' : 'Fire'}); net &lt; 0 ⟹ y = 0 ({pt ? 'Água' : 'Water'})</li>
            </ul>
            <h3 style={{ marginTop: 14 }}>{pt ? 'Regra de aprendizagem do perceptron' : 'Perceptron learning rule'}</h3>
            <div className="formula">{`e  = d − y
Δw1 = η · e · x1
Δw2 = η · e · x2
Δθ  = η · e · (−1)`}</div>
            <p className="small muted" style={{ margin: 0 }}>
              {pt
                ? 'É a versão original da ideia do backpropagation: o erro da saída volta e corrige cada peso na proporção da sua entrada. Se e = 0, nada muda.'
                : 'This is the original form of the idea behind backpropagation: the output error flows back and corrects each weight in proportion to its input. If e = 0, nothing changes.'}
            </p>
          </article>
        </div>
      </section>

      {/* ---------- 2. Training ---------- */}
      <section className="section" id="treino" aria-labelledby="h-treino">
        <div className="section-head">
          <h2 id="h-treino">2 · {pt ? 'Funcionamento e treinamento: passo a passo' : 'How it works and trains: step by step'}</h2>
          {cached && (
            <span className="label">
              <span className={`led ${cached.source === 'cache' ? 'led--ok' : ''}`} />{' '}
              {cached.source === 'cache'
                ? (pt ? `modelo em cache · ${CACHED_MODEL_PATH}` : `cached model · ${CACHED_MODEL_PATH}`)
                : (pt ? 'cache indisponível · treinado no navegador' : 'cache unavailable · trained in the browser')}
            </span>
          )}
        </div>
        {cached ? <Trainer model={cached.model} /> : <div className="panel">{pt ? 'Carregando o modelo…' : 'Loading the model…'}</div>}
      </section>

      {/* ---------- 3. Try ---------- */}
      <section className="section" id="experimente" aria-labelledby="h-try">
        <div className="section-head">
          <h2 id="h-try">3 · {pt ? 'Experimente' : 'Try it yourself'}</h2>
          <span className="label">{pt ? 'laboratório' : 'lab'}</span>
        </div>
        {cached && <TryIt trained={cached.model.final} />}
      </section>
    </div>
  );
}

/** Player over the cached training trace: graph on the left, the same step as arithmetic on the right. */
function Trainer({ model }: { model: CachedModel }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const reduced = usePrefersReducedMotion();
  const steps = model.trace;
  const [idx, setIdx] = useState(0);
  const [clock, setClock] = useState(0); // 0..PHASES.length, phase = floor
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const raf = useRef<number | null>(null);
  const last = useRef<number | null>(null);
  // the animation loop reads and writes these refs; state only mirrors them for rendering
  const idxRef = useRef(0);
  const clockRef = useRef(0);

  const total = PHASES.length;
  const phase = Math.min(total - 1, Math.floor(clock));
  const u = clock >= total ? 1 : clock - Math.floor(clock);
  const step = steps[idx];

  const goto = (i: number, c = 0) => {
    idxRef.current = Math.max(0, Math.min(steps.length - 1, i));
    clockRef.current = c;
    setIdx(idxRef.current);
    setClock(c);
  };

  useEffect(() => {
    if (!playing) return;
    const tick = (t: number) => {
      const dt = last.current === null ? 0 : t - last.current;
      last.current = t;
      let c = clockRef.current + (dt * speed) / PHASE_MS;
      if (c >= total) {
        // end of this step: next sample, or stop at the end of training
        if (idxRef.current + 1 < steps.length) {
          idxRef.current += 1;
          setIdx(idxRef.current);
          c = 0;
        } else {
          c = total;
          setPlaying(false);
        }
      }
      clockRef.current = c;
      setClock(c);
      if (c < total) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
      last.current = null;
    };
  }, [playing, speed, steps.length, total]);

  const nextPhase = () => {
    setPlaying(false);
    if (phase < total - 1) goto(idx, phase + 1 + END);
    else if (idx + 1 < steps.length) goto(idx + 1, END);
    else goto(idx, total);
  };
  const prevPhase = () => {
    setPlaying(false);
    if (phase > 0) goto(idx, phase - 1 + END);
    else if (idx > 0) goto(idx - 1, total - 1 + END);
  };
  const play = () => {
    if (idx === steps.length - 1 && clock >= total) goto(0);
    setPlaying((p) => !p);
  };

  const phaseLabel = pt
    ? ['Entradas', 'Somatória', 'Ativação', 'Erro', 'Ajuste']
    : ['Inputs', 'Sum', 'Activation', 'Error', 'Update'];
  const shownW = phase === total - 1 && u > 0.55 ? step.after : step.before;
  const finished = idx === steps.length - 1 && clock >= total;
  const lastE1 = steps.filter((s) => s.epoch === 1).length - 1;
  const e1 = steps[lastE1];

  return (
    <>
      <div className="panel">
        <div className="player">
          <button className="btn btn--gold btn--sm" onClick={play}>
            {playing ? (pt ? '❚❚ Pausar' : '❚❚ Pause') : finished ? (pt ? '↺ Reiniciar' : '↺ Replay') : (pt ? '▶ Treinar' : '▶ Train')}
          </button>
          <button className="btn btn--ghost btn--sm" onClick={prevPhase} disabled={idx === 0 && phase === 0}>◀ {pt ? 'Voltar' : 'Back'}</button>
          <button className="btn btn--ghost btn--sm" onClick={nextPhase} disabled={finished}>{pt ? 'Próximo' : 'Next'} ▶</button>
          <button className="btn btn--ghost btn--sm" onClick={() => { setPlaying(false); goto(0); }}>{pt ? 'Do início' : 'From start'}</button>
          <label className="speed label">
            {pt ? 'Velocidade' : 'Speed'}
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
              <option value={0.5}>0.5×</option>
              <option value={1}>1×</option>
              <option value={2}>2×</option>
              <option value={4}>4×</option>
            </select>
          </label>
        </div>

        <div className="steps" role="tablist" aria-label={pt ? 'Passos do treino' : 'Training steps'}>
          {steps.map((s, i) => (
            <button key={i} role="tab" aria-selected={i === idx}
              className={`step-pill ${i === idx ? 'active' : i < idx ? 'done' : ''}`}
              onClick={() => { setPlaying(false); goto(i, END); }}>
              <span className={`dot ${i < idx || (i === idx && phase >= 3) ? (s.e === 0 ? 'ok' : 'err') : ''}`} />
              {pt ? 'Época' : 'Epoch'} {s.epoch} · {pokeName(s.sample.id)}
            </button>
          ))}
        </div>

        <div className="grid grid-side" style={{ alignItems: 'start' }}>
          <div>
            <PerceptronGraph step={step} phase={phase} u={u} motion={!reduced && playing} />
            <div className="legend" style={{ justifyContent: 'center' }}>
              {PHASES.map((p, i) => (
                <span key={p} className={i === phase ? 'gold' : i < phase ? 'cyan' : ''}>
                  {i + 1}. {phaseLabel[i]}
                </span>
              ))}
            </div>
            <div className="legend" style={{ justifyContent: 'center' }}>
              <span><i style={{ background: '#ffd25e', borderRadius: '50%' }} />{pt ? 'dados indo para a frente' : 'data going forward'}</span>
              <span><i style={{ background: '#e8312f', borderRadius: '50%' }} />{pt ? 'erro voltando aos pesos' : 'error going back to the weights'}</span>
            </div>
          </div>
          <CalcPanel step={step} eta={model.config.eta} phase={phase} />
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: 20 }}>
        <div className="panel">
          <h3>{pt ? 'Fronteira de decisão neste passo' : 'Decision boundary at this step'}</h3>
          <BoundaryChart w={shownW} points={LESSON_POINTS} height={300} />
        </div>
        <div className="panel">
          <h3>{pt ? 'Resumo do treino' : 'Training summary'}</h3>
          <div className="table-wrap">
            <table className="hud">
              <thead>
                <tr><th>{pt ? 'Época' : 'Epoch'}</th><th>Pokémon</th><th>net</th><th>y</th><th>e</th><th>w1</th><th>w2</th><th>θ</th></tr>
              </thead>
              <tbody>
                {steps.map((s, i) => (
                  <tr key={i} className={i === idx ? 'current' : ''}>
                    <td>{s.epoch}</td>
                    <td>{pokeName(s.sample.id)}</td>
                    <td>{i <= idx ? num(s.net, lang) : '·'}</td>
                    <td>{i <= idx ? s.y : '·'}</td>
                    <td className={s.e === 0 ? 'ok' : 'bad'}>{i < idx || (i === idx && phase >= 3) ? num(s.e, lang) : '·'}</td>
                    <td>{i < idx || (i === idx && phase === 4 && u > 0.55) ? num(s.after.w1, lang) : '·'}</td>
                    <td>{i < idx || (i === idx && phase === 4 && u > 0.55) ? num(s.after.w2, lang) : '·'}</td>
                    <td>{i < idx || (i === idx && phase === 4 && u > 0.55) ? num(s.after.theta, lang) : '·'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small" style={{ marginTop: 12 }}>
            {pt
              ? `Fim da época 1: w1 = ${num(e1.after.w1, lang)}, w2 = ${num(e1.after.w2, lang)}, θ = ${num(e1.after.theta, lang)}. Na época 2 o perceptron testa os mesmos Pokémon com esses pesos: se errar zero vezes, o treino terminou.`
              : `End of epoch 1: w1 = ${num(e1.after.w1, lang)}, w2 = ${num(e1.after.w2, lang)}, θ = ${num(e1.after.theta, lang)}. In epoch 2 the perceptron tests the same Pokémon with these weights: if it makes zero errors, training is over.`}
          </p>
          {finished && model.converged && (
            <p className="small ok" style={{ margin: 0 }}>
              {pt
                ? `Convergiu em ${model.epochs.length} épocas: erros por época = ${model.epochs.map((e) => e.errors).join(', ')}.`
                : `Converged in ${model.epochs.length} epochs: errors per epoch = ${model.epochs.map((e) => e.errors).join(', ')}.`}
            </p>
          )}
        </div>
      </div>

      <article className="panel panel--gold tip" style={{ marginTop: 20 }}>
        <span className="label">★ {pt ? 'Dica de ouro para a aula' : 'Golden tip for the class'}</span>
        <p style={{ margin: '8px 0 0' }}>
          {pt
            ? 'Reforce que o erro foi zero na época 2. Isso significa que o perceptron encontrou uma fronteira de decisão linear que separa perfeitamente o Charizard (muita velocidade, pouca defesa) do Blastoise (muita defesa, pouca velocidade). Com w1 = 22 e w2 = −22, a regra aprendida é simples: se a Velocidade passa a Defesa (por mais de 0,5/22), é Fogo. O aprendizado terminou com sucesso!'
            : 'Stress that the error was zero in epoch 2. It means the perceptron found a linear decision boundary that perfectly separates Charizard (high speed, low defense) from Blastoise (high defense, low speed). With w1 = 22 and w2 = −22, the learned rule is simple: if Speed beats Defense (by more than 0.5/22), it is Fire. Learning finished successfully!'}
        </p>
      </article>
    </>
  );
}
