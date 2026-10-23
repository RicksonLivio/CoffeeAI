import { useEffect, useRef, useState } from 'react';
import { assets, loadCachedModel, type Loaded } from '../../api';
import { num, useLang } from '../../i18n';
import type { CachedModel } from '../../lib/model';
import BoundaryChart, { type ChartPoint } from './BoundaryChart';
import CalcPanel, { pokeName } from './CalcPanel';
import Note from './Note';
import PerceptronGraph, { PHASES } from './PerceptronGraph';
import TryIt from './TryIt';

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

export default function Lesson1() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [cached, setCached] = useState<Loaded<CachedModel> | null>(null);

  useEffect(() => {
    loadCachedModel().then(setCached);
  }, []);

  const jump = (id: string) => (e: React.MouseEvent) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView(); };

  return (
    <div className="lesson-body">
      <div className="parts">
        <a href="#l1-inicio" onClick={jump('l1-inicio')}><b>1</b>{pt ? 'Início: o problema' : 'Start: the problem'}</a>
        <a href="#l1-treino" onClick={jump('l1-treino')}><b>2</b>{pt ? 'Funcionamento e treinamento' : 'How it works and trains'}</a>
        <a href="#l1-experimente" onClick={jump('l1-experimente')}><b>3</b>{pt ? 'Experimente' : 'Try it yourself'}</a>
      </div>

      {/* ---------- 1. Start ---------- */}
      <section className="section" id="l1-inicio" aria-labelledby="l1-h-inicio">
        <div className="section-head">
          <h2 id="l1-h-inicio">1 · {pt ? 'Início: montando o problema' : 'Start: setting up the problem'}</h2>
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
        <Note title={pt ? 'Por que Velocidade e Defesa?' : 'Why Speed and Defense?'}>
          {pt
            ? 'São dois atributos base dos jogos. Charizard tem mais Velocidade do que Defesa; Blastoise, o contrário. Essa diferença é o padrão que o neurônio vai aprender a enxergar.'
            : 'They are two base stats from the games. Charizard has more Speed than Defense; Blastoise is the opposite. That difference is the pattern the neuron will learn to see.'}
        </Note>

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
            <Note title={pt ? 'O que é o bias?' : 'What is the bias?'}>
              {pt
                ? 'θ funciona como um limiar: o neurônio só responde 1 quando w1·x1 + w2·x2 passa de θ. x0 = −1 não é peso: é uma entrada fixa, que nunca muda; o peso dela é θ. Ligar o bias a essa entrada deixa θ ser treinado como mais um peso, com a mesma regra. Por isso o termo que entra na soma é θ·(−1) = −θ, e quando o neurônio erra por não acender (e = 1), Δθ = η·e·(−1) é negativo: o limiar baixa, e pode ficar abaixo de zero, como no primeiro passo do Charizard (0,5 → −0,5).'
                : 'θ works as a threshold: the neuron only answers 1 when w1·x1 + w2·x2 is above θ. x0 = −1 is not a weight: it is a fixed input that never changes; its weight is θ. Wiring the bias to that input lets θ be trained as one more weight, with the same rule. So the term that enters the sum is θ·(−1) = −θ, and when the neuron fails to fire (e = 1), Δθ = η·e·(−1) is negative: the threshold drops, and can go below zero, as in Charizard’s first step (0.5 → −0.5).'}
            </Note>
            <Note title={pt ? 'Fronteira de decisão' : 'Decision boundary'}>
              {pt
                ? 'É a reta onde net = 0, ou seja, w1·x1 + w2·x2 = θ. De um lado dela o neurônio responde 1, do outro responde 0. Treinar é mover essa reta.'
                : 'It is the line where net = 0, that is, w1·x1 + w2·x2 = θ. On one side the neuron answers 1, on the other 0. Training means moving this line.'}
            </Note>
          </article>
          <article className="panel">
            <h3>{pt ? 'Convenções e regras' : 'Conventions and rules'}</h3>
            <ul className="arrows rules">
              <li>{pt ? 'Entradas' : 'Inputs'}: <span className="var">x1</span> = {pt ? 'Velocidade' : 'Speed'}, <span className="var">x2</span> = {pt ? 'Defesa' : 'Defense'}, <span className="var">x0 = −1</span> ({pt ? 'entrada fixa do bias' : 'fixed input for the bias'})</li>
              <li>{pt ? 'Pesos iniciais' : 'Initial weights'}: <span className="var">w1 = 0</span>, <span className="var">w2 = 0</span>, <span className="var">θ = {num(0.5, lang)}</span></li>
              <li>{pt ? 'Taxa de aprendizado' : 'Learning rate'}: <span className="var">η = 1</span> {pt ? '(para simplificar as contas)' : '(keeps the arithmetic simple)'}</li>
              <li>{pt ? 'Função degrau' : 'Step function'}: net ≥ 0 ⟹ y = 1 ({pt ? 'Fogo' : 'Fire'}); net &lt; 0 ⟹ y = 0 ({pt ? 'Água' : 'Water'})</li>
              <li>{pt ? 'Saída desejada' : 'Desired output'}: <span className="var">d</span> = 1 {pt ? 'para Charizard' : 'for Charizard'}, <span className="var">d</span> = 0 {pt ? 'para Blastoise' : 'for Blastoise'}</li>
            </ul>
            <h3 style={{ marginTop: 14 }}>{pt ? 'Regra de aprendizagem do perceptron' : 'Perceptron learning rule'}</h3>
            <div className="formula">{`e  = d − y
Δw1 = η · e · x1
Δw2 = η · e · x2
Δθ  = η · e · (−1)`}</div>
            <p className="small muted" style={{ margin: 0 }}>
              {pt
                ? 'Se e = 0, nada muda. Se o neurônio respondeu 0 e devia ser 1 (e = 1), cada peso ganha a sua entrada; no caso contrário (e = −1), perde. Uma época é uma passada por todos os exemplos.'
                : 'If e = 0, nothing changes. If the neuron answered 0 and should have said 1 (e = 1), each weight gains its input; in the opposite case (e = −1), it loses it. An epoch is one pass over all the examples.'}
            </p>
            <Note title={pt ? 'É backpropagation?' : 'Is this backpropagation?'}>
              {pt
                ? 'Ainda não. Esta é a regra de aprendizagem do perceptron (Rosenblatt, 1958): ela é a ancestral do backpropagation, porque o erro da saída corrige cada peso na proporção da sua entrada. O backpropagation leva a ideia para várias camadas, mas precisa de uma ativação com derivada, e o degrau não tem. Isso fica para a Aula 2.'
                : 'Not yet. This is the perceptron learning rule (Rosenblatt, 1958): it is the ancestor of backpropagation, because the output error corrects each weight in proportion to its input. Backpropagation carries the idea through several layers, but it needs an activation with a derivative, and the step has none. That is lesson 2.'}
            </Note>
          </article>
        </div>
      </section>

      {/* ---------- 2. Training ---------- */}
      <section className="section" id="l1-treino" aria-labelledby="l1-h-treino">
        <div className="section-head">
          <h2 id="l1-h-treino">2 · {pt ? 'Funcionamento e treinamento: passo a passo' : 'How it works and trains: step by step'}</h2>
          {cached && (
            <span className="label">
              <span className={`led ${cached.source === 'cache' ? 'led--ok' : ''}`} />{' '}
              {cached.source === 'cache'
                ? (pt ? `modelo em cache · ${cached.path}` : `cached model · ${cached.path}`)
                : (pt ? 'cache indisponível · treinado no navegador' : 'cache unavailable · trained in the browser')}
            </span>
          )}
        </div>
        {cached ? <Trainer model={cached.model} /> : <div className="panel">{pt ? 'Carregando o modelo…' : 'Loading the model…'}</div>}
      </section>

      {/* ---------- 3. Try ---------- */}
      <section className="section" id="l1-experimente" aria-labelledby="l1-h-try">
        <div className="section-head">
          <h2 id="l1-h-try">3 · {pt ? 'Experimente' : 'Try it yourself'}</h2>
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
        <span className="label">★ {pt ? 'Ponto-chave' : 'Key point'}</span>
        <p style={{ margin: '8px 0 0' }}>
          {pt
            ? 'Repare na época 2: o erro foi zero para os dois Pokémon, sem nenhum ajuste. O perceptron encontrou uma reta que separa o Charizard (muita velocidade, pouca defesa) do Blastoise (muita defesa, pouca velocidade). Com w1 = 22, w2 = −22 e θ = 0,5, a regra que ele aprendeu é simples: é Fogo quando 22 · (Velocidade − Defesa) ≥ 0,5, isto é, quando a Velocidade passa a Defesa. Aprendizado concluído!'
            : 'Look at epoch 2: the error was zero for both Pokémon, with no update. The perceptron found a line that separates Charizard (high speed, low defense) from Blastoise (high defense, low speed). With w1 = 22, w2 = −22 and θ = 0.5, the rule it learned is simple: it is Fire when 22 · (Speed − Defense) ≥ 0.5, that is, when Speed beats Defense. Learning complete!'}
        </p>
        <Note title={pt ? 'Por que os pesos pularam tanto?' : 'Why did the weights jump so much?'}>
          {pt
            ? 'Com η = 1 e entradas na casa de 100, cada correção soma a entrada inteira ao peso: w1 foi de 0 a 100 de uma vez. Na prática, as entradas são normalizadas e η é bem menor; no laboratório abaixo, teste η = 0,1 e 0,01: os pesos ficam 10 e 100 vezes menores, e a reta final continua paralela à de η = 1, separando os dois.'
            : 'With η = 1 and inputs around 100, each correction adds the whole input to the weight: w1 went from 0 to 100 at once. In practice, inputs are normalized and η is much smaller; in the lab below, try η = 0.1 and 0.01: the weights get 10 and 100 times smaller, and the final line stays parallel to the η = 1 one, still separating the two.'}
        </Note>
        <Note title={pt ? 'Sempre converge?' : 'Does it always converge?'}>
          {pt
            ? 'Se existe uma reta que separa as duas classes, a regra do perceptron encontra uma em número finito de passos (teorema de convergência do perceptron, Novikoff, 1962). Se não existe, ela nunca para de errar. É exatamente o que acontece com o tipo Planta na Aula 2.'
            : 'If a line separating the two classes exists, the perceptron rule finds one in a finite number of steps (perceptron convergence theorem, Novikoff, 1962). If none exists, it never stops making mistakes. That is exactly what happens with the Grass type in lesson 2.'}
        </Note>
      </article>
    </>
  );
}
