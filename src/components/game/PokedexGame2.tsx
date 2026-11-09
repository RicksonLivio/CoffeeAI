import { useEffect, useMemo, useRef, useState } from 'react';
import { assets } from '../../api';
import { num, par, useLang, type Lang } from '../../i18n';
import {
  buildGame2, questionCount2, trainingSummary, XP_PER_ANSWER,
  type Ball3, type Encounter2, type HandEncounter, type LearnEncounter, type OrderChoice, type Poke2, type Vec3,
} from '../../lib/game2';
import { HAND_NET } from '../../lib/mlp';
import { pokeName, sumExpr } from '../learn/CalcPanel';
import { ballName, Choices, reducedMotion, typeName, Wild } from './parts';

type Screen = 'intro' | 'play' | 'badge' | 'learn-intro' | 'end';
/**
 * Hand-made brain: h1 → h2 → ball → aim → throw → result.
 * Brain that learns: ball → aim → throw → result → (loss → delta → order → update → learned when it breaks free).
 */
type Stage = 'h1' | 'h2' | 'ball' | 'aim' | 'throw' | 'result' | 'loss' | 'delta' | 'order' | 'update' | 'learned';
type DexState = 'unknown' | 'seen' | 'caught';
type Dex = Record<Poke2, DexState>;

const DEX_ORDER: Poke2[] = ['charizard', 'blastoise', 'venusaur'];
const NEW_DEX: Dex = { charizard: 'unknown', blastoise: 'unknown', venusaur: 'unknown' };
const BALLS: Ball3[] = ['fire', 'water', 'grass'];
const THROW_MS = 2600;
const GRASS = 2;

const vec = (v: number[], lang: Lang) => `(${v.map((x) => num(x, lang)).join(' ; ')})`;

function orderText(o: OrderChoice, pt: boolean): string {
  if (o === 'right') {
    return pt
      ? 'Calcular o erro da camada oculta com os pesos antigos e só depois atualizar os 15 pesos de uma vez'
      : 'Work out the hidden-layer error with the old weights, and only then update all 15 weights at once';
  }
  if (o === 'output-first') {
    return pt
      ? 'Atualizar primeiro os pesos da saída e depois calcular o erro da camada oculta com os pesos novos'
      : 'Update the output weights first, then work out the hidden-layer error with the new weights';
  }
  return pt
    ? 'Atualizar primeiro os pesos da camada oculta e depois calcular de novo o erro da saída'
    : 'Update the hidden-layer weights first, then work out the output error again';
}

function Probs({ p, lang }: { p: Vec3; lang: Lang }) {
  const pt = lang === 'pt';
  return (
    <div className="pd-probs">
      {BALLS.map((t, k) => (
        <div key={t} className={t}>
          <span>{typeName(t, pt)}</span>
          <b>p = {num(p[k], lang)}</b>
          <i style={{ width: `${Math.round(100 * p[k])}%` }} />
        </div>
      ))}
    </div>
  );
}

export default function PokedexGame2({ onClose, onLesson }: { onClose: () => void; onLesson: (n: number) => void }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const game = useMemo(() => buildGame2(), []);
  const totalQ = questionCount2(game);
  const handCount = game.filter((g) => g.kind === 'hand').length;
  const learn = game.filter((g): g is LearnEncounter => g.kind === 'learn');
  const dialog = useRef<HTMLDialogElement>(null);

  const [screen, setScreen] = useState<Screen>('intro');
  const [k, setK] = useState(0);
  const [stage, setStage] = useState<Stage>('h1');
  const [wrong, setWrong] = useState<string[]>([]);
  const [xp, setXp] = useState(0);
  const [firstTry, setFirstTry] = useState(0);
  const [dex, setDex] = useState<Dex>(NEW_DEX);
  const [catches, setCatches] = useState(0);

  const c: Encounter2 = game[k];
  const summary = useMemo(() => (screen === 'end' ? trainingSummary() : null), [screen]);

  useEffect(() => {
    const d = dialog.current;
    if (d && !d.open) d.showModal();
    return () => d?.close();
  }, []);

  // The Ball flies, shakes three times, then the result shows.
  useEffect(() => {
    if (stage !== 'throw') return;
    const t = window.setTimeout(() => {
      setDex((cur) => ({ ...cur, [c.pokemon]: c.caught ? 'caught' : cur[c.pokemon] === 'caught' ? 'caught' : 'seen' }));
      if (c.caught) setCatches((n) => n + 1);
      setStage('result');
    }, reducedMotion() ? 300 : THROW_MS);
    return () => window.clearTimeout(t);
  }, [stage, c]);

  function answer(right: boolean, key: string, next: Stage) {
    if (!right) {
      setWrong((w) => [...w, key]);
      return;
    }
    if (wrong.length === 0) {
      setXp((x) => x + XP_PER_ANSWER);
      setFirstTry((f) => f + 1);
    }
    setWrong([]);
    setStage(next);
  }

  function goTo(i: number) {
    setK(i);
    setStage(game[i].kind === 'hand' ? 'h1' : 'ball');
    setWrong([]);
    setScreen('play');
  }

  /** After an encounter: next one, the badge between the two brains, or the end. */
  function advance() {
    const nx = game[k + 1];
    if (!nx) return setScreen('end');
    if (nx.kind !== c.kind) return setScreen('badge');
    goTo(k + 1);
  }

  function restart() {
    setXp(0);
    setFirstTry(0);
    setDex(NEW_DEX);
    setCatches(0);
    setK(0);
    setStage('h1');
    setWrong([]);
    setScreen('intro');
  }

  const revealed = dex[c.pokemon] !== 'unknown' || c.kind === 'learn';
  const rank = firstTry === totalQ
    ? (pt ? 'Mestre do Backpropagation' : 'Backpropagation Master')
    : firstTry >= totalQ - 4
      ? (pt ? 'Treinador de Ginásio' : 'Gym Trainer')
      : (pt ? 'Treinador Novato' : 'Rookie Trainer');

  const hud = c.kind === 'hand'
    ? `${pt ? 'ROTA 2 · REDE À MÃO · ENCONTRO' : 'ROUTE 2 · HAND-MADE NETWORK · ENCOUNTER'} ${c.n}/${handCount}`
    : `${pt ? 'ROTA 2 · REDE QUE APRENDE · TENTATIVA' : 'ROUTE 2 · LEARNING NETWORK · ATTEMPT'} ${c.attempt}/2`;

  const before = stage === 'h1' || stage === 'h2' || stage === 'ball' || stage === 'aim';
  const field = (() => {
    if (stage === 'throw') {
      return (
        <>
          <Wild id={c.pokemon} revealed={revealed} className="absorb" />
          <span className="pd-ball-path throw"><span className={`pd-ball ${c.ball} wobble`} /></span>
        </>
      );
    }
    if (!before && c.caught) {
      return (
        <>
          <Wild id={c.pokemon} revealed className="gone" />
          <span className="pd-ball-path"><span className={`pd-ball ${c.ball}`} /></span>
          <span className="pd-stars" aria-hidden="true"><i>★</i><i>★</i><i>★</i></span>
        </>
      );
    }
    return <Wild id={c.pokemon} revealed={revealed} className={stage === 'result' ? 'pop' : ''} />;
  })();

  const appear = !before && stage !== 'throw'
    ? (c.caught
      ? (pt ? `Gotcha! ${pokeName(c.pokemon)} foi capturado!` : `Gotcha! ${pokeName(c.pokemon)} was caught!`)
      : (pt ? `Ah não! ${pokeName(c.pokemon)} escapou da Ball!` : `Oh no! ${pokeName(c.pokemon)} broke free!`))
    : c.kind === 'learn' && c.attempt === 2
      ? (pt ? 'O Venusaur voltou!' : 'Venusaur is back!')
      : c.kind === 'learn'
        ? (pt ? 'Outro Venusaur selvagem apareceu!' : 'Another wild Venusaur appeared!')
        : (pt ? `Um ${revealed ? pokeName(c.pokemon) : '???'} selvagem apareceu!` : `A wild ${revealed ? pokeName(c.pokemon) : '???'} appeared!`);

  return (
    <dialog ref={dialog} className="pd-dialog" aria-labelledby="pd2-title"
      onCancel={(e) => { e.preventDefault(); onClose(); }}
      onClick={(e) => { if (e.target === dialog.current) onClose(); }}>
      <div className="pd">
        <div className="pd-top">
          <span className={`pd-lens ${stage === 'throw' ? 'busy' : ''}`} aria-hidden="true" />
          <span className="pd-led r" aria-hidden="true" />
          <span className="pd-led y" aria-hidden="true" />
          <span className="pd-led g" aria-hidden="true" />
          <h2 id="pd2-title" className="pd-title">Pokédex · {pt ? 'fase 2' : 'phase 2'}</h2>
          <button className="pd-close" onClick={onClose} aria-label={pt ? 'Fechar o Pokédex' : 'Close the Pokédex'}>✕</button>
        </div>

        <div className="pd-screen" aria-live="polite">
          {screen === 'intro' && (
            <div className="pd-page">
              <div className="pd-prof">
                <img src={assets.avatar} alt="" width={56} height={56} />
                <div>
                  <span className="pd-kicker">{pt ? 'Mensagem do Professor Rick' : 'Message from Professor Rick'}</span>
                  <p>
                    {pt
                      ? 'Bem-vindo à Rota 2! Lembra do Venusaur que nenhuma Ball segurava? Atualizei o seu Pokédex com a rede da Aula 2: uma camada oculta com dois detectores e três saídas, uma para cada Ball. Agora existe a Grass Ball.'
                      : 'Welcome to Route 2! Remember the Venusaur no Ball could hold? I upgraded your Pokédex with the network of Lesson 2: a hidden layer with two detectors and three outputs, one per Ball. Now there is a Grass Ball.'}
                  </p>
                </div>
              </div>
              <div className="pd-brain">
                h1: net = x1 − x2 − 11 · h2: net = x2 − x1 − 11<br />
                {pt ? 'Fogo = h1 · Água = h2 · Planta = nem h1 nem h2' : 'Fire = h1 · Water = h2 · Grass = neither h1 nor h2'}
              </div>
              <p>
                {pt
                  ? 'Cada detector é um perceptron da Aula 1, com degrau: vale 1 se net ≥ 0. Primeiro você usa os pesos que eu escolhi à mão. Depois, o Pokédex vai ter que aprender os pesos sozinho, com backpropagation.'
                  : 'Each detector is a Lesson 1 perceptron, with a step: it is 1 if net ≥ 0. First you use the weights I picked by hand. Then the Pokédex will have to learn its weights on its own, with backpropagation.'}
              </p>
              <p className="pd-small">
                {pt
                  ? `Acerte de primeira para ganhar ${XP_PER_ANSWER} XP por resposta. Errou? O Pokédex dá uma dica e você tenta de novo.`
                  : `Answer right on the first try to earn ${XP_PER_ANSWER} XP per answer. Got it wrong? The Pokédex gives a hint and you try again.`}
              </p>
              <div className="pd-actions">
                <button className="btn btn--gold" onClick={() => goTo(0)}>{pt ? '▶ Entrar no mato alto' : '▶ Walk into the tall grass'}</button>
              </div>
            </div>
          )}

          {screen === 'play' && (
            <div className="pd-page">
              <div className="pd-hud"><span>{hud}</span><span>XP {xp}</span></div>
              <div className="pd-field">{field}</div>
              <p className="pd-appear">{appear}</p>
              {c.kind === 'hand'
                ? <HandPlay c={c} stage={stage} wrong={wrong} lang={lang} answer={answer} setStage={setStage} advance={advance} />
                : <LearnPlay c={c} stage={stage} wrong={wrong} lang={lang} answer={answer} setStage={setStage} advance={advance} />}
            </div>
          )}

          {screen === 'badge' && (
            <div className="pd-page pd-center">
              <span className="pd-badge" aria-hidden="true">h</span>
              <span className="pd-kicker">{pt ? 'Insígnia da Camada Oculta' : 'Hidden Layer Badge'}</span>
              <h3>{pt ? 'Três tipos, três Balls, três capturas' : 'Three types, three Balls, three catches'}</h3>
              <p>
                {pt
                  ? 'A camada oculta transformou cada Pokémon no par (h1, h2): Fogo em (1, 0), Água em (0, 1) e Planta em (0, 0). Com três cantos, retas simples separam tudo, e o Venusaur coube na Grass Ball.'
                  : 'The hidden layer turned each Pokémon into the pair (h1, h2): Fire at (1, 0), Water at (0, 1) and Grass at (0, 0). With three corners, simple lines separate everything, and Venusaur fit in the Grass Ball.'}
              </p>
              <p className="pd-small">
                {pt ? 'Mas fui eu que escolhi esses pesos. Uma rede de verdade precisa encontrá-los sozinha.' : 'But I picked those weights. A real network has to find them on its own.'}
              </p>
              <div className="pd-actions pd-center">
                <button className="btn btn--gold" onClick={() => setScreen('learn-intro')}>{pt ? 'O Professor Rick está chamando…' : 'Professor Rick is calling…'}</button>
              </div>
            </div>
          )}

          {screen === 'learn-intro' && (
            <div className="pd-page">
              <div className="pd-prof">
                <img src={assets.avatar} alt="" width={56} height={56} />
                <div>
                  <span className="pd-kicker">{pt ? 'Mensagem do Professor Rick' : 'Message from Professor Rick'}</span>
                  <p>
                    {pt
                      ? 'Troquei o cérebro do Pokédex por uma rede que aprende: sigmoide na camada oculta, softmax na saída e 15 pesos sorteados, os mesmos da Aula 2. As entradas entram divididas por 100. O Pokédex carrega a Ball do tipo com a maior probabilidade.'
                      : 'I swapped the Pokédex brain for a network that learns: sigmoid in the hidden layer, softmax at the output and 15 random weights, the same as in Lesson 2. The inputs go in divided by 100. The Pokédex loads the Ball of the type with the highest probability.'}
                  </p>
                </div>
              </div>
              <p>
                {pt
                  ? 'Quando o Pokémon escapar, você faz um passo de backpropagation: mede o erro com a entropia cruzada, calcula o erro na saída, leva o erro para trás na ordem certa e atualiza um peso.'
                  : 'When the Pokémon breaks free, you do one backpropagation step: measure the error with cross-entropy, work out the output error, carry the error back in the right order and update one weight.'}
              </p>
              <div className="pd-actions">
                <button className="btn btn--gold" onClick={() => goTo(k + 1)}>{pt ? '▶ Voltar ao mato' : '▶ Back to the grass'}</button>
              </div>
            </div>
          )}

          {screen === 'end' && summary && (
            <div className="pd-page">
              <div className="pd-center">
                <span className="pd-badge" aria-hidden="true">∇</span>
                <span className="pd-kicker">{pt ? 'Insígnia do Backpropagation' : 'Backpropagation Badge'}</span>
              </div>
              <div className="pd-entry">
                <Wild id="venusaur" revealed />
                <div>
                  <span className="pd-kicker">{pt ? 'Registro nº 003 · capturado' : 'Entry #003 · caught'}</span>
                  <h3>Venusaur</h3>
                  <p className="pd-small">
                    {pt
                      ? `Velocidade 80, Defesa 83. Tipo Planta. Na fase 1 nenhuma Ball servia; agora a rede à mão e a rede que aprende acertaram. Um único passo de backpropagation levou Planta de ${Math.round(100 * learn[0].p[GRASS])}% para ${Math.round(100 * learn[1].p[GRASS])}%.`
                      : `Speed 80, Defense 83. Grass type. In phase 1 no Ball fit; now both the hand-made network and the learning one got it right. A single backpropagation step took Grass from ${Math.round(100 * learn[0].p[GRASS])}% to ${Math.round(100 * learn[1].p[GRASS])}%.`}
                  </p>
                </div>
              </div>
              <div className="pd-score">
                <div><span>XP</span><b>{xp} / {totalQ * XP_PER_ANSWER}</b></div>
                <div><span>{pt ? 'De primeira' : 'First try'}</span><b>{firstTry} / {totalQ}</b></div>
                <div><span>{pt ? 'Capturas' : 'Catches'}</span><b>{catches} / {game.length}</b></div>
              </div>
              <p className="pd-center"><b className="gold">{rank}</b></p>
              <p className="pd-small">
                {pt
                  ? `Um passo só com o Venusaur é pouco: a rede poderia esquecer os outros. No treino da Aula 2, cada época usa os nove iniciais. Com η = 2, ela acerta os nove pela primeira vez na época ${summary.firstAllRight} e, depois de ${summary.epochs} épocas, o erro médio é ${num(Number(summary.finalLoss.toFixed(4)), lang)} e o Venusaur recebe ${Math.round(100 * summary.venusaurGrass)}% de Planta.`
                  : `One step with Venusaur alone is not much: the network could forget the others. In the Lesson 2 training run, each epoch uses all nine starters. With η = 2, it first gets all nine right at epoch ${summary.firstAllRight} and, after ${summary.epochs} epochs, the average error is ${num(Number(summary.finalLoss.toFixed(4)), lang)} and Venusaur gets ${Math.round(100 * summary.venusaurGrass)}% Grass.`}
              </p>
              <div className="pd-actions">
                <button className="btn btn--ghost btn--sm" onClick={restart}>↺ {pt ? 'Jogar de novo' : 'Play again'}</button>
                <button className="btn btn--gold btn--sm" onClick={() => { onClose(); onLesson(2); }}>{pt ? 'Abrir a Aula 2' : 'Open Lesson 2'}</button>
              </div>
            </div>
          )}
        </div>

        <div className="pd-dex" aria-label={pt ? 'Registros do Pokédex' : 'Pokédex entries'}>
          {DEX_ORDER.map((p, i) => (
            <div key={p} className={`pd-slot ${dex[p]}`}>
              <span className="pd-slot-n">{String(i + 1).padStart(3, '0')}</span>
              {dex[p] === 'unknown'
                ? <span className="pd-slot-img unknown">?</span>
                : <img className="pd-slot-img" src={assets.card(p)} alt="" />}
              <span className="pd-slot-name">{dex[p] === 'unknown' ? '???' : pokeName(p)}</span>
              <span className="pd-slot-state">
                {dex[p] === 'caught' ? (pt ? '● capturado' : '● caught') : dex[p] === 'seen' ? (pt ? '○ visto' : '○ seen') : '—'}
              </span>
            </div>
          ))}
        </div>
      </div>
    </dialog>
  );
}

interface PlayProps<E> {
  c: E;
  stage: Stage;
  wrong: string[];
  lang: Lang;
  answer: (right: boolean, key: string, next: Stage) => void;
  setStage: (s: Stage) => void;
  advance: () => void;
}

/** Part A: the hand-made step network. */
function HandPlay({ c, stage, wrong, lang, answer, setStage, advance }: PlayProps<HandEncounter>) {
  const pt = lang === 'pt';
  const [n1, n2] = HAND_NET.hidden;
  const outNames = BALLS.map((t) => typeName(t, pt));
  const prods = (w: number[], theta: number) => [w[0] * c.x1, w[1] * c.x2, -theta];
  const hLine = (j: 0 | 1) => {
    const n = j === 0 ? n1 : n2;
    return `net_h${j + 1} = ${sumExpr(prods(n.w, n.theta), lang)} = ${num(c.hiddenNet[j], lang)} ${c.hiddenNet[j] >= 0 ? '≥' : '<'} 0 ⟹ h${j + 1} = ${c.h[j]}`;
  };
  const outLine = () => HAND_NET.output.map((_, q) =>
    `${outNames[q]}: ${num(c.outputNet[q], lang)} ⟹ ${c.outputNet[q] >= 0 ? 1 : 0}`).join(' · ');
  const hint = (j: 0 | 1) => {
    const n = j === 0 ? n1 : n2;
    return (
      <p className="pd-hint">
        {pt ? 'Dica: x0 = −1, então o bias entra como −θ = −11. ' : 'Hint: x0 = −1, so the bias enters as −θ = −11. '}
        <span className="mono">({par(n.w[0], lang)} · {c.x1}) + ({par(n.w[1], lang)} · {c.x2}) + ({num(n.theta, lang)} · (−1))</span>
      </p>
    );
  };
  const want = typeName(c.type, pt);

  return (
    <>
      <div className="pd-scan">
        <div><span>{pt ? 'Velocidade' : 'Speed'}</span><b>x1 = {c.x1}</b></div>
        <div><span>{pt ? 'Defesa' : 'Defense'}</span><b>x2 = {c.x2}</b></div>
        <div><span>bias</span><b>x0 = −1</b></div>
      </div>
      <div className="pd-brain">
        h1: w = (1, −1), θ = 11 · h2: w = (−1, 1), θ = 11
      </div>

      {stage === 'h1' && (
        <div className="pd-q">
          <p><b>1 ·</b> {pt ? 'O detector de Fogo, h1. Calcule o net dele:' : 'The Fire detector, h1. Work out its net:'}{' '}
            <span className="mono">net_h1 = 1·x1 + (−1)·x2 + 11·(−1)</span></p>
          <Choices options={c.h1Options} keyOf={String} label={(v) => `net_h1 = ${num(v, lang)}`}
            isRight={(v) => v === c.hiddenNet[0]} wrong={wrong} onPick={(r, key) => answer(r, key, 'h2')} />
          {wrong.length > 0 && hint(0)}
        </div>
      )}

      {stage === 'h2' && (
        <div className="pd-q">
          <p className="mono pd-done">{hLine(0)}</p>
          <p><b>2 ·</b> {pt ? 'Agora o detector de Água, h2:' : 'Now the Water detector, h2:'}{' '}
            <span className="mono">net_h2 = (−1)·x1 + 1·x2 + 11·(−1)</span></p>
          <Choices options={c.h2Options} keyOf={String} label={(v) => `net_h2 = ${num(v, lang)}`}
            isRight={(v) => v === c.hiddenNet[1]} wrong={wrong} onPick={(r, key) => answer(r, key, 'ball')} />
          {wrong.length > 0 && hint(1)}
        </div>
      )}

      {stage === 'ball' && (
        <div className="pd-q">
          <p className="mono pd-done">{hLine(0)}<br />{hLine(1)}</p>
          <p><b>3 ·</b> {pt
            ? `Camada de saída, com h1 = ${c.h[0]} e h2 = ${c.h[1]}. Fogo = h1, Água = h2, Planta = nem h1 nem h2. Qual Ball o Pokédex carrega?`
            : `Output layer, with h1 = ${c.h[0]} and h2 = ${c.h[1]}. Fire = h1, Water = h2, Grass = neither h1 nor h2. Which Ball does the Pokédex load?`}</p>
          <Choices options={BALLS} keyOf={String} label={(b) => ballName(b)}
            isRight={(b) => b === c.ball} wrong={wrong} onPick={(r, key) => answer(r, key, 'aim')} />
          {wrong.length > 0 && (
            <p className="pd-hint">
              {pt ? 'Dica: cada saída é um perceptron que acende com net ≥ 0. ' : 'Hint: each output is a perceptron that lights up when net ≥ 0. '}
              <span className="mono">{outNames[0]}: h1 − {num(0.5, lang)} · {outNames[1]}: h2 − {num(0.5, lang)} · {outNames[2]}: {num(0.5, lang)} − h1 − h2</span>
            </p>
          )}
        </div>
      )}

      {stage === 'aim' && (
        <div className="pd-q">
          <p className="mono pd-done">{outLine()} ⟹ {ballName(c.ball)}</p>
          <div className="pd-actions">
            <button className="btn btn--gold" onClick={() => setStage('throw')}>
              {pt ? `Lançar a ${ballName(c.ball)}!` : `Throw the ${ballName(c.ball)}!`}
            </button>
          </div>
        </div>
      )}

      {stage === 'throw' && <p className="pd-small pd-center">{pt ? '…balança… balança… balança…' : '…wobble… wobble… wobble…'}</p>}

      {stage === 'result' && (
        <div className="pd-q">
          <p>
            {c.type === 'grass'
              ? (pt
                ? `Nenhum detector acendeu: h1 = 0 e h2 = 0. Na fase 1, o perceptron só conhecia Fogo e Água; agora a saída Planta, 0,5 − 0 − 0 = 0,5, acendeu. O Venusaur é seu!`
                : `Neither detector lit up: h1 = 0 and h2 = 0. In phase 1, the perceptron only knew Fire and Water; now the Grass output, 0.5 − 0 − 0 = 0.5, lit up. Venusaur is yours!`)
              : (pt
                ? `A rede previu ${want}, o tipo real. Os pesos foram escolhidos à mão, então nada muda.`
                : `The network predicted ${want}, the real type. The weights were picked by hand, so nothing changes.`)}
          </p>
          <div className="pd-actions">
            <button className="btn btn--gold" onClick={advance}>{pt ? 'Continuar' : 'Continue'}</button>
          </div>
        </div>
      )}
    </>
  );
}

/** Part B: the sigmoid/softmax network and one backpropagation step. */
function LearnPlay({ c, stage, wrong, lang, answer, setStage, advance }: PlayProps<LearnEncounter>) {
  const pt = lang === 'pt';
  const t = c.train;
  const grass = typeName('grass', pt);

  return (
    <>
      <div className="pd-scan">
        <div><span>{pt ? 'Velocidade' : 'Speed'}</span><b>x1 = {num(c.x1 / 100, lang)}</b></div>
        <div><span>{pt ? 'Defesa' : 'Defense'}</span><b>x2 = {num(c.x2 / 100, lang)}</b></div>
        <div><span>{pt ? 'oculta' : 'hidden'}</span><b>h = {vec(c.h, lang)}</b></div>
      </div>
      <div className={`pd-brain ${stage === 'learned' ? 'changed' : ''}`}>
        {c.attempt === 1
          ? (pt ? 'Cérebro: 15 pesos sorteados (semente fixa) · sigmoide · softmax' : 'Brain: 15 random weights (fixed seed) · sigmoid · softmax')
          : (pt ? 'Cérebro: 15 pesos depois de 1 passo de backpropagation' : 'Brain: 15 weights after 1 backpropagation step')}
      </div>
      <Probs p={c.p} lang={lang} />

      {stage === 'ball' && (
        <div className="pd-q">
          <p><b>{c.attempt === 1 ? '1' : '6'} ·</b> {pt
            ? 'A softmax deu uma probabilidade para cada tipo, e elas somam 1. O Pokédex carrega a Ball do tipo mais provável. Qual?'
            : 'Softmax gave each type a probability, and they add up to 1. The Pokédex loads the Ball of the most likely type. Which one?'}</p>
          <Choices options={BALLS} keyOf={String} label={(b) => ballName(b)}
            isRight={(b) => b === c.ball} wrong={wrong} onPick={(r, key) => answer(r, key, 'aim')} />
          {wrong.length > 0 && (
            <p className="pd-hint">
              {pt
                ? 'Dica: o Pokédex segue a rede, não o que você sabe sobre o Pokémon. Escolha o maior p.'
                : 'Hint: the Pokédex follows the network, not what you know about the Pokémon. Pick the largest p.'}
            </p>
          )}
        </div>
      )}

      {stage === 'aim' && (
        <div className="pd-q">
          <p className="mono pd-done">max p = {num(Math.max(...c.p), lang)} ⟹ {ballName(c.ball)}</p>
          <div className="pd-actions">
            <button className="btn btn--gold" onClick={() => setStage('throw')}>
              {pt ? `Lançar a ${ballName(c.ball)}!` : `Throw the ${ballName(c.ball)}!`}
            </button>
          </div>
        </div>
      )}

      {stage === 'throw' && <p className="pd-small pd-center">{pt ? '…balança… balança… balança…' : '…wobble… wobble… wobble…'}</p>}

      {stage === 'result' && (
        <div className="pd-q">
          {c.caught ? (
            <p>
              {pt
                ? `Agora Planta tem p = ${num(c.p[GRASS], lang)}. Um passo de backpropagation bastou para este Venusaur.`
                : `Grass now has p = ${num(c.p[GRASS], lang)}. One backpropagation step was enough for this Venusaur.`}
            </p>
          ) : (
            <p>
              {pt
                ? `O Venusaur é ${grass}: d = (0, 0, 1). A rede deu só ${num(c.p[GRASS], lang)} para ${grass} e carregou a ${ballName(c.ball)}, igual ao perceptron da fase 1. Hora de treinar.`
                : `Venusaur is ${grass}: d = (0, 0, 1). The network gave ${grass} only ${num(c.p[GRASS], lang)} and loaded the ${ballName(c.ball)}, just like the phase 1 perceptron. Time to train.`}
            </p>
          )}
          <div className="pd-actions">
            {c.caught
              ? <button className="btn btn--gold" onClick={advance}>{pt ? 'Ver o registro no Pokédex' : 'See the Pokédex entry'}</button>
              : <button className="btn btn--gold" onClick={() => setStage('loss')}>{pt ? 'Treinar com backpropagation' : 'Train with backpropagation'}</button>}
          </div>
        </div>
      )}

      {stage === 'loss' && t && (
        <div className="pd-q">
          <p><b>2 ·</b> {pt ? 'Entropia cruzada: ' : 'Cross-entropy: '}<span className="mono">L = −ln p_{grass} = −ln {num(c.p[GRASS], lang)}</span>. {pt ? 'Quanto vale a loss?' : 'What is the loss?'}</p>
          <Choices options={t.lossOptions} keyOf={String} label={(v) => `L = ${num(v, lang)}`}
            isRight={(v) => v === t.loss} wrong={wrong} onPick={(r, key) => answer(r, key, 'delta')} />
          {wrong.length > 0 && (
            <p className="pd-hint">
              {pt
                ? `Dica: ln ${num(c.p[GRASS], lang)} ≈ ${num(-t.loss, lang)}, e o sinal de menos deixa a loss positiva. A loss não é a própria probabilidade.`
                : `Hint: ln ${num(c.p[GRASS], lang)} ≈ ${num(-t.loss, lang)}, and the minus sign makes the loss positive. The loss is not the probability itself.`}
            </p>
          )}
        </div>
      )}

      {stage === 'delta' && t && (
        <div className="pd-q">
          <p className="mono pd-done">L = −ln {num(c.p[GRASS], lang)} = {num(t.loss, lang)}</p>
          <p><b>3 ·</b> {pt ? 'Erro na saída: ' : 'Output error: '}<span className="mono">δo = p − d</span>, {pt ? 'com' : 'with'} p = {vec(c.p, lang)} {pt ? 'e' : 'and'} d = (0 ; 0 ; 1). {pt ? 'Quanto vale δo?' : 'What is δo?'}</p>
          <Choices options={t.deltaOptions} keyOf={(v) => v.join('|')} label={(v) => `δo = ${vec(v, lang)}`}
            isRight={(v) => v.join('|') === t.delta.join('|')} wrong={wrong} onPick={(r, key) => answer(r, key, 'order')} />
          {wrong.length > 0 && (
            <p className="pd-hint">
              {pt ? 'Dica: o obtido menos o desejado, posição por posição. Para Planta: ' : 'Hint: what you got minus what you wanted, one position at a time. For Grass: '}
              <span className="mono">{num(c.p[GRASS], lang)} − 1</span>.
            </p>
          )}
        </div>
      )}

      {stage === 'order' && t && (
        <div className="pd-q">
          <p className="mono pd-done">δo = {vec(t.delta, lang)}</p>
          <p><b>4 ·</b> {pt
            ? 'O erro agora volta para a camada oculta: δh = (Σ v·δo)·h(1 − h). Em que ordem o backpropagation faz as contas?'
            : 'The error now goes back to the hidden layer: δh = (Σ v·δo)·h(1 − h). In what order does backpropagation do the work?'}</p>
          <Choices options={t.orderOptions} keyOf={String} label={(o) => orderText(o, pt)} wide
            isRight={(o) => o === 'right'} wrong={wrong} onPick={(r, key) => answer(r, key, 'update')} />
          {wrong.length > 0 && (
            <p className="pd-hint">
              {pt
                ? 'Dica: o erro da camada oculta usa os pesos v que geraram a previsão. Se eles mudassem antes, o gradiente já não corresponderia ao erro que a rede cometeu.'
                : 'Hint: the hidden-layer error uses the weights v that produced the prediction. If they changed first, the gradient would no longer match the error the network made.'}
            </p>
          )}
        </div>
      )}

      {stage === 'update' && t && (
        <div className="pd-q">
          <p className="mono pd-done">δh = {vec(t.deltaHidden, lang)} · {pt ? 'pesos antigos' : 'old weights'}</p>
          <p><b>5 ·</b> {pt ? 'Atualize o bias da saída Planta: ' : 'Update the bias of the Grass output: '}
            <span className="mono">φ ← φ − η·∂L/∂φ</span>, {pt ? 'com' : 'with'} φ = {num(t.phi, lang)}, η = {num(t.eta, lang)} {pt ? 'e' : 'and'}{' '}
            <span className="mono">∂L/∂φ = δo_{grass}·(−1) = {num(t.gradPhi, lang)}</span>. {pt ? 'Qual é o novo φ?' : 'What is the new φ?'}</p>
          <Choices options={t.phiOptions} keyOf={String} label={(v) => `φ = ${num(v, lang)}`}
            isRight={(v) => v === t.phiAfter} wrong={wrong} onPick={(r, key) => answer(r, key, 'learned')} />
          {wrong.length > 0 && (
            <p className="pd-hint">
              {pt ? 'Dica: multiplique por η e subtraia, porque o gradiente aponta morro acima. ' : 'Hint: multiply by η and subtract, because the gradient points uphill. '}
              <span className="mono">{num(t.phi, lang)} − {num(t.eta, lang)}·{num(t.gradPhi, lang)}</span>
            </p>
          )}
        </div>
      )}

      {stage === 'learned' && t && (
        <div className="pd-q">
          <p>
            {pt ? 'Pokédex treinado! ' : 'Pokédex trained! '}
            <span className="mono">φ_{grass}: {num(t.phi, lang)} → {num(t.phiAfter, lang)}</span>
          </p>
          <p className="pd-small">
            {pt
              ? `Os outros 14 pesos andaram contra o gradiente no mesmo passo. Para o Venusaur, Planta vai de ${Math.round(100 * c.p[GRASS])}% para ${t.grassAfter}%, e a loss cai de ${num(t.loss, lang)} para ${num(t.lossAfter, lang)}.`
              : `The other 14 weights moved against the gradient in the same step. For Venusaur, Grass goes from ${Math.round(100 * c.p[GRASS])}% to ${t.grassAfter}%, and the loss drops from ${num(t.loss, lang)} to ${num(t.lossAfter, lang)}.`}
          </p>
          <div className="pd-actions">
            <button className="btn btn--gold" onClick={advance}>{pt ? 'Voltar ao mato' : 'Back to the grass'}</button>
          </div>
        </div>
      )}
    </>
  );
}
