import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { assets } from '../../api';
import { num, par, useLang, type Lang } from '../../i18n';
import {
  buildGame, ERROR_OPTIONS, questionCount, XP_PER_ANSWER, type Ball, type Encounter, type PokeId,
} from '../../lib/game';
import { LESSON_ETA, type Weights } from '../../lib/perceptron';
import { pokeName, sumExpr } from '../learn/CalcPanel';

type Screen = 'intro' | 'play' | 'epoch' | 'badge' | 'end';
/** net → ball → aim → throw → result → (error → update → learned when it breaks free) */
type Stage = 'net' | 'ball' | 'aim' | 'throw' | 'result' | 'error' | 'update' | 'learned';
type DexState = 'unknown' | 'seen' | 'caught';
type Dex = Record<PokeId, DexState>;

const DEX_ORDER: PokeId[] = ['charizard', 'blastoise', 'venusaur'];
const NEW_DEX: Dex = { charizard: 'unknown', blastoise: 'unknown', venusaur: 'unknown' };
const THROW_MS = 2600;

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function weights(w: Weights, lang: Lang): string {
  return `w1 = ${num(w.w1, lang)}, w2 = ${num(w.w2, lang)}, θ = ${num(w.theta, lang)}`;
}

function typeName(t: 'fire' | 'water' | 'grass', pt: boolean): string {
  if (t === 'fire') return pt ? 'Fogo' : 'Fire';
  if (t === 'water') return pt ? 'Água' : 'Water';
  return pt ? 'Planta' : 'Grass';
}

/** Same name in both languages, like the game's item names. */
function ballName(b: Ball): string {
  return b === 'fire' ? 'Fire Ball 🔥' : 'Water Ball 💧';
}

interface ChoicesProps<T> {
  options: T[];
  keyOf: (o: T) => string;
  label: (o: T) => ReactNode;
  isRight: (o: T) => boolean;
  wrong: string[];
  onPick: (right: boolean, key: string) => void;
}

function Choices<T>({ options, keyOf, label, isRight, wrong, onPick }: ChoicesProps<T>) {
  return (
    <div className="pd-choices">
      {options.map((o) => {
        const k = keyOf(o);
        const bad = wrong.includes(k);
        return (
          <button key={k} className={`pd-choice ${bad ? 'wrong' : ''}`} disabled={bad} onClick={() => onPick(isRight(o), k)}>
            {label(o)}
          </button>
        );
      })}
    </div>
  );
}

/** Small Pokémon portrait cropped from the card art; dark until the Pokédex has seen it. */
function Wild({ id, revealed, className = '' }: { id: PokeId; revealed: boolean; className?: string }) {
  return (
    <div className={`pd-wild ${revealed ? id : 'hidden'} ${className}`}>
      <img src={assets.card(id)} alt={revealed ? pokeName(id) : '???'} draggable={false} />
      {!revealed && <span className="pd-qmark" aria-hidden="true">?</span>}
    </div>
  );
}

export default function PokedexGame({ onClose, onLesson }: { onClose: () => void; onLesson: (n: number) => void }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const game = useMemo(() => buildGame(), []);
  const totalQ = questionCount(game);
  const dialog = useRef<HTMLDialogElement>(null);

  const [screen, setScreen] = useState<Screen>('intro');
  const [k, setK] = useState(0);
  const [stage, setStage] = useState<Stage>('net');
  const [wrong, setWrong] = useState<string[]>([]);
  const [xp, setXp] = useState(0);
  const [firstTry, setFirstTry] = useState(0);
  const [dex, setDex] = useState<Dex>(NEW_DEX);

  const c: Encounter = game[k];

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
    setStage('net');
    setWrong([]);
    setScreen('play');
  }

  /** After an encounter: next one, or the end-of-epoch / badge screen when the epoch changes. */
  function advance() {
    const nx = game[k + 1];
    if (!nx) return setScreen('end');
    if (nx.epoch === null) return setScreen('badge');
    if (nx.epoch !== c.epoch) return setScreen('epoch');
    goTo(k + 1);
  }

  function restart() {
    setXp(0);
    setFirstTry(0);
    setDex(NEW_DEX);
    setK(0);
    setStage('net');
    setWrong([]);
    setScreen('intro');
  }

  const revealed = dex[c.pokemon] !== 'unknown';
  const shownW = stage === 'learned' ? c.after : c.before;
  const epochSize = game.filter((g) => g.epoch === 1).length;
  const caughtCount = DEX_ORDER.filter((p) => dex[p] === 'caught').length;
  const rank = firstTry === totalQ
    ? (pt ? 'Mestre do Perceptron' : 'Perceptron Master')
    : firstTry >= totalQ - 4
      ? (pt ? 'Treinador de Ginásio' : 'Gym Trainer')
      : (pt ? 'Treinador Novato' : 'Rookie Trainer');

  const hud = c.epoch === null
    ? (pt ? 'ROTA 1 · BÔNUS · DEPOIS DO TREINO' : 'ROUTE 1 · BONUS · AFTER TRAINING')
    : `${pt ? 'ROTA 1 · ÉPOCA' : 'ROUTE 1 · EPOCH'} ${c.epoch} · ${pt ? 'ENCONTRO' : 'ENCOUNTER'} ${((c.n - 1) % epochSize) + 1}/${epochSize}`;

  const field = (() => {
    if (stage === 'throw') {
      return (
        <>
          <Wild id={c.pokemon} revealed={revealed} className="absorb" />
          <span className="pd-ball-path throw"><span className={`pd-ball ${c.ball} wobble`} /></span>
        </>
      );
    }
    if (stage !== 'net' && stage !== 'ball' && stage !== 'aim' && c.caught) {
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

  const want = c.d === null ? '' : typeName(c.type, pt);
  const got = typeName(c.ball, pt);

  return (
    <dialog ref={dialog} className="pd-dialog" aria-labelledby="pd-title"
      onCancel={(e) => { e.preventDefault(); onClose(); }}
      onClick={(e) => { if (e.target === dialog.current) onClose(); }}>
      <div className="pd">
        <div className="pd-top">
          <span className={`pd-lens ${stage === 'throw' ? 'busy' : ''}`} aria-hidden="true" />
          <span className="pd-led r" aria-hidden="true" />
          <span className="pd-led y" aria-hidden="true" />
          <span className="pd-led g" aria-hidden="true" />
          <h2 id="pd-title" className="pd-title">Pokédex · {pt ? 'fase 1' : 'phase 1'}</h2>
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
                      ? 'Bem-vindo à Rota 1, treinador! Seu Pokédex tem um cérebro simples: um perceptron. Ele lê a Velocidade (x1) e a Defesa (x2) do Pokémon selvagem e escolhe a Ball: Fire Ball se net ≥ 0, Water Ball se net < 0.'
                      : 'Welcome to Route 1, trainer! Your Pokédex has a simple brain: a perceptron. It reads the wild Pokémon\'s Speed (x1) and Defense (x2) and picks the Ball: Fire Ball if net ≥ 0, Water Ball if net < 0.'}
                  </p>
                </div>
              </div>
              <p>
                {pt
                  ? 'A captura só funciona se a Ball for do tipo certo. Quando o Pokémon escapa, você usa o erro para treinar o cérebro. Ele começa com os números da Aula 1:'
                  : 'The capture only works if the Ball matches the real type. When the Pokémon breaks free, you use the error to train the brain. It starts with the numbers of Lesson 1:'}
              </p>
              <div className="pd-brain">{weights(game[0].before, lang)} · η = {num(LESSON_ETA, lang)} · x0 = −1</div>
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
              <p className="pd-appear">
                {stage === 'result' || stage === 'error' || stage === 'update' || stage === 'learned'
                  ? (c.caught
                    ? (pt ? `Gotcha! ${pokeName(c.pokemon)} foi capturado!` : `Gotcha! ${pokeName(c.pokemon)} was caught!`)
                    : (pt ? `Ah não! ${pokeName(c.pokemon)} escapou da Ball!` : `Oh no! ${pokeName(c.pokemon)} broke free!`))
                  : (pt
                    ? `Um ${revealed ? pokeName(c.pokemon) : '???'} selvagem apareceu!`
                    : `A wild ${revealed ? pokeName(c.pokemon) : '???'} appeared!`)}
              </p>

              <div className="pd-scan">
                <div><span>{pt ? 'Velocidade' : 'Speed'}</span><b>x1 = {c.x1}</b></div>
                <div><span>{pt ? 'Defesa' : 'Defense'}</span><b>x2 = {c.x2}</b></div>
                <div><span>bias</span><b>x0 = −1</b></div>
              </div>
              <div className={`pd-brain ${stage === 'learned' ? 'changed' : ''}`}>
                {pt ? 'Cérebro' : 'Brain'}: {weights(shownW, lang)}
              </div>

              {stage === 'net' && (
                <div className="pd-q">
                  <p><b>1 ·</b> {pt ? 'Calcule o net que o cérebro do Pokédex vê:' : 'Work out the net the Pokédex brain sees:'}{' '}
                    <span className="mono">net = w1·x1 + w2·x2 + θ·x0</span></p>
                  <Choices options={c.netOptions} keyOf={String} label={(v) => `net = ${num(v, lang)}`}
                    isRight={(v) => v === c.net} wrong={wrong} onPick={(r, key) => answer(r, key, 'ball')} />
                  {wrong.length > 0 && (
                    <p className="pd-hint">
                      {pt ? 'Dica: x0 = −1, então o bias entra como −θ. ' : 'Hint: x0 = −1, so the bias enters as −θ. '}
                      <span className="mono">({par(c.before.w1, lang)} · {c.x1}) + ({par(c.before.w2, lang)} · {c.x2}) + ({par(c.before.theta, lang)} · (−1))</span>
                    </p>
                  )}
                </div>
              )}

              {stage === 'ball' && (
                <div className="pd-q">
                  <p className="mono pd-done">net = {sumExpr(c.products, lang)} = {num(c.net, lang)}</p>
                  <p><b>2 ·</b> {pt
                    ? 'Função degrau: y = 1 se net ≥ 0, senão y = 0. Com y = 1 o Pokédex carrega a Fire Ball, com y = 0 a Water Ball. Qual Ball ele carrega?'
                    : 'Step function: y = 1 if net ≥ 0, else y = 0. With y = 1 the Pokédex loads the Fire Ball, with y = 0 the Water Ball. Which Ball does it load?'}</p>
                  <Choices options={['fire', 'water'] as Ball[]} keyOf={String} label={(b) => ballName(b)}
                    isRight={(b) => b === c.ball} wrong={wrong} onPick={(r, key) => answer(r, key, 'aim')} />
                  {wrong.length > 0 && (
                    <p className="pd-hint">
                      {pt
                        ? `Dica: o Pokédex segue o perceptron, não o que você sabe sobre o Pokémon. net = ${num(c.net, lang)} ${c.net >= 0 ? '≥' : '<'} 0.`
                        : `Hint: the Pokédex follows the perceptron, not what you know about the Pokémon. net = ${num(c.net, lang)} ${c.net >= 0 ? '≥' : '<'} 0.`}
                    </p>
                  )}
                </div>
              )}

              {stage === 'aim' && (
                <div className="pd-q">
                  <p className="mono pd-done">net = {num(c.net, lang)} {c.net >= 0 ? '≥' : '<'} 0 ⟹ y = {c.y} ⟹ {ballName(c.ball)}</p>
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
                  {c.d === null ? (
                    <p>
                      {pt
                        ? `Venusaur é do tipo Planta. O perceptron calculou net = ${num(c.net, lang)} < 0 e carregou a Water Ball, mas nenhuma das duas Balls serve: com uma única saída 0/1, o cérebro só conhece dois lados, Fogo ou Água.`
                        : `Venusaur is a Grass type. The perceptron worked out net = ${num(c.net, lang)} < 0 and loaded the Water Ball, but neither Ball fits: with a single 0/1 output, the brain only knows two sides, Fire or Water.`}
                    </p>
                  ) : c.caught ? (
                    <p>
                      {pt
                        ? `A previsão bateu com o tipo real: d = ${c.d} (${want}), y = ${c.y} (${got}), e = d − y = 0. Os pesos não mudam.`
                        : `The prediction matched the real type: d = ${c.d} (${want}), y = ${c.y} (${got}), e = d − y = 0. The weights stay the same.`}
                    </p>
                  ) : (
                    <p>
                      {pt
                        ? `${pokeName(c.pokemon)} é do tipo ${want} (d = ${c.d}), mas o Pokédex previu ${got} (y = ${c.y}). Ele continua no mato. Use o erro para treinar o cérebro.`
                        : `${pokeName(c.pokemon)} is a ${want} type (d = ${c.d}), but the Pokédex predicted ${got} (y = ${c.y}). It is still in the grass. Use the error to train the brain.`}
                    </p>
                  )}
                  <div className="pd-actions">
                    {c.d === null
                      ? <button className="btn btn--gold" onClick={() => setScreen('end')}>{pt ? 'Ver o registro no Pokédex' : 'See the Pokédex entry'}</button>
                      : c.caught
                        ? <button className="btn btn--gold" onClick={advance}>{pt ? 'Continuar' : 'Continue'}</button>
                        : <button className="btn btn--gold" onClick={() => setStage('error')}>{pt ? 'Treinar o Pokédex' : 'Train the Pokédex'}</button>}
                  </div>
                </div>
              )}

              {stage === 'error' && c.d !== null && (
                <div className="pd-q">
                  <p><b>3 ·</b> {pt ? 'Erro: ' : 'Error: '}<span className="mono">e = d − y</span>, {pt ? 'com' : 'with'} d = {c.d} ({want}) {pt ? 'e' : 'and'} y = {c.y} ({got}). {pt ? 'Quanto vale e?' : 'What is e?'}</p>
                  <Choices options={ERROR_OPTIONS} keyOf={String} label={(v) => `e = ${num(v, lang)}`}
                    isRight={(v) => v === c.e} wrong={wrong} onPick={(r, key) => answer(r, key, 'update')} />
                  {wrong.length > 0 && (
                    <p className="pd-hint">{pt ? 'Dica: o desejado menos o obtido, ' : 'Hint: the desired output minus the one you got, '}<span className="mono">{c.d} − {c.y}</span>.</p>
                  )}
                </div>
              )}

              {stage === 'update' && c.e !== null && (
                <div className="pd-q">
                  <p className="mono pd-done">e = {c.d} − {c.y} = {num(c.e, lang)}</p>
                  <p><b>4 ·</b> {pt ? 'Regra de aprendizado: ' : 'Learning rule: '}<span className="mono">w ← w + η·e·x</span>,{' '}
                    {pt ? 'com' : 'with'} η = {num(LESSON_ETA, lang)} {pt ? 'e' : 'and'} (x1, x2, x0) = ({c.x1}, {c.x2}, −1). {pt ? 'Quais são os novos pesos?' : 'What are the new weights?'}</p>
                  <Choices options={c.updateOptions} keyOf={(w) => `${w.w1}|${w.w2}|${w.theta}`} label={(w) => weights(w, lang)}
                    isRight={(w) => w.w1 === c.after.w1 && w.w2 === c.after.w2 && w.theta === c.after.theta}
                    wrong={wrong} onPick={(r, key) => answer(r, key, 'learned')} />
                  {wrong.length > 0 && (
                    <p className="pd-hint">
                      {pt ? 'Dica: some Δ a cada peso. ' : 'Hint: add Δ to each weight. '}
                      <span className="mono">
                        Δw1 = {num(LESSON_ETA, lang)}·{par(c.e, lang)}·{c.x1}, Δw2 = {num(LESSON_ETA, lang)}·{par(c.e, lang)}·{c.x2}, Δθ = {num(LESSON_ETA, lang)}·{par(c.e, lang)}·(−1)
                      </span>
                      {pt ? '. O bias anda no sentido contrário de e.' : '. The bias moves against the sign of e.'}
                    </p>
                  )}
                </div>
              )}

              {stage === 'learned' && (
                <div className="pd-q">
                  <p>
                    {pt ? 'Pokédex treinado! ' : 'Pokédex trained! '}
                    <span className="mono">{weights(c.before, lang)} → {weights(c.after, lang)}</span>
                  </p>
                  <div className="pd-actions">
                    <button className="btn btn--gold" onClick={advance}>{pt ? 'Voltar ao mato' : 'Back to the grass'}</button>
                  </div>
                </div>
              )}
            </div>
          )}

          {screen === 'epoch' && (
            <div className="pd-page pd-center">
              <span className="pd-kicker">{pt ? `Fim da época ${c.epoch}` : `End of epoch ${c.epoch}`}</span>
              <h3>{(() => {
                const errs = game.filter((g) => g.epoch === c.epoch && !g.caught).length;
                return pt ? `${errs} erro${errs === 1 ? '' : 's'} em ${epochSize} encontros` : `${errs} error${errs === 1 ? '' : 's'} in ${epochSize} encounters`;
              })()}</h3>
              <p>
                {pt
                  ? 'Uma época com erro pede outra. Charizard e Blastoise continuam na Rota 1, e o cérebro agora tem novos pesos:'
                  : 'An epoch with errors calls for another one. Charizard and Blastoise are still on Route 1, and the brain now has new weights:'}
              </p>
              <div className="pd-brain">{weights(c.after, lang)}</div>
              <div className="pd-actions pd-center">
                <button className="btn btn--gold" onClick={() => goTo(k + 1)}>▶ {pt ? `Época ${(c.epoch ?? 0) + 1}` : `Epoch ${(c.epoch ?? 0) + 1}`}</button>
              </div>
            </div>
          )}

          {screen === 'badge' && (
            <div className="pd-page pd-center">
              <span className="pd-badge" aria-hidden="true">∑</span>
              <span className="pd-kicker">{pt ? 'Insígnia do Perceptron' : 'Perceptron Badge'}</span>
              <h3>{pt ? `Época ${c.epoch} sem erros: o perceptron convergiu` : `Epoch ${c.epoch} with no errors: the perceptron converged`}</h3>
              <p>
                {pt
                  ? 'Pesos finais, os mesmos da Aula 1. A fronteira 22·Velocidade − 22·Defesa = 0,5 diz: mais rápido que resistente é Fogo, o contrário é Água.'
                  : 'Final weights, the same as Lesson 1. The boundary 22·Speed − 22·Defense = 0.5 says: faster than sturdy is Fire, the other way round is Water.'}
              </p>
              <div className="pd-brain">{weights(c.after, lang)}</div>
              <div className="pd-actions pd-center">
                <button className="btn btn--gold" onClick={() => goTo(k + 1)}>{pt ? 'Algo se mexe no mato…' : 'Something rustles in the grass…'}</button>
              </div>
            </div>
          )}

          {screen === 'end' && (
            <div className="pd-page">
              <div className="pd-entry">
                <Wild id="venusaur" revealed />
                <div>
                  <span className="pd-kicker">{pt ? 'Registro nº 003 · visto' : 'Entry #003 · seen'}</span>
                  <h3>Venusaur</h3>
                  <p className="pd-small">
                    {pt
                      ? `Velocidade 80, Defesa 83. net = ${num(c.net, lang)} ⟹ Water Ball. Tipo real: Planta. Com uma saída 0/1, o perceptron só tem dois lados, Fogo ou Água; um terceiro tipo não cabe nele.`
                      : `Speed 80, Defense 83. net = ${num(c.net, lang)} ⟹ Water Ball. Real type: Grass. With one 0/1 output, the perceptron only has two sides, Fire or Water; a third type does not fit.`}
                  </p>
                </div>
              </div>
              <div className="pd-score">
                <div><span>XP</span><b>{xp} / {totalQ * XP_PER_ANSWER}</b></div>
                <div><span>{pt ? 'De primeira' : 'First try'}</span><b>{firstTry} / {totalQ}</b></div>
                <div><span>{pt ? 'Capturados' : 'Caught'}</span><b>{caughtCount} / 3</b></div>
              </div>
              <p className="pd-center"><b className="gold">{rank}</b></p>
              <p className="pd-small">
                {pt
                  ? 'Mais saídas não bastam: com os nove iniciais, Ivysaur e Venusaur ficam dentro da área dos Fogo e Água, e nenhuma reta isola o tipo Planta. Para capturá-los, o Pokédex precisa de uma camada oculta. É o assunto da Aula 2 e da próxima fase.'
                  : 'More outputs are not enough: with the nine starters, Ivysaur and Venusaur sit inside the area of the Fire and Water ones, and no line isolates Grass. To catch them, the Pokédex needs a hidden layer. That is the subject of Lesson 2 and of the next phase.'}
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
