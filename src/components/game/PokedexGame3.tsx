import { useEffect, useMemo, useRef, useState } from 'react';
import { assets } from '../../api';
import { num, par, useLang, type Lang } from '../../i18n';
import { CNN_CARDS_PATH, CNN_MODEL_PATH, POOL_PATCH, type CachedCnn, type CardSample } from '../../lib/cnn';
import {
  buildGame3, convWindow, ETA_BY_HAND, PADDED_CORNER, questionCount3, XP_PER_ANSWER,
  type Ball3, type ClueChoice, type Encounter3, type LensEncounter, type LensKey, type LensQuestion, type PoolChoice,
  type ShortcutEncounter, type Vec3,
} from '../../lib/game3';
import { ballName, Choices, reducedMotion, typeName } from './parts';

const BASE = import.meta.env.BASE_URL;

type Screen = 'loading' | 'failed' | 'intro' | 'play' | 'badge' | 'cut-intro' | 'end';
/**
 * Lens: scan → q0 → q1 → ball → aim → throw → result.
 * Shortcut: scan → ball → aim → throw → result → loss → delta → pool → grad → update → clue → learned.
 */
type Stage = 'scan' | 'q0' | 'q1' | 'ball' | 'aim' | 'throw' | 'result' | 'loss' | 'delta' | 'pool' | 'grad' | 'update' | 'clue' | 'learned';
type DexState = 'unknown' | 'seen' | 'caught';

const BALLS: Ball3[] = ['fire', 'water', 'grass'];
const THROW_MS = 2600;
const SCAN_MS = 1500;

const vec = (v: number[], lang: Lang) => `(${v.map((x) => num(x, lang)).join(' ; ')})`;
const SUP: Record<string, string> = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
/** 8.8e-11 → "8.8 × 10⁻¹¹" (decimal comma in Portuguese). */
function sci(v: number, lang: Lang): string {
  const [m, e] = v.toExponential(1).split('e');
  return `${num(Number(m), lang)} × 10${String(Number(e)).split('').map((ch) => SUP[ch] ?? ch).join('')}`;
}

async function loadData(): Promise<{ model: CachedCnn; cards: CardSample[] }> {
  const [m, c] = await Promise.all([fetch(`${BASE}${CNN_MODEL_PATH}`), fetch(`${BASE}${CNN_CARDS_PATH}`)]);
  if (!m.ok || !c.ok) throw new Error('cnn files');
  return { model: (await m.json()) as CachedCnn, cards: (await c.json()) as CardSample[] };
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

/** The card the camera sees, in the round Pokédex window; a scan line runs over it while the CNN reads it. */
function CardWild({ c, scanning, className = '' }: { c: Encounter3; scanning: boolean; className?: string }) {
  return (
    <div className={`pd-wild ${c.type} ${className}`}>
      <img src={`${BASE}${c.file}`} alt={c.name} draggable={false} />
      {scanning && <span className="pd-scanline" aria-hidden="true" />}
    </div>
  );
}

export default function PokedexGame3({ onClose, onLesson }: { onClose: () => void; onLesson: (n: number) => void }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const dialog = useRef<HTMLDialogElement>(null);
  const [game, setGame] = useState<Encounter3[] | null>(null);
  const [screen, setScreen] = useState<Screen>('loading');
  const [k, setK] = useState(0);
  const [stage, setStage] = useState<Stage>('scan');
  const [wrong, setWrong] = useState<string[]>([]);
  const [xp, setXp] = useState(0);
  const [firstTry, setFirstTry] = useState(0);
  const [dex, setDex] = useState<Record<string, DexState>>({});
  const [catches, setCatches] = useState(0);

  useEffect(() => {
    const d = dialog.current;
    if (d && !d.open) d.showModal();
    return () => d?.close();
  }, []);

  useEffect(() => {
    let alive = true;
    loadData()
      .then(({ model, cards }) => { if (alive) { setGame(buildGame3(model, cards)); setScreen('intro'); } })
      .catch(() => { if (alive) setScreen('failed'); });
    return () => { alive = false; };
  }, []);

  const c = game?.[k];

  // The camera scans the card, then the first question shows.
  useEffect(() => {
    if (stage !== 'scan' || screen !== 'play' || !c) return;
    const t = window.setTimeout(() => setStage(c.kind === 'lens' ? 'q0' : 'ball'), reducedMotion() ? 200 : SCAN_MS);
    return () => window.clearTimeout(t);
  }, [stage, screen, c]);

  // The Ball flies, shakes three times, then the result shows.
  useEffect(() => {
    if (stage !== 'throw' || !c) return;
    const t = window.setTimeout(() => {
      setDex((cur) => ({ ...cur, [c.id]: c.caught ? 'caught' : 'seen' }));
      if (c.caught) setCatches((n) => n + 1);
      setStage('result');
    }, reducedMotion() ? 300 : THROW_MS);
    return () => window.clearTimeout(t);
  }, [stage, c]);

  const totalQ = useMemo(() => (game ? questionCount3(game) : 0), [game]);

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
    setStage('scan');
    setWrong([]);
    setScreen('play');
  }

  function advance() {
    if (!game || !c) return;
    const nx = game[k + 1];
    if (!nx) return setScreen('end');
    if (nx.kind !== c.kind) return setScreen('badge');
    goTo(k + 1);
  }

  function restart() {
    setXp(0);
    setFirstTry(0);
    setDex({});
    setCatches(0);
    setK(0);
    setStage('scan');
    setWrong([]);
    setScreen('intro');
  }

  const lensCount = game?.filter((g) => g.kind === 'lens').length ?? 0;
  const cut = game?.find((g): g is ShortcutEncounter => g.kind === 'shortcut');
  const rank = firstTry === totalQ
    ? (pt ? 'Mestre da Convolução' : 'Convolution Master')
    : firstTry >= totalQ - 4
      ? (pt ? 'Treinador de Ginásio' : 'Gym Trainer')
      : (pt ? 'Treinador Novato' : 'Rookie Trainer');

  const before = ['scan', 'q0', 'q1', 'ball', 'aim'].includes(stage);
  const field = (() => {
    if (!c) return null;
    if (stage === 'throw') {
      return (
        <>
          <CardWild c={c} scanning={false} className="absorb" />
          <span className="pd-ball-path throw"><span className={`pd-ball ${c.ball} wobble`} /></span>
        </>
      );
    }
    if (!before && c.caught) {
      return (
        <>
          <CardWild c={c} scanning={false} className="gone" />
          <span className="pd-ball-path"><span className={`pd-ball ${c.ball}`} /></span>
          <span className="pd-stars" aria-hidden="true"><i>★</i><i>★</i><i>★</i></span>
        </>
      );
    }
    return <CardWild c={c} scanning={stage === 'scan'} className={stage === 'result' ? 'pop' : ''} />;
  })();

  const appear = !c ? '' : stage === 'scan'
    ? (pt ? `Uma carta de ${c.name} apareceu! A câmera está lendo…` : `A ${c.name} card appeared! The camera is reading it…`)
    : !before && stage !== 'throw'
      ? (c.caught
        ? (pt ? `Gotcha! ${c.name} foi capturado!` : `Gotcha! ${c.name} was caught!`)
        : (pt ? `Ah não! ${c.name} escapou da Ball!` : `Oh no! ${c.name} broke free!`))
      : (pt ? `Carta escaneada: ${c.name}` : `Card scanned: ${c.name}`);

  const hud = !c ? '' : c.kind === 'lens'
    ? `${pt ? 'ROTA 3 · CÂMERA DO POKÉDEX · CARTA' : 'ROUTE 3 · POKÉDEX CAMERA · CARD'} ${c.n}/${lensCount}`
    : (pt ? 'ROTA 3 · A CARTA QUE ENGANA' : 'ROUTE 3 · THE TRICKY CARD');

  return (
    <dialog ref={dialog} className="pd-dialog" aria-labelledby="pd3-title"
      onCancel={(e) => { e.preventDefault(); onClose(); }}
      onClick={(e) => { if (e.target === dialog.current) onClose(); }}>
      <div className="pd">
        <div className="pd-top">
          <span className={`pd-lens ${stage === 'throw' || stage === 'scan' ? 'busy' : ''}`} aria-hidden="true" />
          <span className="pd-led r" aria-hidden="true" />
          <span className="pd-led y" aria-hidden="true" />
          <span className="pd-led g" aria-hidden="true" />
          <h2 id="pd3-title" className="pd-title">Pokédex · {pt ? 'fase 3' : 'phase 3'}</h2>
          <button className="pd-close" onClick={onClose} aria-label={pt ? 'Fechar o Pokédex' : 'Close the Pokédex'}>✕</button>
        </div>

        <div className="pd-screen" aria-live="polite">
          {screen === 'loading' && <p className="pd-small pd-center">{pt ? 'Ligando a câmera e carregando a CNN…' : 'Turning on the camera and loading the CNN…'}</p>}
          {screen === 'failed' && (
            <div className="pd-page">
              <p>{pt ? 'Não consegui carregar a rede e as cartas. Tente abrir a Aula 3 e voltar.' : 'Could not load the network and the cards. Try opening Lesson 3 and coming back.'}</p>
              <div className="pd-actions"><button className="btn btn--gold btn--sm" onClick={() => { onClose(); onLesson(3); }}>{pt ? 'Abrir a Aula 3' : 'Open Lesson 3'}</button></div>
            </div>
          )}

          {screen === 'intro' && (
            <div className="pd-page">
              <div className="pd-prof">
                <img src={assets.avatar} alt="" width={56} height={56} />
                <div>
                  <span className="pd-kicker">{pt ? 'Mensagem do Professor Rick' : 'Message from Professor Rick'}</span>
                  <p>
                    {pt
                      ? 'Bem-vindo à Rota 3! Seu Pokédex ganhou uma câmera. Ele não precisa mais de Velocidade e Defesa: fotografa a carta do Pokémon, espreme a foto em 32 × 32 pixels e passa pela CNN da Aula 3, com 1763 pesos treinados em 69 cartas.'
                      : 'Welcome to Route 3! Your Pokédex got a camera. It no longer needs Speed and Defense: it takes a picture of the Pokémon card, squeezes it to 32 × 32 pixels and runs it through the CNN of Lesson 3, with 1763 weights trained on 69 cards.'}
                  </p>
                </div>
              </div>
              <div className="pd-brain">
                32×32×3 → conv 5×5 ×6 → pool → conv 3×3 ×8 → pool → 288 → softmax
              </div>
              <p>
                {pt
                  ? 'As cartas desta rota são de teste: a rede nunca as viu no treino. Em cada uma, você confere uma peça da CNN (pixels, convolução, pooling, ReLU) antes de lançar a Ball que o Pokédex escolher.'
                  : 'The cards on this route are test cards: the network never saw them in training. For each one, you check one piece of the CNN (pixels, convolution, pooling, ReLU) before throwing the Ball the Pokédex picks.'}
              </p>
              <p className="pd-small">
                {pt
                  ? `Acerte de primeira para ganhar ${XP_PER_ANSWER} XP por resposta. Errou? O Pokédex dá uma dica e você tenta de novo.`
                  : `Answer right on the first try to earn ${XP_PER_ANSWER} XP per answer. Got it wrong? The Pokédex gives a hint and you try again.`}
              </p>
              <div className="pd-actions">
                <button className="btn btn--gold" onClick={() => goTo(0)}>{pt ? '▶ Ligar a câmera' : '▶ Turn on the camera'}</button>
              </div>
            </div>
          )}

          {screen === 'play' && c && (
            <div className="pd-page">
              <div className="pd-hud"><span>{hud}</span><span>XP {xp}</span></div>
              <div className="pd-field">{field}</div>
              <p className="pd-appear">{appear}</p>
              {c.kind === 'lens'
                ? <LensPlay c={c} stage={stage} wrong={wrong} lang={lang} answer={answer} setStage={setStage} advance={advance} />
                : <CutPlay c={c} stage={stage} wrong={wrong} lang={lang} answer={answer} setStage={setStage} advance={advance} />}
            </div>
          )}

          {screen === 'badge' && (
            <div className="pd-page pd-center">
              <span className="pd-badge" aria-hidden="true">∗</span>
              <span className="pd-kicker">{pt ? 'Insígnia da Convolução' : 'Convolution Badge'}</span>
              <h3>{pt ? 'Três cartas novas, três capturas' : 'Three new cards, three catches'}</h3>
              <p>
                {pt
                  ? 'A câmera leu Fogo, Água e Planta em cartas que a rede nunca tinha visto. Você conferiu cada peça: 3072 números por carta, um filtro que acende na borda, 28 × 28 depois do primeiro filtro, o maior de cada bloco no pooling, a ReLU zerando o negativo e 456 pesos na primeira camada.'
                  : 'The camera read Fire, Water and Grass on cards the network had never seen. You checked every piece: 3072 numbers per card, a filter that lights up on the edge, 28 × 28 after the first filter, the largest of each block in pooling, ReLU zeroing the negatives and 456 weights in the first layer.'}
              </p>
              <p className="pd-small">{pt ? 'Mas ainda falta uma carta na rota, e ela engana.' : 'But there is one more card on the route, and it is tricky.'}</p>
              <div className="pd-actions pd-center">
                <button className="btn btn--gold" onClick={() => setScreen('cut-intro')}>{pt ? 'O Professor Rick está chamando…' : 'Professor Rick is calling…'}</button>
              </div>
            </div>
          )}

          {screen === 'cut-intro' && (
            <div className="pd-page">
              <div className="pd-prof">
                <img src={assets.avatar} alt="" width={56} height={56} />
                <div>
                  <span className="pd-kicker">{pt ? 'Mensagem do Professor Rick' : 'Message from Professor Rick'}</span>
                  <p>
                    {pt
                      ? 'Das 24 cartas de teste, a CNN erra só uma. Ela está logo ali no mato. Siga o Pokédex, mesmo que você desconfie dele, e depois vamos ver o caminho que o erro faz de volta pela rede.'
                      : 'Of the 24 test cards, the CNN gets only one wrong. It is right there in the grass. Follow the Pokédex, even if you doubt it, and then we will follow the path the error takes back through the network.'}
                  </p>
                </div>
              </div>
              <div className="pd-actions">
                <button className="btn btn--gold" onClick={() => goTo(k + 1)}>{pt ? '▶ Voltar ao mato' : '▶ Back to the grass'}</button>
              </div>
            </div>
          )}

          {screen === 'end' && cut && (
            <div className="pd-page">
              <div className="pd-center">
                <span className="pd-badge" aria-hidden="true">◉</span>
                <span className="pd-kicker">{pt ? 'Insígnia da Visão' : 'Vision Badge'}</span>
              </div>
              <div className="pd-entry">
                <CardWild c={cut} scanning={false} />
                <div>
                  <span className="pd-kicker">{pt ? `Registro · ${cut.name} · visto, não capturado` : `Entry · ${cut.name} · seen, not caught`}</span>
                  <h3>{cut.name}</h3>
                  <p className="pd-small">
                    {pt
                      ? `Tipo Planta, numa carta de arte cheia com um Pokémon vermelho. A CNN disse Fogo. Um classificador que só olha a cor média acerta ${cut.baselineHits} das ${cut.testCards} cartas de teste; a CNN acerta ${cut.cnnHits}. Ela aprendeu principalmente a cor da carta.`
                      : `Grass type, on a full-art card with a red Pokémon. The CNN said Fire. A classifier that only looks at the average colour gets ${cut.baselineHits} of the ${cut.testCards} test cards right; the CNN gets ${cut.cnnHits}. It mostly learned the card colour.`}
                  </p>
                </div>
              </div>
              <div className="pd-score">
                <div><span>XP</span><b>{xp} / {totalQ * XP_PER_ANSWER}</b></div>
                <div><span>{pt ? 'De primeira' : 'First try'}</span><b>{firstTry} / {totalQ}</b></div>
                <div><span>{pt ? 'Capturas' : 'Catches'}</span><b>{catches} / {game?.length ?? 0}</b></div>
              </div>
              <p className="pd-center"><b className="gold">{rank}</b></p>
              <p className="pd-small">
                {pt
                  ? 'Por que não treinar a rede com o Orbeetle até ela acertar? Porque ele é uma carta de teste: se a rede aprender com ele, o teste deixa de medir cartas novas. O caminho honesto é ter mais cartas de arte cheia no treino, ou usar transferência de aprendizado, com uma rede que já sabe ler formas (parte 7 da Aula 3).'
                  : 'Why not train the network on Orbeetle until it gets it right? Because it is a test card: if the network learns from it, the test no longer measures new cards. The honest fix is more full-art cards in training, or transfer learning with a network that already reads shapes (part 7 of Lesson 3).'}
              </p>
              <div className="pd-actions">
                <button className="btn btn--ghost btn--sm" onClick={restart}>↺ {pt ? 'Jogar de novo' : 'Play again'}</button>
                <button className="btn btn--gold btn--sm" onClick={() => { onClose(); onLesson(3); }}>{pt ? 'Abrir a Aula 3' : 'Open Lesson 3'}</button>
              </div>
            </div>
          )}
        </div>

        <div className="pd-dex pd-dex--4" aria-label={pt ? 'Registros do Pokédex' : 'Pokédex entries'}>
          {(game ?? []).map((g) => {
            const st = dex[g.id] ?? 'unknown';
            return (
              <div key={g.id} className={`pd-slot ${st}`}>
                {st === 'unknown'
                  ? <span className="pd-slot-img unknown">?</span>
                  : <img className="pd-slot-img" src={`${BASE}${g.file}`} alt="" />}
                <span className="pd-slot-name">{st === 'unknown' ? '???' : g.name}</span>
                <span className="pd-slot-state">
                  {st === 'caught' ? (pt ? '● capturado' : '● caught') : st === 'seen' ? (pt ? '○ visto' : '○ seen') : '—'}
                </span>
              </div>
            );
          })}
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

/** Question text, answer label, worked line and hint of each lens question. */
function lensText(q: LensQuestion, lang: Lang) {
  const pt = lang === 'pt';
  const w = convWindow();
  const block = [POOL_PATCH[0][0], POOL_PATCH[0][1], POOL_PATCH[1][0], POOL_PATCH[1][1]];
  const T: Record<LensKey, { ask: string; label: (v: number) => string; done: string; hint: string }> = {
    pixels: {
      ask: pt ? 'A câmera espreme a carta em 32 × 32 pixels, e cada pixel tem R, G e B. Quantos números entram na rede?'
        : 'The camera squeezes the card to 32 × 32 pixels, and each pixel has R, G and B. How many numbers go into the network?',
      label: (v) => `${v}`,
      done: `32 × 32 × 3 = ${q.answer} ${pt ? 'números' : 'numbers'}`,
      hint: pt ? 'Dica: são 32 × 32 pixels, e cada um tem três cores.' : 'Hint: there are 32 × 32 pixels, and each one has three colours.',
    },
    conv: {
      ask: pt
        ? `O filtro de borda vertical (coluna esquerda +1, meio 0, direita −1) está nesta janela 3 × 3 de um pedaço da carta: ${w.win.map((r) => r.join(' ')).join(' | ')}. Quanto vale o net?`
        : `The vertical-edge filter (left column +1, middle 0, right −1) is on this 3 × 3 window of a patch of the card: ${w.win.map((r) => r.join(' ')).join(' | ')}. What is the net?`,
      label: (v) => `net = ${num(v, lang)}`,
      done: `net = (${w.win.map((r) => r[0]).join(' + ')}) − (${w.win.map((r) => r[2]).join(' + ')}) = ${w.left} − ${w.right} = ${w.net}`,
      hint: pt ? 'Dica: os pesos da direita são −1, então a coluna da direita é subtraída, não somada. A do meio tem peso 0.'
        : 'Hint: the right-hand weights are −1, so the right column is subtracted, not added. The middle one has weight 0.',
    },
    size: {
      ask: pt ? 'A primeira camada usa filtros 5 × 5, padding 0 e stride 1 na imagem 32 × 32. Qual é o lado do mapa de saída?'
        : 'The first layer uses 5 × 5 filters, padding 0 and stride 1 on the 32 × 32 image. What is the side of the output map?',
      label: (v) => `${v} × ${v}`,
      done: `⌊(32 + 2·0 − 5) / 1⌋ + 1 = ${q.answer}`,
      hint: pt ? 'Dica: saída = ⌊(n + 2p − k) / s⌋ + 1. Sem padding o mapa encolhe, e não esqueça o + 1.'
        : 'Hint: output = ⌊(n + 2p − k) / s⌋ + 1. With no padding the map shrinks, and don’t forget the + 1.',
    },
    pool: {
      ask: pt ? `Max pooling 2 × 2: o primeiro bloco do mapa tem ${block.join(', ')}. Que número segue para a próxima camada?`
        : `2 × 2 max pooling: the first block of the map has ${block.join(', ')}. Which number moves on to the next layer?`,
      label: (v) => num(v, lang),
      done: `max(${block.join(', ')}) = ${q.answer}`,
      hint: pt ? 'Dica: no max pooling só o mais forte do bloco passa. A média é o average pooling.'
        : 'Hint: in max pooling only the strongest of the block passes. The mean is average pooling.',
    },
    relu: {
      ask: pt ? `Com padding 1, o filtro no canto da imagem dá net = ${num(PADDED_CORNER, lang)}. Quanto sai depois da ReLU?`
        : `With padding 1, the filter at the corner of the image gives net = ${num(PADDED_CORNER, lang)}. What comes out after ReLU?`,
      label: (v) => num(v, lang),
      done: `ReLU(${num(PADDED_CORNER, lang)}) = max(0, ${num(PADDED_CORNER, lang)}) = ${q.answer}`,
      hint: pt ? 'Dica: ReLU(net) = max(0, net). Negativo quer dizer "não achei o padrão aqui".'
        : 'Hint: ReLU(net) = max(0, net). Negative means "pattern not found here".',
    },
    params: {
      ask: pt ? 'A primeira convolução tem 6 filtros, e cada um olha 3 cores numa janela 5 × 5, com um bias por filtro. Quantos pesos ela tem?'
        : 'The first convolution has 6 filters, each looking at 3 colours in a 5 × 5 window, with one bias per filter. How many weights does it have?',
      label: (v) => `${v}`,
      done: `6 · (3 · 5 · 5) + 6 = ${q.answer}`,
      hint: pt ? 'Dica: cada filtro tem 3 · 5 · 5 = 75 pesos. São 6 filtros, e cada um ainda tem o seu bias, como o θ das aulas 1 e 2.'
        : 'Hint: each filter has 3 · 5 · 5 = 75 weights. There are 6 filters, and each one also has its bias, like the θ of lessons 1 and 2.',
    },
  };
  return T[q.key];
}

/** Part A: one CNN piece per question, then the Ball the softmax picks. */
function LensPlay({ c, stage, wrong, lang, answer, setStage, advance }: PlayProps<LensEncounter>) {
  const pt = lang === 'pt';
  const done = (i: number) => lensText(c.questions[i], lang).done;
  const ask = (i: 0 | 1, next: Stage) => {
    const q = c.questions[i], t = lensText(q, lang);
    return (
      <div className="pd-q">
        {i === 1 && <p className="mono pd-done">{done(0)}</p>}
        <p><b>{c.n * 3 - 2 + i} ·</b> {t.ask}</p>
        <Choices options={q.options} keyOf={String} label={t.label}
          isRight={(v) => v === q.answer} wrong={wrong} onPick={(r, key) => answer(r, key, next)} />
        {wrong.length > 0 && <p className="pd-hint">{t.hint}</p>}
      </div>
    );
  };

  return (
    <>
      {stage === 'scan' && <p className="pd-small pd-center">{pt ? '…escaneando 32 × 32 × 3…' : '…scanning 32 × 32 × 3…'}</p>}
      {stage === 'q0' && ask(0, 'q1')}
      {stage === 'q1' && ask(1, 'ball')}

      {stage === 'ball' && (
        <div className="pd-q">
          <p className="mono pd-done">{done(0)}<br />{done(1)}</p>
          <Probs p={c.p} lang={lang} />
          <p><b>{c.n * 3} ·</b> {pt
            ? 'A CNN terminou a leitura e a softmax deu uma probabilidade para cada tipo. Qual Ball o Pokédex carrega?'
            : 'The CNN finished reading and softmax gave each type a probability. Which Ball does the Pokédex load?'}</p>
          <Choices options={BALLS} keyOf={String} label={(b) => ballName(b)}
            isRight={(b) => b === c.ball} wrong={wrong} onPick={(r, key) => answer(r, key, 'aim')} />
          {wrong.length > 0 && <p className="pd-hint">{pt ? 'Dica: o Pokédex carrega a Ball do maior p.' : 'Hint: the Pokédex loads the Ball with the largest p.'}</p>}
        </div>
      )}

      {stage === 'aim' && (
        <div className="pd-q">
          <p className="mono pd-done">max p = {num(Math.max(...c.p), lang)} ⟹ {ballName(c.ball)}</p>
          <div className="pd-actions">
            <button className="btn btn--gold" onClick={() => setStage('throw')}>{pt ? `Lançar a ${ballName(c.ball)}!` : `Throw the ${ballName(c.ball)}!`}</button>
          </div>
        </div>
      )}

      {stage === 'throw' && <p className="pd-small pd-center">{pt ? '…balança… balança… balança…' : '…wobble… wobble… wobble…'}</p>}

      {stage === 'result' && (
        <div className="pd-q">
          <p>
            {pt
              ? `A CNN leu ${typeName(c.type, pt)} numa carta que nunca tinha visto, e acertou. Nenhum peso muda: a rede já foi treinada.`
              : `The CNN read ${typeName(c.type, pt)} on a card it had never seen, and got it right. No weight changes: the network is already trained.`}
          </p>
          <div className="pd-actions"><button className="btn btn--gold" onClick={advance}>{pt ? 'Continuar' : 'Continue'}</button></div>
        </div>
      )}
    </>
  );
}

function poolText(o: PoolChoice, pt: boolean): string {
  if (o === 'winner') return pt ? 'Só a casa que venceu cada bloco recebe o erro; as outras três recebem 0' : 'Only the cell that won each block gets the error; the other three get 0';
  if (o === 'all') return pt ? 'As quatro casas do bloco recebem o erro dividido por 4' : 'All four cells of the block get the error divided by 4';
  return pt ? 'Nenhuma: o pooling não tem pesos, então o erro para ali' : 'None: pooling has no weights, so the error stops there';
}

function clueText(o: ClueChoice, pt: boolean): string {
  if (o === 'colour') return pt ? 'Principalmente a cor da carta (moldura e fundo)' : 'Mostly the colour of the card (frame and background)';
  if (o === 'shape') return pt ? 'Principalmente o desenho e a forma do Pokémon' : 'Mostly the drawing and the shape of the Pokémon';
  return pt ? 'O texto escrito na carta, como o nome e o tipo' : 'The text printed on the card, like the name and the type';
}

/** Part B: the card the CNN gets wrong, and the error going back. */
function CutPlay({ c, stage, wrong, lang, answer, setStage, advance }: PlayProps<ShortcutEncounter>) {
  const pt = lang === 'pt';
  const grass = typeName('grass', pt);
  const terms = c.gradTerms.map(([d, x]) => `${par(d, lang)}·${x}`).join(' + ');

  return (
    <>
      {stage === 'scan' && <p className="pd-small pd-center">{pt ? '…escaneando 32 × 32 × 3…' : '…scanning 32 × 32 × 3…'}</p>}
      {stage !== 'scan' && <Probs p={c.p} lang={lang} />}

      {stage === 'ball' && (
        <div className="pd-q">
          <p><b>10 ·</b> {pt ? 'A softmax respondeu. Qual Ball o Pokédex carrega?' : 'Softmax has answered. Which Ball does the Pokédex load?'}</p>
          <Choices options={BALLS} keyOf={String} label={(b) => ballName(b)}
            isRight={(b) => b === c.ball} wrong={wrong} onPick={(r, key) => answer(r, key, 'aim')} />
          {wrong.length > 0 && (
            <p className="pd-hint">{pt ? 'Dica: o Pokédex segue a rede, não o que você acha da carta. Escolha o maior p.' : 'Hint: the Pokédex follows the network, not what you think of the card. Pick the largest p.'}</p>
          )}
        </div>
      )}

      {stage === 'aim' && (
        <div className="pd-q">
          <p className="mono pd-done">max p = {num(Math.max(...c.p), lang)} ⟹ {ballName(c.ball)}</p>
          <div className="pd-actions">
            <button className="btn btn--gold" onClick={() => setStage('throw')}>{pt ? `Lançar a ${ballName(c.ball)}!` : `Throw the ${ballName(c.ball)}!`}</button>
          </div>
        </div>
      )}

      {stage === 'throw' && <p className="pd-small pd-center">{pt ? '…balança… balança… balança…' : '…wobble… wobble… wobble…'}</p>}

      {stage === 'result' && (
        <div className="pd-q">
          <p>
            {pt
              ? `O ${c.name} é ${grass}: d = (0 ; 0 ; 1). A carta tem arte cheia e o Pokémon vermelho cobre quase tudo; a CNN deu quase 100% de Fogo. Vamos seguir o erro de volta.`
              : `${c.name} is ${grass}: d = (0 ; 0 ; 1). It is a full-art card and the red Pokémon covers almost all of it; the CNN gave almost 100% Fire. Let’s follow the error back.`}
          </p>
          <div className="pd-actions"><button className="btn btn--gold" onClick={() => setStage('loss')}>{pt ? 'Seguir o erro' : 'Follow the error'}</button></div>
        </div>
      )}

      {stage === 'loss' && (
        <div className="pd-q">
          <p><b>11 ·</b> {pt ? 'Entropia cruzada: ' : 'Cross-entropy: '}<span className="mono">L = −ln p_{grass}</span>.{' '}
            {pt ? `A probabilidade exata de ${grass} é ${sci(c.pTrue, lang)}, quase zero. Quanto vale a loss?` : `The exact ${grass} probability is ${sci(c.pTrue, lang)}, almost zero. What is the loss?`}</p>
          <Choices options={c.lossOptions} keyOf={String} label={(v) => `L = ${num(v, lang)}`}
            isRight={(v) => v === c.loss} wrong={wrong} onPick={(r, key) => answer(r, key, 'delta')} />
          {wrong.length > 0 && (
            <p className="pd-hint">
              {pt
                ? `Dica: a loss usa a probabilidade do tipo certo (${grass}), não a do tipo previsto. Quanto menor a probabilidade, maior a loss, e o menos deixa o número positivo.`
                : `Hint: the loss uses the probability of the right type (${grass}), not the predicted one. The smaller the probability, the larger the loss, and the minus makes the number positive.`}
            </p>
          )}
        </div>
      )}

      {stage === 'delta' && (
        <div className="pd-q">
          <p className="mono pd-done">L = −ln p_{grass} = {num(c.loss, lang)}</p>
          <p><b>12 ·</b> {pt ? 'Erro na saída: ' : 'Output error: '}<span className="mono">δ = p − d</span>, {pt ? 'com' : 'with'} p = {vec(c.p, lang)} {pt ? 'e' : 'and'} d = (0 ; 0 ; 1). {pt ? 'Quanto vale δ?' : 'What is δ?'}</p>
          <Choices options={c.deltaOptions} keyOf={(v) => v.join('|')} label={(v) => `δ = ${vec(v, lang)}`}
            isRight={(v) => v.join('|') === c.delta.join('|')} wrong={wrong} onPick={(r, key) => answer(r, key, 'pool')} />
          {wrong.length > 0 && <p className="pd-hint">{pt ? 'Dica: o obtido menos o desejado, posição por posição, como na Aula 2.' : 'Hint: what you got minus what you wanted, one position at a time, as in Lesson 2.'}</p>}
        </div>
      )}

      {stage === 'pool' && (
        <div className="pd-q">
          <p className="mono pd-done">δ = {vec(c.delta, lang)} → {pt ? 'densa → 288 → pool 2' : 'dense → 288 → pool 2'}</p>
          <p><b>13 ·</b> {pt ? 'O erro chegou ao max pooling. Para onde ele vai dentro de cada bloco 2 × 2?' : 'The error has reached max pooling. Where does it go inside each 2 × 2 block?'}</p>
          <Choices options={c.poolOptions} keyOf={String} label={(o) => poolText(o, pt)} wide
            isRight={(o) => o === 'winner'} wrong={wrong} onPick={(r, key) => answer(r, key, 'grad')} />
          {wrong.length > 0 && (
            <p className="pd-hint">
              {pt ? 'Dica: só o vencedor de cada bloco influenciou a saída. Dividir por 4 é o average pooling, e o pooling não para o erro: ele só não tem pesos para atualizar.'
                : 'Hint: only the winner of each block affected the output. Dividing by 4 is average pooling, and pooling does not stop the error: it just has no weights to update.'}
            </p>
          )}
        </div>
      )}

      {stage === 'grad' && (
        <div className="pd-q">
          <p className="mono pd-done">{pt ? 'pool: erro só para o vencedor · ReLU: passa onde net > 0' : 'pool: error only to the winner · ReLU: passes where net > 0'}</p>
          <p><b>14 ·</b> {pt
            ? 'Agora um pedaço da convolução, à mão, como na Aula 3: entrada x = (1 2 0 | 0 1 3 | 2 1 1), filtro K = (1 −1 | 0 2) e erro na saída δ = (0,5 −1 | 0 0,25). Quanto vale ∂L/∂K do canto de baixo à direita?'
            : 'Now a piece of the convolution, by hand, as in Lesson 3: input x = (1 2 0 | 0 1 3 | 2 1 1), filter K = (1 −1 | 0 2) and output error δ = (0.5 −1 | 0 0.25). What is ∂L/∂K for the bottom-right weight?'}</p>
          <Choices options={c.gradOptions} keyOf={String} label={(v) => `∂L/∂K = ${num(v, lang)}`}
            isRight={(v) => v === c.grad} wrong={wrong} onPick={(r, key) => answer(r, key, 'update')} />
          {wrong.length > 0 && (
            <p className="pd-hint">
              {pt ? 'Dica: o peso de baixo à direita tocou, em cada janela, o pixel de baixo à direita dela: 1, 3, 1 e 1. Some δ · pixel. ' : 'Hint: in each window, the bottom-right weight touched that window’s bottom-right pixel: 1, 3, 1 and 1. Add up δ · pixel. '}
              <span className="mono">{terms}</span>
            </p>
          )}
        </div>
      )}

      {stage === 'update' && (
        <div className="pd-q">
          <p className="mono pd-done">∂L/∂K = {terms} = {num(c.grad, lang)}</p>
          <p><b>15 ·</b> {pt ? 'Atualize o peso: ' : 'Update the weight: '}<span className="mono">K ← K − η·∂L/∂K</span>, {pt ? 'com' : 'with'} K = {c.k} {pt ? 'e' : 'and'} η = {num(ETA_BY_HAND, lang)}. {pt ? 'Qual é o novo K?' : 'What is the new K?'}</p>
          <Choices options={c.kOptions} keyOf={String} label={(v) => `K = ${num(v, lang)}`}
            isRight={(v) => v === c.kAfter} wrong={wrong} onPick={(r, key) => answer(r, key, 'clue')} />
          {wrong.length > 0 && (
            <p className="pd-hint">
              {pt ? 'Dica: multiplique o gradiente por η e subtraia. Menos com menos dá mais. ' : 'Hint: multiply the gradient by η and subtract. Minus a minus is a plus. '}
              <span className="mono">{c.k} − {num(ETA_BY_HAND, lang)}·{par(c.grad, lang)}</span>
            </p>
          )}
        </div>
      )}

      {stage === 'clue' && (
        <div className="pd-q">
          <p className="mono pd-done">K: {c.k} → {num(c.kAfter, lang)}</p>
          <p><b>16 ·</b> {pt
            ? `Uma última pista. Um classificador sem convolução nenhuma, que só compara a cor média da carta com a cor média de cada tipo, acerta ${c.baselineHits} das ${c.testCards} cartas de teste. A CNN acerta ${c.cnnHits}. O que a CNN aprendeu a ler, principalmente?`
            : `One last clue. A classifier with no convolution at all, which only compares the card’s average colour with the average colour of each type, gets ${c.baselineHits} of the ${c.testCards} test cards right. The CNN gets ${c.cnnHits}. What did the CNN mostly learn to read?`}</p>
          <Choices options={c.clueOptions} keyOf={String} label={(o) => clueText(o, pt)} wide
            isRight={(o) => o === 'colour'} wrong={wrong} onPick={(r, key) => answer(r, key, 'learned')} />
          {wrong.length > 0 && (
            <p className="pd-hint">
              {pt ? 'Dica: se só a cor média já acerta quase tudo, a CNN quase não ganhou nada lendo formas. E o Orbeetle vermelho foi parar em Fogo.'
                : 'Hint: if the average colour alone gets almost everything right, the CNN gained very little from reading shapes. And the red Orbeetle ended up as Fire.'}
            </p>
          )}
        </div>
      )}

      {stage === 'learned' && (
        <div className="pd-q">
          <p>
            {pt
              ? 'Isso mesmo: cada tipo tem a cor da sua carta (laranja, azul, verde), e a rede pegou esse atalho. Funciona nestas cartas, mas engana numa carta de arte cheia como esta.'
              : 'Right: each type has its own card colour (orange, blue, green), and the network took that shortcut. It works on these cards, but it fails on a full-art card like this one.'}
          </p>
          <div className="pd-actions"><button className="btn btn--gold" onClick={advance}>{pt ? 'Ver o registro no Pokédex' : 'See the Pokédex entry'}</button></div>
        </div>
      )}
    </>
  );
}
