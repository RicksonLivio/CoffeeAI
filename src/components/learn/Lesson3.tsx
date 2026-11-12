import { useEffect, useMemo, useRef, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { fx, num, par, useLang, type Lang } from '../../i18n';
import {
  BACK_DELTA, BACK_K, BACK_X, CLASSES, CNN_CARDS_PATH, CNN_CONFIG, CNN_MODEL_PATH, EDGE_PATCH, KERNELS, POOL_PATCH,
  colorBaseline, conv2dGrid, countParams, decodePx, evaluate, forward, inputGrad, kernelGrad, maxPoolBack, outSize,
  pool2dGrid, tensor, toTensor, trainCardNet, type CachedCnn, type CardClass, type CardNet, type CardSample, type PoolKind, type Tensor,
} from '../../lib/cnn';
import { ArchDiagram, BlockGrid, channel, MapCanvas, ProbBars, RgbCanvas } from './cnn/Blocks';
import Note from './Note';

const BASE = import.meta.env.BASE_URL;
const TYPE_NAME: Record<Lang, Record<CardClass, string>> = {
  pt: { fire: 'Fogo', water: 'Água', grass: 'Planta' },
  en: { fire: 'Fire', water: 'Water', grass: 'Grass' },
};
const names = (lang: Lang) => CLASSES.map((c) => TYPE_NAME[lang][c]);
const argmax = (p: number[]) => p.indexOf(Math.max(...p));
const LESSON_IDS = ['charizard', 'blastoise', 'venusaur'];

interface CnnData { model: CachedCnn; cards: CardSample[]; x: Map<string, Tensor>; source: 'cache' | 'browser' }

/** Reads the 32×32 cards and the cached network; if the network file is missing, trains it here (a few seconds). */
async function loadCnn(): Promise<CnnData> {
  const cards: CardSample[] = await (await fetch(`${BASE}${CNN_CARDS_PATH}`)).json();
  const x = new Map(cards.map((c) => [c.id, toTensor(decodePx(c.px))]));
  try {
    const r = await fetch(`${BASE}${CNN_MODEL_PATH}`);
    if (!r.ok) throw new Error(String(r.status));
    return { model: (await r.json()) as CachedCnn, cards, x, source: 'cache' };
  } catch {
    await new Promise((res) => setTimeout(res, 50));
    const set = (s: 'train' | 'test') => cards.filter((c) => c.split === s).map((c) => ({ x: x.get(c.id)!, y: CLASSES.indexOf(c.type) }));
    const tr = set('train'), te = set('test');
    const { net, curve } = trainCardNet(tr, te, CNN_CONFIG);
    const a = evaluate(net, tr), b = evaluate(net, te);
    const model: CachedCnn = {
      config: CNN_CONFIG, params: countParams(net), net, curve,
      train: { n: tr.length, accuracy: a.accuracy, confusion: a.confusion }, test: { n: te.length, accuracy: b.accuracy, confusion: b.confusion },
      samples: cards.map(({ px: _px, ...c }) => ({ ...c, p: forward(net, x.get(c.id)!).p })),
    };
    return { model, cards, x, source: 'browser' };
  }
}

export default function Lesson3() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [data, setData] = useState<CnnData | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => { loadCnn().then(setData, () => setFailed(true)); }, []);
  const jump = (id: string) => (e: React.MouseEvent) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView(); };
  const wait = <div className="panel">{failed ? (pt ? 'Não consegui carregar as cartas.' : 'Could not load the cards.') : (pt ? 'Carregando as cartas…' : 'Loading the cards…')}</div>;

  return (
    <div className="lesson-body">
      <div className="parts">
        <a href="#l3-imagem" onClick={jump('l3-imagem')}><b>1</b>{pt ? 'Uma imagem é uma grade de números' : 'An image is a grid of numbers'}</a>
        <a href="#l3-conv" onClick={jump('l3-conv')}><b>2</b>{pt ? 'Convolução: o filtro que desliza' : 'Convolution: the sliding filter'}</a>
        <a href="#l3-pool" onClick={jump('l3-pool')}><b>3</b>{pt ? 'ReLU e pooling' : 'ReLU and pooling'}</a>
        <a href="#l3-arq" onClick={jump('l3-arq')}><b>4</b>{pt ? 'Do quadrado grande ao pequeno' : 'From the big square to the small one'}</a>
        <a href="#l3-treino" onClick={jump('l3-treino')}><b>5</b>{pt ? 'Como a CNN aprende' : 'How the CNN learns'}</a>
        <a href="#l3-pokedex" onClick={jump('l3-pokedex')}><b>6</b>{pt ? 'Pokédex: descubra o tipo' : 'Pokédex: discover the type'}</a>
        <a href="#l3-tecnicas" onClick={jump('l3-tecnicas')}><b>7</b>{pt ? 'Outras técnicas' : 'Other techniques'}</a>
      </div>
      {data ? <Picture data={data} /> : wait}
      <Convolution />
      <Pooling />
      {data ? <Architecture data={data} /> : wait}
      {data ? <Learning data={data} /> : wait}
      {data ? <Pokedex data={data} /> : wait}
      <Techniques />
    </div>
  );
}

function CardPicker({ value, onChange, data }: { value: string; onChange: (id: string) => void; data: CnnData }) {
  return (
    <div className="poke-pick" role="radiogroup" aria-label="Pokémon">
      {LESSON_IDS.map((id) => {
        const c = data.cards.find((k) => k.id === id)!;
        return (
          <button key={id} role="radio" aria-checked={value === id} className={`${c.type} ${value === id ? 'active' : ''}`} onClick={() => onChange(id)}>
            {c.name}
          </button>
        );
      })}
    </div>
  );
}

// --------------------------------------------------------------------------- //
// 1. An image is a grid of numbers
// --------------------------------------------------------------------------- //
function Picture({ data }: { data: CnnData }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [id, setId] = useState('charizard');
  const card = data.cards.find((c) => c.id === id)!, x = data.x.get(id)!;
  const px = decodePx(card.px);
  const corner = [0, 1, 2].map((c) => px[(0 * 32 + 0) * 3 + c]);
  const mid = [0, 1, 2].map((c) => px[(14 * 32 + 16) * 3 + c]);

  return (
    <section className="section" id="l3-imagem" aria-labelledby="l3-h-imagem">
      <div className="section-head">
        <h2 id="l3-h-imagem">1 · {pt ? 'Uma imagem é uma grade de números' : 'An image is a grid of numbers'}</h2>
      </div>
      <p className="lead" style={{ maxWidth: 'none' }}>
        {pt
          ? 'Nas aulas 1 e 2 a rede lia dois números por Pokémon: Velocidade e Defesa. Agora ela vai olhar a carta. Para o computador, a carta é uma grade de pixels, e cada pixel são três números de 0 a 255: quanto de vermelho, de verde e de azul.'
          : 'In lessons 1 and 2 the network read two numbers per Pokémon: Speed and Defense. Now it will look at the card. To the computer, the card is a grid of pixels, and each pixel is three numbers from 0 to 255: how much red, green and blue.'}
      </p>
      <CardPicker value={id} onChange={setId} data={data} />
      <div className="grid grid-2">
        <article className={`panel poke ${card.type}`}>
          <img src={`${BASE}${card.file}`} alt={`${card.name} card`} width={128} loading="lazy" />
          <div>
            <h3>{card.name}</h3>
            <span className={`type ${card.type}`}>{TYPE_NAME[lang][card.type]}</span>
            <p className="small">{pt ? 'Espremida em 32 × 32 pixels (é isso que a rede vê):' : 'Squeezed into 32 × 32 pixels (this is what the network sees):'}</p>
            <RgbCanvas t={x} px={112} title="32×32" />
          </div>
        </article>
        <article className="panel">
          <h3>{pt ? 'Três canais: vermelho, verde, azul' : 'Three channels: red, green, blue'}</h3>
          <div className="fmaps">
            {(['r', 'g', 'b'] as const).map((t, c) => (
              <figure key={t} style={{ margin: 0 }}>
                <MapCanvas data={channel(x, c)} h={32} w={32} px={96} tint={t} title={['R', 'G', 'B'][c]} />
                <figcaption className="small">{[pt ? 'vermelho' : 'red', pt ? 'verde' : 'green', pt ? 'azul' : 'blue'][c]}</figcaption>
              </figure>
            ))}
          </div>
          <div className="formula">{`${pt ? 'pixel (0, 0), o canto' : 'pixel (0, 0), the corner'}:   (R, G, B) = (${corner.join(', ')})
${pt ? 'pixel (14, 16), o meio' : 'pixel (14, 16), the middle'}: (R, G, B) = (${mid.join(', ')})
32 × 32 × 3 = 3072 ${pt ? 'números por carta' : 'numbers per card'}`}</div>
          <p className="small" style={{ margin: 0 }}>
            {pt
              ? 'A rede recebe cada número como x = valor/255 − 0,5, entre −0,5 e 0,5, como a aula 2 dividia os atributos por 100.'
              : 'The network gets each number as x = value/255 − 0.5, between −0.5 and 0.5, the way lesson 2 divided the stats by 100.'}
          </p>
        </article>
      </div>
      <div className="grid grid-2">
        <article className="panel">
          <h3>{pt ? 'Por que não usar o MLP da aula 2?' : 'Why not use the MLP from lesson 2?'}</h3>
          <ul className="arrows">
            <li>
              <b>{pt ? 'Pesos demais.' : 'Too many weights.'}</b>{' '}
              {pt
                ? 'Cada neurônio de uma camada densa liga-se aos 3072 números. Só a saída com 3 neurônios já tem 3 · 3072 + 3 = 9219 pesos; uma camada oculta de 100 neurônios teria 307 300.'
                : 'Every neuron in a dense layer connects to all 3072 numbers. Just the output with 3 neurons has 3 · 3072 + 3 = 9219 weights; a hidden layer of 100 neurons would have 307,300.'}
            </li>
            <li>
              <b>{pt ? 'Não sabe o que é vizinho.' : 'It does not know what is next to what.'}</b>{' '}
              {pt
                ? 'Para o MLP, o pixel 1 e o pixel 2 são entradas tão distantes quanto o 1 e o 3000. Se a carta andar um pixel para o lado, todas as entradas mudam e ele precisa reaprender.'
                : 'To the MLP, pixel 1 and pixel 2 are as unrelated as pixel 1 and pixel 3000. If the card moves one pixel sideways, every input changes and it has to learn again.'}
            </li>
          </ul>
        </article>
        <article className="panel panel--gold">
          <h3>{pt ? 'As duas ideias da CNN' : 'The two ideas of a CNN'}</h3>
          <ul className="arrows">
            <li>
              <b>{pt ? 'Conexão local:' : 'Local connections:'}</b>{' '}
              {pt ? 'cada neurônio olha só uma janelinha (por exemplo 5 × 5 pixels), onde estão as bordas, cores e texturas.' : 'each neuron looks at a small window only (say 5 × 5 pixels), where edges, colours and textures are.'}
            </li>
            <li>
              <b>{pt ? 'Pesos compartilhados:' : 'Shared weights:'}</b>{' '}
              {pt ? 'a mesma janelinha de pesos (o filtro) desliza pela imagem inteira. Uma chama no canto ou no meio é detectada pelo mesmo filtro.' : 'the same small window of weights (the filter) slides over the whole picture. A flame in the corner or in the middle is found by the same filter.'}
            </li>
          </ul>
          <p className="small" style={{ margin: 0 }}>
            {pt ? 'Resultado: a rede desta aula tem 1763 pesos, menos que a camada densa de 9219, e vê a carta inteira.' : 'Result: this lesson’s network has 1763 weights, fewer than the 9219 of the dense layer, and it still sees the whole card.'}
          </p>
        </article>
      </div>
    </section>
  );
}

// --------------------------------------------------------------------------- //
// 2. Convolution
// --------------------------------------------------------------------------- //
type KName = keyof typeof KERNELS;
const KLABEL: Record<KName, [string, string]> = {
  vertical: ['borda vertical', 'vertical edge'], horizontal: ['borda horizontal', 'horizontal edge'],
  blur: ['desfoque (média)', 'blur (average)'], sharpen: ['realce', 'sharpen'],
};

function Convolution() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [kname, setK] = useState<KName>('vertical');
  const [pad, setPad] = useState(0);
  const [stride, setStride] = useState(1);
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(false);
  const K = KERNELS[kname];
  const out = conv2dGrid(EDGE_PATCH, K, pad, stride);
  const oh = out.length, ow = out[0].length, n = oh * ow;
  const p = pos % n, oy = Math.floor(p / ow), ox = p % ow;
  const size = 6 + 2 * pad;
  const padded = Array.from({ length: size }, (_, y) => Array.from({ length: size }, (_, x) => {
    const sy = y - pad, sx = x - pad;
    return sy >= 0 && sy < 6 && sx >= 0 && sx < 6 ? EDGE_PATCH[sy][sx] : 0;
  }));
  const win = new Set<string>();
  const terms: string[] = [];
  for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
    const y = oy * stride + ky, x = ox * stride + kx;
    win.add(`${y},${x}`);
    terms.push(`${padded[y][x]}·${par(Number(K[ky][kx].toFixed(2)), lang)}`);
  }

  useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => setPos((v) => v + 1), 900);
    return () => window.clearInterval(t);
  }, [playing]);

  return (
    <section className="section" id="l3-conv" aria-labelledby="l3-h-conv">
      <div className="section-head">
        <h2 id="l3-h-conv">2 · {pt ? 'Convolução: o filtro que desliza' : 'Convolution: the sliding filter'}</h2>
      </div>
      <Note title={pt ? 'Um filtro é um perceptron que anda' : 'A filter is a perceptron that walks'}>
        {pt
          ? 'O filtro é uma grade de pesos, aqui 3 × 3. Em cada posição ele faz a mesma conta da aula 1: multiplica cada pixel da janela pelo peso que está em cima dele e soma tudo (net = Σ w·x). Depois anda uma casa e repete. Cada soma vira um quadrado do mapa de saída, o "mapa de características".'
          : 'The filter is a grid of weights, here 3 × 3. At each position it does the lesson 1 calculation: it multiplies each pixel in the window by the weight on top of it and adds everything up (net = Σ w·x). Then it moves one square and repeats. Each sum becomes one square of the output map, the "feature map".'}
      </Note>
      <div className="presets" role="radiogroup" aria-label={pt ? 'Filtro' : 'Filter'}>
        {(Object.keys(KERNELS) as KName[]).map((k) => (
          <button key={k} role="radio" aria-checked={kname === k} className={`btn btn--sm ${kname === k ? 'btn--gold' : 'btn--ghost'}`} onClick={() => setK(k)}>
            {KLABEL[k][pt ? 0 : 1]}
          </button>
        ))}
      </div>
      <div className="player">
        <button className="btn btn--sm" onClick={() => setPlaying((v) => !v)}>{playing ? (pt ? '❚❚ Pausar' : '❚❚ Pause') : (pt ? '▶ Deslizar' : '▶ Slide')}</button>
        <button className="btn btn--ghost btn--sm" onClick={() => { setPlaying(false); setPos((v) => (v % n) + n - 1); }}>◀</button>
        <button className="btn btn--ghost btn--sm" onClick={() => { setPlaying(false); setPos((v) => v + 1); }}>▶|</button>
        <label className="small">padding
          <select value={pad} onChange={(e) => { setPad(Number(e.target.value)); setPos(0); }} style={{ marginLeft: 6 }}>
            <option value={0}>0</option><option value={1}>1</option>
          </select>
        </label>
        <label className="small">stride
          <select value={stride} onChange={(e) => { setStride(Number(e.target.value)); setPos(0); }} style={{ marginLeft: 6 }}>
            <option value={1}>1</option><option value={2}>2</option>
          </select>
        </label>
      </div>
      <div className="grid grid-3 conv-row">
        <div className="panel">
          <span className="label">{pt ? `entrada ${size}×${size}` : `input ${size}×${size}`}{pad ? (pt ? ' (com zeros na borda)' : ' (zeros around the edge)') : ''}</span>
          <BlockGrid grid={padded} hi={win} pad={pad} digits={0} cell={size > 6 ? 30 : 34} title="input" />
        </div>
        <div className="panel">
          <span className="label">{pt ? 'filtro 3×3' : '3×3 filter'} · {KLABEL[kname][pt ? 0 : 1]}</span>
          <BlockGrid grid={K} signed digits={2} title="kernel" caption={kname === 'blur' ? (pt ? 'cada peso é 1/9 ≈ 0,11' : 'each weight is 1/9 ≈ 0.11') : undefined} />
          <p className="small">{pt ? 'Clique em um quadrado da saída para ver a conta dele.' : 'Click a square of the output to see its calculation.'}</p>
        </div>
        <div className="panel">
          <span className="label">{pt ? `saída ${oh}×${ow}` : `output ${oh}×${ow}`}</span>
          <BlockGrid grid={out} hi={new Set([`${oy},${ox}`])} signed digits={2} title="output"
            onPick={(y, x) => { setPlaying(false); setPos(y * ow + x); }} />
          <span className="label" style={{ marginTop: 10, display: 'block' }}>{pt ? 'depois da ReLU: max(0, net)' : 'after ReLU: max(0, net)'}</span>
          <BlockGrid grid={out.map((r) => r.map((v) => Math.max(0, v)))} digits={2} cell={28} title="relu" />
        </div>
      </div>
      <div className="formula">{`${pt ? 'posição' : 'position'} (${oy}, ${ox}):  net = ${terms.join(' + ')}
                = ${num(Number(out[oy][ox].toFixed(2)), lang)}`}</div>
      <div className="grid grid-2">
        <article className="panel">
          <h3>{pt ? 'Tamanho da saída' : 'Output size'}</h3>
          <div className="formula">{`${pt ? 'saída' : 'output'} = ⌊(n + 2p − k) / s⌋ + 1
       = ⌊(6 + 2·${pad} − 3) / ${stride}⌋ + 1 = ${outSize(6, 3, pad, stride)}`}</div>
          <ul className="arrows small">
            <li><b>padding p</b>: {pt ? 'zeros em volta da imagem. Com p = 1 e filtro 3×3, a saída fica do mesmo tamanho da entrada ("same").' : 'zeros around the picture. With p = 1 and a 3×3 filter, the output keeps the input size ("same").'}</li>
            <li><b>stride s</b>: {pt ? 'quantas casas o filtro pula. Com s = 2 a saída tem metade do lado.' : 'how many squares the filter jumps. With s = 2 the output has half the side.'}</li>
          </ul>
        </article>
        <article className="panel">
          <h3>{pt ? 'O que a borda vertical mostra' : 'What the vertical edge shows'}</h3>
          <p className="small">
            {pt
              ? 'O filtro soma a coluna da esquerda e subtrai a da direita. Onde a imagem é quase lisa (só 9 e 8, ou só 1 e 0) o resultado fica entre 1 e 5; onde o claro encontra o escuro ele dá 18 a 23. As duas colunas do meio da saída acendem: é ali que está a borda.'
              : 'The filter adds the left column and subtracts the right one. Where the picture is nearly flat (only 9s and 8s, or only 1s and 0s) the result is between 1 and 5; where light meets dark it gives 18 to 23. The two middle columns of the output light up: that is where the edge is.'}
          </p>
          <p className="small" style={{ margin: 0 }}>
            {pt
              ? 'Com cores, o filtro tem uma camada de pesos por canal (3 × 5 × 5 na primeira camada) e soma os três. Com 6 filtros saem 6 mapas.'
              : 'With colour, the filter has one layer of weights per channel (3 × 5 × 5 in the first layer) and adds the three up. With 6 filters you get 6 maps.'}
          </p>
        </article>
      </div>
      <CardFilter kname={kname} />
    </section>
  );
}

/** The same filter on a real card (grey levels, 120×168). */
function CardFilter({ kname }: { kname: KName }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [id, setId] = useState('charizard');
  const [grey, setGrey] = useState<number[][] | null>(null);
  useEffect(() => {
    const file = id === 'venusaur' ? 'pokemon/venusaur_card.png' : `pokemon/${id}_card.jpg`;
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas'); c.width = 120; c.height = 168;
      const ctx = c.getContext('2d'); if (!ctx) return;
      ctx.drawImage(img, 0, 0, 120, 168);
      const d = ctx.getImageData(0, 0, 120, 168).data;
      setGrey(Array.from({ length: 168 }, (_, y) => Array.from({ length: 120 }, (_, x) => {
        const i = (y * 120 + x) * 4;
        return (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
      })));
    };
    img.src = `${BASE}${file}`;
  }, [id]);
  const out = useMemo(() => {
    if (!grey) return null;
    const y = conv2dGrid(grey, KERNELS[kname], 1, 1);
    const edge = kname === 'vertical' || kname === 'horizontal';
    return y.flat().map((v) => (edge ? Math.abs(v) : v));
  }, [grey, kname]);
  const cards = [['charizard', 'Charizard'], ['blastoise', 'Blastoise'], ['venusaur', 'Venusaur']];
  return (
    <article className="panel">
      <h3>{pt ? 'O mesmo filtro em uma carta de verdade' : 'The same filter on a real card'}</h3>
      <div className="poke-pick">
        {cards.map(([k, n]) => <button key={k} className={id === k ? 'active' : ''} onClick={() => setId(k)}>{n}</button>)}
      </div>
      {grey && out ? (
        <div className="fmaps">
          <figure style={{ margin: 0 }}><MapCanvas data={grey.flat()} h={168} w={120} px={150} tint="gold" title="grey" /><figcaption className="small">{pt ? 'tons de cinza, 120×168' : 'grey levels, 120×168'}</figcaption></figure>
          <span className="arrow-big" aria-hidden="true">⟶</span>
          <figure style={{ margin: 0 }}><MapCanvas data={out} h={168} w={120} px={150} title={KLABEL[kname][1]} /><figcaption className="small">{KLABEL[kname][pt ? 0 : 1]}{kname === 'vertical' || kname === 'horizontal' ? ' (|net|)' : ''}</figcaption></figure>
        </div>
      ) : <p className="small">…</p>}
      <p className="small" style={{ margin: 0 }}>
        {pt
          ? 'Aqui os pesos do filtro foram escolhidos à mão. Na CNN ninguém escolhe: os pesos começam aleatórios e o backpropagation os ajusta, como na aula 2. Os filtros que ela aprende aparecem na parte 5.'
          : 'Here the filter weights were picked by hand. In a CNN nobody picks them: the weights start random and backpropagation adjusts them, as in lesson 2. The filters it learns are shown in part 5.'}
      </p>
    </article>
  );
}

// --------------------------------------------------------------------------- //
// 3. ReLU and pooling
// --------------------------------------------------------------------------- //
function Pooling() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [kind, setKind] = useState<PoolKind>('max');
  const out = pool2dGrid(POOL_PATCH, 2, 2, kind);
  const block = (y: number, x: number) => Math.floor(y / 2) * 2 + Math.floor(x / 2);
  const win = new Set<string>();
  if (kind !== 'avg') out.forEach((row, oy) => row.forEach((v, ox) => {
    for (let y = oy * 2; y < oy * 2 + 2; y++) for (let x = ox * 2; x < ox * 2 + 2; x++) if (POOL_PATCH[y][x] === v) win.add(`${y},${x}`);
  }));
  const blocks = [0, 1].flatMap((oy) => [0, 1].map((ox) => [POOL_PATCH[oy * 2][ox * 2], POOL_PATCH[oy * 2][ox * 2 + 1], POOL_PATCH[oy * 2 + 1][ox * 2], POOL_PATCH[oy * 2 + 1][ox * 2 + 1]]));
  const fn = kind === 'max' ? 'max' : kind === 'min' ? 'min' : (pt ? 'média' : 'mean');
  const delta = [[0.4, -0.2], [0.1, 0.3]];
  const back = maxPoolBack(POOL_PATCH, delta);

  return (
    <section className="section" id="l3-pool" aria-labelledby="l3-h-pool">
      <div className="section-head">
        <h2 id="l3-h-pool">3 · {pt ? 'ReLU e pooling: guardar o que importa, encolher o resto' : 'ReLU and pooling: keep what matters, shrink the rest'}</h2>
      </div>
      <div className="grid grid-2">
        <article className="panel">
          <h3>ReLU</h3>
          <div className="formula">ReLU(net) = max(0, net)</div>
          <p className="small" style={{ margin: 0 }}>
            {pt
              ? 'É a ativação das CNNs. Valores negativos ("não achei o padrão") viram 0 e os positivos passam iguais. A derivada é 1 onde net > 0 e 0 onde net ≤ 0, mais simples que a da sigmoide, e não encolhe o erro em cada camada. Você viu a ReLU na parte 2, embaixo da saída.'
              : 'It is the activation of CNNs. Negative values ("pattern not found") become 0 and positive ones pass unchanged. The derivative is 1 where net > 0 and 0 where net ≤ 0, simpler than the sigmoid’s, and it does not shrink the error at every layer. You saw ReLU in part 2, under the output.'}
          </p>
        </article>
        <article className="panel">
          <h3>Pooling</h3>
          <p className="small" style={{ margin: 0 }}>
            {pt
              ? 'O pooling corta o mapa em blocos 2 × 2 e troca cada bloco por um número. O lado cai pela metade e a quantidade de números cai para um quarto (28×28 → 14×14). Não tem pesos para aprender. Ele também dá tolerância: se a chama andar um pixel dentro do bloco, o máximo continua o mesmo.'
              : 'Pooling cuts the map into 2 × 2 blocks and replaces each block with one number. The side halves and the count of numbers drops to a quarter (28×28 → 14×14). It has no weights to learn. It also gives tolerance: if the flame moves one pixel inside the block, the maximum stays the same.'}
          </p>
        </article>
      </div>
      <div className="presets" role="radiogroup" aria-label="pooling">
        {(['max', 'avg', 'min'] as PoolKind[]).map((k) => (
          <button key={k} role="radio" aria-checked={kind === k} className={`btn btn--sm ${kind === k ? 'btn--gold' : 'btn--ghost'}`} onClick={() => setKind(k)}>
            {k === 'max' ? 'max pooling' : k === 'avg' ? (pt ? 'average pooling (média)' : 'average pooling') : 'min pooling'}
          </button>
        ))}
      </div>
      <div className="grid grid-3 conv-row">
        <div className="panel">
          <span className="label">{pt ? 'mapa 4×4, quatro blocos' : '4×4 map, four blocks'}</span>
          <BlockGrid grid={POOL_PATCH} group={block} hi={win} digits={0} cell={40} title="pool input" />
        </div>
        <div className="panel">
          <span className="label">{pt ? 'conta de cada bloco' : 'each block'}</span>
          <div className="formula">{blocks.map((b) => `${fn}(${b.join(', ')}) = ${num(kind === 'max' ? Math.max(...b) : kind === 'min' ? Math.min(...b) : b.reduce((a, v) => a + v, 0) / 4, lang)}`).join('\n')}</div>
        </div>
        <div className="panel">
          <span className="label">{pt ? 'saída 2×2' : '2×2 output'}</span>
          <BlockGrid grid={out} group={(y, x) => y * 2 + x} digits={2} cell={48} title="pool output" />
        </div>
      </div>
      <div className="grid grid-2">
        <article className="panel">
          <h3>{pt ? 'Qual usar?' : 'Which one?'}</h3>
          <ul className="arrows small">
            <li><b>max</b>: {pt ? 'guarda o sinal mais forte do bloco ("tem chama aqui?"). É o mais usado entre as convoluções e é o desta aula.' : 'keeps the strongest signal in the block ("is there a flame here?"). It is the most common between convolutions and the one in this lesson.'}</li>
            <li><b>average</b>: {pt ? 'suaviza. A versão "global" (média do mapa inteiro, um número por canal) substitui a camada densa no fim de redes como a ResNet.' : 'smooths. The "global" version (mean of the whole map, one number per channel) replaces the dense layer at the end of networks like ResNet.'}</li>
            <li><b>min</b>: {pt ? 'quase não se usa: depois da ReLU o mínimo costuma ser 0 e perde informação. Está aqui para comparar.' : 'rarely used: after ReLU the minimum is usually 0 and information is lost. It is here for comparison.'}</li>
            <li><b>stride 2</b>: {pt ? 'uma convolução com stride 2 também encolhe o mapa, mas com pesos aprendidos.' : 'a convolution with stride 2 also shrinks the map, with learned weights.'}</li>
          </ul>
        </article>
        <article className="panel">
          <h3>{pt ? 'E o erro, como volta?' : 'And how does the error come back?'}</h3>
          <p className="small">
            {pt
              ? `No max pooling só o vencedor de cada bloco influenciou a saída, então só ele recebe o erro. Com δ = (0,4, −0,2; 0,1, 0,3) vindo da camada de cima:`
              : `In max pooling only the winner of each block affected the output, so only it receives the error. With δ = (0.4, −0.2; 0.1, 0.3) coming from the layer above:`}
          </p>
          <BlockGrid grid={back} signed digits={2} cell={40} hi={new Set(['1,1', '0,3', '2,0', '3,2'])} title="pool backward" />
          <p className="small" style={{ margin: '6px 0 0' }}>
            {pt ? 'No average pooling, cada um dos 4 recebe δ/4.' : 'In average pooling, each of the 4 receives δ/4.'}
          </p>
        </article>
      </div>
    </section>
  );
}

// --------------------------------------------------------------------------- //
// 4. Architecture: big square → small square
// --------------------------------------------------------------------------- //
function FeatureStrip({ net, x }: { net: CardNet; x: Tensor }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const f = useMemo(() => forward(net, x), [net, x]);
  const maps = (t: Tensor, px: number, cols: number) => (
    <div className="fgrid" style={{ gridTemplateColumns: `repeat(${cols}, ${px}px)` }}>
      {Array.from({ length: t.c }, (_, c) => <MapCanvas key={c} data={channel(t, c)} h={t.h} w={t.w} px={px} title={`${pt ? 'canal' : 'channel'} ${c + 1}`} />)}
    </div>
  );
  return (
    <div className="fstrip">
      <figure><RgbCanvas t={x} px={96} title="input" /><figcaption>32×32×3</figcaption></figure>
      <span className="arrow-big" aria-hidden="true">⟶</span>
      <figure>{maps(f.a1, 46, 2)}<figcaption>conv1 + ReLU<br />28×28×6</figcaption></figure>
      <span className="arrow-big" aria-hidden="true">⟶</span>
      <figure>{maps(f.p1, 30, 2)}<figcaption>pool<br />14×14×6</figcaption></figure>
      <span className="arrow-big" aria-hidden="true">⟶</span>
      <figure>{maps(f.a2, 28, 2)}<figcaption>conv2 + ReLU<br />12×12×8</figcaption></figure>
      <span className="arrow-big" aria-hidden="true">⟶</span>
      <figure>{maps(f.p2, 18, 2)}<figcaption>pool<br />6×6×8</figcaption></figure>
      <span className="arrow-big" aria-hidden="true">⟶</span>
      <figure><MapCanvas data={f.flat} h={48} w={6} px={14} title="flatten" /><figcaption>288</figcaption></figure>
      <span className="arrow-big" aria-hidden="true">⟶</span>
      <figure style={{ minWidth: 190 }}><ProbBars p={f.p} names={names(lang)} /><figcaption>softmax</figcaption></figure>
    </div>
  );
}

function Architecture({ data }: { data: CnnData }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [id, setId] = useState('charizard');
  const rows: [string, string, string, string][] = [
    ['conv1', '3 → 6, 5×5', '6 · (3 · 5 · 5) + 6', '456'],
    ['pool1', '2×2, stride 2', pt ? 'sem pesos' : 'no weights', '0'],
    ['conv2', '6 → 8, 3×3', '8 · (6 · 3 · 3) + 8', '440'],
    ['pool2', '2×2, stride 2', pt ? 'sem pesos' : 'no weights', '0'],
    [pt ? 'densa' : 'dense', '288 → 3', '3 · 288 + 3', '867'],
  ];
  return (
    <section className="section" id="l3-arq" aria-labelledby="l3-h-arq">
      <div className="section-head">
        <h2 id="l3-h-arq">4 · {pt ? 'Do quadrado grande ao quadrado pequeno' : 'From the big square to the small square'}</h2>
        <SourceTag data={data} />
      </div>
      <p className="lead" style={{ maxWidth: 'none' }}>
        {pt
          ? 'A CNN alterna convolução (acha padrões) e pooling (encolhe). A cada passo o quadrado fica menor e mais fundo: menos pixels, mais canais. No fim, os 6 × 6 × 8 = 288 números viram uma fila e entram em uma camada densa com softmax, a mesma saída Fogo / Água / Planta da aula 2.'
          : 'A CNN alternates convolution (finds patterns) and pooling (shrinks). At every step the square gets smaller and deeper: fewer pixels, more channels. At the end, the 6 × 6 × 8 = 288 numbers are laid out in a row and go into a dense layer with softmax, the same Fire / Water / Grass output as lesson 2.'}
      </p>
      <div className="panel"><ArchDiagram /></div>
      <div className="grid grid-side">
        <article className="panel">
          <h3>{pt ? 'Quantos pesos' : 'How many weights'}</h3>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>{pt ? 'camada' : 'layer'}</th><th>{pt ? 'forma' : 'shape'}</th><th>{pt ? 'conta' : 'count'}</th><th>{pt ? 'pesos' : 'weights'}</th></tr></thead>
              <tbody>
                {rows.map((r) => <tr key={r[0] + r[1]}><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td className="gold">{r[3]}</td></tr>)}
                <tr className="current"><td colSpan={3}>total</td><td className="gold">{data.model.params}</td></tr>
              </tbody>
            </table>
          </div>
          <p className="small">
            {pt
              ? 'O "+ 6" e o "+ 8" são os vieses, um por filtro, como o θ das aulas anteriores. Cada filtro usa os mesmos 75 (ou 54) pesos em todas as posições: é isso que mantém a conta pequena.'
              : 'The "+ 6" and "+ 8" are the biases, one per filter, like the θ of the earlier lessons. Each filter uses the same 75 (or 54) weights at every position: that is what keeps the count small.'}
          </p>
          <div className="formula">{`conv1: 32 − 5 + 1 = 28 → pool: 28 / 2 = 14
conv2: 14 − 3 + 1 = 12 → pool: 12 / 2 = 6
6 × 6 × 8 = 288 → 3`}</div>
        </article>
        <article className="panel">
          <h3>{pt ? 'Por dentro da rede treinada' : 'Inside the trained network'}</h3>
          <p className="small">
            {pt
              ? 'Cada quadradinho é um mapa de características da carta escolhida (claro = valor alto). Os primeiros mapas ainda lembram a carta; os últimos são pequenos demais para nós, mas é com eles que a camada densa decide.'
              : 'Each small square is one feature map of the chosen card (bright = high value). The first maps still look like the card; the last ones are too small for us to read, but they are what the dense layer decides from.'}
          </p>
          <CardPicker value={id} onChange={setId} data={data} />
        </article>
      </div>
      <div className="panel"><FeatureStrip net={data.model.net} x={data.x.get(id)!} /></div>
    </section>
  );
}

function SourceTag({ data }: { data: CnnData }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  return (
    <span className="label">
      <span className={`led ${data.source === 'cache' ? 'led--ok' : ''}`} />{' '}
      {data.source === 'cache'
        ? (pt ? `modelo em cache · ${CNN_MODEL_PATH}` : `cached model · ${CNN_MODEL_PATH}`)
        : (pt ? 'cache indisponível · treinado no navegador' : 'cache unavailable · trained in the browser')}
    </span>
  );
}

// --------------------------------------------------------------------------- //
// 5. How the CNN learns
// --------------------------------------------------------------------------- //
function Learning({ data }: { data: CnnData }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const m = data.model;
  const y = conv2dGrid(BACK_X, BACK_K);
  const gK = kernelGrad(BACK_X, BACK_DELTA, 2);
  const gX = inputGrad(BACK_K, BACK_DELTA, 3, 3);
  const gKexpr = (a: number, b: number) => [0, 1].flatMap((i) => [0, 1].map((j) => `${par(BACK_DELTA[i][j], lang)}·${BACK_X[i + a][j + b]}`)).join(' + ');
  const curve = m.curve.map((c) => ({ epoch: c.epoch, loss: Math.max(c.loss, 1e-4), test: c.testAcc * 100, train: c.trainAcc * 100 }));
  const train = data.cards.filter((c) => c.split === 'train'), test = data.cards.filter((c) => c.split === 'test');
  const base = useMemo(() => {
    const { predictColor } = colorBaseline(train.map((c) => ({ x: data.x.get(c.id)!, y: CLASSES.indexOf(c.type) })));
    return test.filter((c) => CLASSES[predictColor(data.x.get(c.id)!)] === c.type).length;
  }, [data]); // eslint-disable-line react-hooks/exhaustive-deps
  const cnnHits = Math.round(m.test.accuracy * m.test.n);
  const miss = m.samples.filter((s) => s.split === 'test' && CLASSES[argmax(s.p)] !== s.type);
  const filters = Array.from({ length: m.net.conv1.f }, (_, f) => {
    const w = m.net.conv1.w.slice(f * 75, (f + 1) * 75), mx = Math.max(...w.map(Math.abs)) * 2;
    const t = tensor(3, 5, 5); t.d = w.map((v) => v / mx);
    return t;
  });
  const sets = new Set(data.cards.map((c) => c.set).filter((s) => s !== 'lesson')).size;

  return (
    <section className="section" id="l3-treino" aria-labelledby="l3-h-treino">
      <div className="section-head">
        <h2 id="l3-h-treino">5 · {pt ? 'Como a CNN aprende' : 'How the CNN learns'}</h2>
        <SourceTag data={data} />
      </div>
      <p className="lead" style={{ maxWidth: 'none' }}>
        {pt
          ? 'É o mesmo backpropagation da aula 2: o erro sai da softmax (δ = p − d), volta camada por camada pela regra da cadeia, e cada peso anda contra o seu gradiente. Só há três peças novas para atravessar no caminho de volta.'
          : 'It is the same backpropagation as in lesson 2: the error leaves the softmax (δ = p − d), goes back layer by layer with the chain rule, and each weight moves against its gradient. There are only three new pieces to cross on the way back.'}
      </p>
      <div className="grid grid-3">
        <article className="panel"><h3>1 · pooling</h3><p className="small" style={{ margin: 0 }}>{pt ? 'O erro vai só para o vencedor de cada bloco (parte 3).' : 'The error goes only to the winner of each block (part 3).'}</p></article>
        <article className="panel"><h3>2 · ReLU</h3><p className="small" style={{ margin: 0 }}>{pt ? 'Passa o erro onde net > 0 e zera onde net ≤ 0.' : 'Passes the error where net > 0 and zeroes it where net ≤ 0.'}</p></article>
        <article className="panel"><h3>3 · {pt ? 'convolução' : 'convolution'}</h3><p className="small" style={{ margin: 0 }}>{pt ? 'Como o peso foi usado em todas as janelas, o gradiente dele soma as contribuições de todas.' : 'Since the weight was used in every window, its gradient adds up the contributions from all of them.'}</p></article>
      </div>
      <article className="panel">
        <h3>{pt ? 'Backpropagation em uma convolução, à mão' : 'Backpropagation through a convolution, by hand'}</h3>
        <p className="small">
          {pt
            ? 'Entrada 3 × 3, filtro 2 × 2, saída 2 × 2. Suponha que o erro que chegou na saída seja δ (ele viria da camada de cima).'
            : 'A 3 × 3 input, a 2 × 2 filter, a 2 × 2 output. Suppose the error that reached the output is δ (it would come from the layer above).'}
        </p>
        <div className="fmaps" style={{ alignItems: 'flex-start' }}>
          <BlockGrid grid={BACK_X} digits={0} caption="x" title="x" />
          <span className="arrow-big">∗</span>
          <BlockGrid grid={BACK_K} signed digits={0} caption="K" title="K" />
          <span className="arrow-big">=</span>
          <BlockGrid grid={y} signed digits={0} caption={pt ? 'y (saída)' : 'y (output)'} title="y" />
          <span className="arrow-big">·</span>
          <BlockGrid grid={BACK_DELTA} signed digits={2} caption={pt ? 'δ (erro na saída)' : 'δ (error at the output)'} title="delta" />
        </div>
        <div className="formula">{`∂L/∂K[a][b] = Σ δ[i][j] · x[i + a][j + b]     (${pt ? 'a entrada "convoluída" com o erro' : 'the input "convolved" with the error'})

∂L/∂K[0][0] = ${gKexpr(0, 0)} = ${num(gK[0][0], lang)}
∂L/∂K[0][1] = ${gKexpr(0, 1)} = ${num(gK[0][1], lang)}
∂L/∂K[1][0] = ${gKexpr(1, 0)} = ${num(gK[1][0], lang)}
∂L/∂K[1][1] = ${gKexpr(1, 1)} = ${num(gK[1][1], lang)}

${pt ? 'novo K = K − η · ∂L/∂K;  com η = 0,1:  K[1][1] = 2 − 0,1·(−2,25) = 2,225' : 'new K = K − η · ∂L/∂K;  with η = 0.1:  K[1][1] = 2 − 0.1·(−2.25) = 2.225'}`}</div>
        <div className="fmaps" style={{ alignItems: 'flex-start' }}>
          <BlockGrid grid={gK} signed digits={2} cell={44} caption="∂L/∂K" title="dK" />
          <BlockGrid grid={gX} signed digits={2} cell={44} caption={pt ? '∂L/∂x: o erro que segue para a camada de baixo' : '∂L/∂x: the error that goes on to the layer below'} title="dx" />
        </div>
        <p className="small" style={{ margin: 0 }}>
          {pt
            ? 'Para ∂L/∂x, cada pixel recebe δ · K de cada janela de que participou. O canto x[0][0] só esteve em uma janela: 0,5 · 1 = 0,5. O meio x[1][1] esteve nas quatro: 0,5·2 + (−1)·0 + 0·(−1) + 0,25·1 = 1,25. tests/cnn.test.ts confere essas contas e o gradiente da rede inteira contra diferenças finitas.'
            : 'For ∂L/∂x, every pixel receives δ · K from each window it was part of. The corner x[0][0] was in one window only: 0.5 · 1 = 0.5. The middle x[1][1] was in all four: 0.5·2 + (−1)·0 + 0·(−1) + 0.25·1 = 1.25. tests/cnn.test.ts checks these numbers and the gradient of the whole network against finite differences.'}
        </p>
      </article>

      <div className="grid grid-side">
        <article className="panel">
          <h3>{pt ? 'O treino das cartas' : 'Training on the cards'}</h3>
          <ul className="arrows small">
            <li>{pt ? `${data.cards.length} cartas de ${sets} coleções do Pokémon TCG (1999 a 2023) mais as três das aulas: 31 de cada tipo.` : `${data.cards.length} cards from ${sets} Pokémon TCG sets (1999 to 2023) plus the three from the lessons: 31 of each type.`}</li>
            <li>{pt ? `${train.length} para treinar e ${test.length} guardadas para teste (8 por tipo). A rede nunca vê as de teste durante o treino.` : `${train.length} to train and ${test.length} kept for testing (8 per type). The network never sees the test cards while training.`}</li>
            <li>{pt ? 'Aumento de dados: a cada época, cada carta entra deslocada até 2 pixels, às vezes espelhada, com brilho e contraste sorteados. Assim 69 cartas parecem muitas mais.' : 'Data augmentation: every epoch, each card comes in shifted by up to 2 pixels, sometimes mirrored, with random brightness and contrast. That way 69 cards look like many more.'}</li>
            <li>{pt ? `SGD com momento: uma carta por vez, η = ${num(m.config.lr, lang)}, momento ${num(m.config.momentum, lang)}, ${m.config.epochs} épocas, semente ${m.config.seed}.` : `SGD with momentum: one card at a time, η = ${num(m.config.lr, lang)}, momentum ${num(m.config.momentum, lang)}, ${m.config.epochs} epochs, seed ${m.config.seed}.`}</li>
          </ul>
          <div className="formula">{`${pt ? 'momento' : 'momentum'}:  v ← ${num(m.config.momentum, lang)}·v − η·∂L/∂w
          w ← w + v`}</div>
        </article>
        <article className="panel">
          <h3>{pt ? 'Erro e acertos por época' : 'Loss and accuracy per epoch'}</h3>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer>
              <LineChart data={curve} margin={{ top: 10, right: 16, bottom: 22, left: 4 }}>
                <CartesianGrid stroke="#2a3547" strokeDasharray="3 3" />
                <XAxis dataKey="epoch" type="number" domain={[1, m.config.epochs]} stroke="#a9b6c8" fontSize={11}
                  label={{ value: pt ? 'época' : 'epoch', position: 'insideBottom', offset: -8, fill: '#a9b6c8', fontSize: 11 }} />
                <YAxis yAxisId="l" scale="log" domain={['auto', 'auto']} allowDataOverflow stroke="#ffd25e" fontSize={11} width={56} tickFormatter={(v) => fx(v, 3, lang)} />
                <YAxis yAxisId="a" orientation="right" domain={[0, 100]} stroke="#29d3e8" fontSize={11} unit="%" width={44} />
                <Tooltip contentStyle={{ background: '#0d1b2a', border: '1px solid #29d3e8', fontFamily: 'var(--font-mono)', fontSize: 12 }}
                  formatter={(v: number, n: string) => (n === 'loss' ? [fx(v, 4, lang), pt ? 'erro médio' : 'mean loss'] : [`${fx(v, 0, lang)}%`, n === 'test' ? (pt ? 'teste' : 'test') : (pt ? 'treino' : 'train')])}
                  labelFormatter={(e) => `${pt ? 'época' : 'epoch'} ${e}`} />
                <Legend wrapperStyle={{ paddingTop: 18 }} formatter={(v) => (v === 'loss' ? (pt ? 'erro médio (log)' : 'mean loss (log)') : v === 'test' ? (pt ? 'acertos no teste (%)' : 'test accuracy (%)') : (pt ? 'acertos no treino (%)' : 'train accuracy (%)'))} />
                <Line yAxisId="l" dataKey="loss" stroke="#ffd25e" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line yAxisId="a" dataKey="train" stroke="#7df0ff" strokeWidth={1.2} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
                <Line yAxisId="a" dataKey="test" stroke="#29d3e8" strokeWidth={1.8} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="small" style={{ margin: 0 }}>
            {pt
              ? `No fim: treino ${Math.round(m.train.accuracy * m.train.n)}/${m.train.n}, teste ${cnnHits}/${m.test.n}.`
              : `At the end: train ${Math.round(m.train.accuracy * m.train.n)}/${m.train.n}, test ${cnnHits}/${m.test.n}.`}
          </p>
        </article>
      </div>

      <div className="grid grid-2">
        <article className="panel">
          <h3>{pt ? 'Matriz de confusão (teste)' : 'Confusion matrix (test)'}</h3>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>{pt ? 'real \\ previsto' : 'actual \\ predicted'}</th>{CLASSES.map((c) => <th key={c} className={`t-${c}`}>{TYPE_NAME[lang][c]}</th>)}</tr></thead>
              <tbody>
                {m.test.confusion.map((row, i) => (
                  <tr key={i}><td className={`t-${CLASSES[i]}`}>{TYPE_NAME[lang][CLASSES[i]]}</td>{row.map((v, j) => <td key={j} className={i === j ? 'ok' : v ? 'bad' : ''}>{v}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
          {miss.map((s) => (
            <div key={s.id} className="poke grass" style={{ marginTop: 12 }}>
              <img src={`${BASE}${s.file}`} alt={`${s.name} card`} width={84} style={{ width: 84 }} loading="lazy" />
              <p className="small" style={{ margin: 0 }}>
                {pt
                  ? `O erro: ${s.name} é Planta, mas a rede deu ${fx(s.p[0] * 100, 0, lang)} % de Fogo. É uma carta de arte cheia: o casco vermelho do Pokémon cobre quase a carta toda, e a moldura verde de Planta quase não aparece.`
                  : `The miss: ${s.name} is Grass, but the network gave ${fx(s.p[0] * 100, 0, lang)} % Fire. It is a full-art card: the Pokémon’s red shell covers almost the whole card, and the green Grass frame barely shows.`}
              </p>
            </div>
          ))}
        </article>
        <article className="panel">
          <h3>{pt ? 'Os 6 filtros que a primeira camada aprendeu' : 'The 6 filters the first layer learned'}</h3>
          <div className="fmaps">
            {filters.map((t, f) => (
              <figure key={f} style={{ margin: 0 }}><RgbCanvas t={t} px={56} title={`filter ${f + 1}`} /><figcaption className="small">{f + 1}</figcaption></figure>
            ))}
          </div>
          <p className="small">
            {pt
              ? 'Cada filtro tem 3 × 5 × 5 pesos, pintados como cor (cinza = 0). Eles saíram do treino, ninguém os desenhou. São manchas ruidosas, e várias puxam para uma cor (verde, azul, vermelho): funcionam mais como detectores de cor do que de borda. A rede descobriu que a cor da carta é a pista mais fácil.'
              : 'Each filter has 3 × 5 × 5 weights, painted as colour (grey = 0). They came out of training; nobody drew them. They are noisy patches, and several lean towards one colour (green, blue, red): they work more as colour detectors than as edge detectors. The network found that the card’s colour is the easiest clue.'}
          </p>
        </article>
      </div>
      <Note title={pt ? 'Honestidade: o que ela aprendeu de verdade?' : 'Honestly: what did it really learn?'}>
        {pt
          ? `As cartas de cada tipo têm cor própria (moldura e fundo laranja para Fogo, azul para Água, verde para Planta). Um classificador sem convolução nenhuma, que só compara a cor média da carta com a cor média de cada tipo, acerta ${base} de ${test.length} no teste; a CNN acerta ${cnnHits}. Então a rede está lendo principalmente cor, e não o desenho do Pokémon. É um atalho válido para este conjunto, mas falharia em cartas de outro estilo ou em fotos do Pokémon sem a carta. Outro detalhe: nas primeiras coleções, Ekans e Koffing são do tipo Planta no jogo de cartas (no jogo de vídeo são Veneno), e a rede segue o rótulo da carta.`
          : `Cards of each type have their own colour (orange frame and background for Fire, blue for Water, green for Grass). A classifier with no convolution at all, which only compares the card’s average colour with the average colour of each type, gets ${base} of ${test.length} test cards right; the CNN gets ${cnnHits}. So the network is mostly reading colour, not the drawing of the Pokémon. That shortcut works for this set, but it would fail on cards of another style or on photos of the Pokémon without the card. One more detail: in the early sets, Ekans and Koffing are Grass type in the card game (Poison in the video games), and the network follows the card’s label.`}
      </Note>
      <details className="xnote">
        <summary>{pt ? `Ver as ${data.cards.length} cartas do conjunto` : `See all ${data.cards.length} cards in the set`}</summary>
        <div className="xbody">
          {CLASSES.map((t) => (
            <div key={t}>
              <h3 className={`t-${t}`}>{TYPE_NAME[lang][t]}</h3>
              <div className="deck">
                {data.cards.filter((c) => c.type === t).map((c) => (
                  <figure key={c.id} className={c.split}>
                    <img src={`${BASE}${c.file}`} alt={c.name} loading="lazy" width={70} />
                    <figcaption>{c.name}<br /><span className={c.split === 'test' ? 'gold' : 'muted'}>{c.split === 'test' ? (pt ? 'teste' : 'test') : (pt ? 'treino' : 'train')}</span></figcaption>
                  </figure>
                ))}
              </div>
            </div>
          ))}
          <p className="small">{pt ? 'Imagens das cartas: Pokémon TCG API (images.pokemontcg.io). © Nintendo, Creatures, GAME FREAK.' : 'Card images: Pokémon TCG API (images.pokemontcg.io). © Nintendo, Creatures, GAME FREAK.'}</p>
        </div>
      </details>
    </section>
  );
}

// --------------------------------------------------------------------------- //
// 6. Pokédex: discover the type of a new card
// --------------------------------------------------------------------------- //
function Pokedex({ data }: { data: CnnData }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const test = data.cards.filter((c) => c.split === 'test');
  const [pick, setPick] = useState<string>(test[0].id);
  const [upload, setUpload] = useState<{ url: string; x: Tensor; name: string } | null>(null);
  const [err, setErr] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => () => { if (upload) URL.revokeObjectURL(upload.url); }, [upload]);

  const card = data.cards.find((c) => c.id === pick)!;
  const x = upload ? upload.x : data.x.get(pick)!;
  const p = useMemo(() => forward(data.model.net, x).p, [data, x]);
  const best = argmax(p), guess = CLASSES[best];
  const right = upload ? null : guess === card.type;

  const onFile = (f: File | undefined) => {
    setErr('');
    if (!f) return;
    if (!f.type.startsWith('image/') || f.size > 15_000_000) { setErr(pt ? 'Use uma imagem (até 15 MB).' : 'Use an image (up to 15 MB).'); return; }
    const url = URL.createObjectURL(f), img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas'); c.width = 32; c.height = 32;
      const ctx = c.getContext('2d'); if (!ctx) return;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, 0, 0, 32, 32);
      const d = ctx.getImageData(0, 0, 32, 32).data, rgb = new Uint8Array(32 * 32 * 3);
      for (let i = 0; i < 32 * 32; i++) for (let k = 0; k < 3; k++) rgb[i * 3 + k] = d[i * 4 + k];
      setUpload({ url, x: toTensor(rgb), name: f.name });
    };
    img.onerror = () => setErr(pt ? 'Não consegui ler essa imagem.' : 'Could not read that image.');
    img.src = url;
  };

  return (
    <section className="section" id="l3-pokedex" aria-labelledby="l3-h-pokedex">
      <div className="section-head">
        <h2 id="l3-h-pokedex">6 · {pt ? 'Pokédex: descubra o tipo de um Pokémon novo' : 'Pokédex: discover the type of a new Pokémon'}</h2>
      </div>
      <p className="lead" style={{ maxWidth: 'none' }}>
        {pt
          ? 'Escolha uma das 24 cartas de teste, que a rede nunca viu, ou envie a foto de uma carta sua. A imagem é processada aqui no seu navegador; nada é enviado.'
          : 'Pick one of the 24 test cards, which the network never saw, or upload a photo of one of your cards. The picture is processed here in your browser; nothing is sent anywhere.'}
      </p>
      <div className="deck deck--pick" role="radiogroup" aria-label={pt ? 'Cartas de teste' : 'Test cards'}>
        {test.map((c) => (
          <button key={c.id} role="radio" aria-checked={!upload && pick === c.id} className={!upload && pick === c.id ? 'active' : ''}
            onClick={() => { setUpload(null); setPick(c.id); }} title={c.name}>
            <img src={`${BASE}${c.file}`} alt={c.name} loading="lazy" width={58} />
          </button>
        ))}
      </div>
      <div className="hero-actions" style={{ marginTop: 10 }}>
        <button className="btn btn--gold btn--sm" onClick={() => fileRef.current?.click()}>⬆ {pt ? 'Enviar a foto de uma carta' : 'Upload a card photo'}</button>
        <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ''; }} />
        {err && <span className="bad small">{err}</span>}
      </div>
      <div className="grid grid-side">
        <article className={`panel pokedex ${upload ? '' : card.type}`}>
          <span className="label">POKÉDEX · {pt ? 'escaneando' : 'scanning'}</span>
          <div className="poke" style={{ marginTop: 8 }}>
            <img src={upload ? upload.url : `${BASE}${card.file}`} alt={upload ? upload.name : card.name} width={150} style={{ width: 150 }} />
            <div>
              <h3>{upload ? upload.name : card.name}</h3>
              {!upload && <p className="small">{pt ? 'tipo real' : 'actual type'}: <span className={`type ${card.type}`}>{TYPE_NAME[lang][card.type]}</span></p>}
              <p className="verdict">
                {pt ? 'A Pokédex diz:' : 'The Pokédex says:'} <span className={`t-${guess}`}>{TYPE_NAME[lang][guess]}</span>{' '}
                ({fx(p[best] * 100, 1, lang)} %){' '}
                {right === null ? '' : right ? <span className="ok">✓</span> : <span className="bad">✗</span>}
              </p>
              <ProbBars p={p} names={names(lang)} />
            </div>
          </div>
        </article>
        <article className="panel">
          <h3>{pt ? 'Como ela decidiu' : 'How it decided'}</h3>
          <ul className="arrows small">
            <li>{pt ? 'A imagem é espremida em 32 × 32 pixels e cada número vira valor/255 − 0,5.' : 'The picture is squeezed into 32 × 32 pixels and each number becomes value/255 − 0.5.'}</li>
            <li>{pt ? 'Passa pelas duas convoluções, ReLU e poolings (veja os mapas abaixo).' : 'It goes through the two convolutions, ReLUs and poolings (see the maps below).'}</li>
            <li>{pt ? 'Os 288 números entram na camada densa; a softmax transforma as 3 saídas em probabilidades que somam 1.' : 'The 288 numbers go into the dense layer; softmax turns the 3 outputs into probabilities that add up to 1.'}</li>
            <li>{pt ? 'Fotos tortas, com fundo ou reflexo confundem: a rede foi treinada só com imagens limpas e retas. E como ela lê cor (parte 5), uma carta de Planta com fundo vermelho pode virar Fogo.' : 'Tilted photos, backgrounds or glare confuse it: it was trained only on clean, straight images. And since it reads colour (part 5), a Grass card with a red background may come out as Fire.'}</li>
            <li>{pt ? '100 % não é certeza. A softmax sempre escolhe um dos três tipos, mesmo para uma foto que não é carta, e esta rede, treinada até o erro quase zerar, ficou confiante demais. Ela não tem a opção "não sei".' : '100 % is not certainty. Softmax always picks one of the three types, even for a picture that is not a card, and this network, trained until the loss was almost zero, became overconfident. It has no "I don’t know" option.'}</li>
          </ul>
        </article>
      </div>
      <div className="panel"><FeatureStrip net={data.model.net} x={x} /></div>
    </section>
  );
}

// --------------------------------------------------------------------------- //
// 7. Other techniques
// --------------------------------------------------------------------------- //
function Techniques() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const items: [string, string, string][] = [
    ['Dilation', 'Convolução dilatada: o filtro 3×3 pula pixels (por exemplo um sim, um não) e cobre 5×5 com os mesmos 9 pesos. Vê mais longe sem encolher o mapa.',
      'Dilated convolution: the 3×3 filter skips pixels (say every other one) and covers 5×5 with the same 9 weights. It sees further without shrinking the map.'],
    ['1×1', 'Convolução 1×1: mistura os canais de cada pixel sem olhar os vizinhos. Serve para reduzir ou aumentar o número de canais de forma barata.',
      '1×1 convolution: mixes the channels of each pixel without looking at neighbours. It is a cheap way to cut or raise the number of channels.'],
    ['Batch normalization', 'Normaliza a saída de cada canal para média 0 e desvio 1 dentro do lote, com uma escala e um deslocamento aprendidos. O treino fica mais estável e aceita η maior.',
      'Normalizes each channel’s output to mean 0 and standard deviation 1 within the batch, with a learned scale and shift. Training becomes more stable and accepts a larger η.'],
    ['Dropout', 'No treino, desliga neurônios ao acaso (por exemplo metade deles) a cada passo. A rede não pode depender de um só e decora menos.',
      'During training, switches neurons off at random (say half of them) at every step. The network cannot rely on any single one and memorizes less.'],
    [pt ? 'Aumento de dados' : 'Data augmentation', 'Deslocar, espelhar, girar, recortar e mudar o brilho das imagens de treino. Usamos deslocamento, espelho e brilho nesta aula.',
      'Shifting, mirroring, rotating, cropping and changing the brightness of the training images. This lesson uses shift, mirror and brightness.'],
    ['Global average pooling', 'Tira a média de cada mapa inteiro: um número por canal. Substitui o "achatar + densa" e reduz muito os pesos.',
      'Takes the mean of each whole map: one number per channel. It replaces "flatten + dense" and cuts the weight count a lot.'],
    [pt ? 'Conexões residuais (ResNet)' : 'Residual connections (ResNet)', 'A saída de um bloco é x + F(x): o sinal pode pular camadas. Foi isso que permitiu treinar redes com mais de 100 camadas (He et al., 2015).',
      'A block’s output is x + F(x): the signal can skip layers. That is what made it possible to train networks with over 100 layers (He et al., 2015).'],
    [pt ? 'Transferência de aprendizado' : 'Transfer learning', 'Pegar uma rede já treinada em milhões de fotos (ImageNet) e treinar de novo só a última camada para as suas classes. Com poucas cartas, é o caminho prático para ler o desenho e não só a cor.',
      'Take a network already trained on millions of photos (ImageNet) and retrain only the last layer for your classes. With few cards, it is the practical way to read the drawing and not just the colour.'],
    [pt ? 'Arquiteturas famosas' : 'Famous architectures', 'LeNet-5 (LeCun et al., 1998) lia dígitos escritos à mão com uma estrutura parecida com a desta aula. AlexNet (2012) venceu o ImageNet com GPUs e ReLU. VGG (2014) empilhou filtros 3×3. ResNet (2015) trouxe as conexões residuais.',
      'LeNet-5 (LeCun et al., 1998) read handwritten digits with a structure much like this lesson’s. AlexNet (2012) won ImageNet with GPUs and ReLU. VGG (2014) stacked 3×3 filters. ResNet (2015) brought residual connections.'],
    [pt ? 'Validação separada' : 'A separate validation set', 'Aqui escolhemos η e o número de épocas olhando um treino só. O correto é separar também um conjunto de validação para escolher hiperparâmetros e deixar o teste intocado até o fim; com 93 cartas, preferimos simplicidade.',
      'Here η and the number of epochs were picked from a single run. The proper way is to also set aside a validation set to choose hyperparameters and leave the test set untouched until the end; with 93 cards, we kept it simple.'],
  ];
  return (
    <section className="section" id="l3-tecnicas" aria-labelledby="l3-h-tecnicas">
      <div className="section-head">
        <h2 id="l3-h-tecnicas">7 · {pt ? 'Outras técnicas que você vai encontrar' : 'Other techniques you will run into'}</h2>
      </div>
      {items.map(([t, p, e]) => (
        <details key={t} className="xnote">
          <summary>{t}</summary>
          <div className="xbody"><p>{pt ? p : e}</p></div>
        </details>
      ))}
    </section>
  );
}
