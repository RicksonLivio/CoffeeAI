import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { assets, loadCachedMlp, type Loaded } from '../../api';
import { fx, num, par, pfx, useLang, type Lang } from '../../i18n';
import {
  backprop, fireWaterHull, HAND_NET, insideConvex, POKE_NAME, predict, SCALE, sgdStep, softForward, STARTERS,
  stepForward, TYPES, loss, type Net, type Poke, type PokeType,
} from '../../lib/mlp';
import { classify, train, type Sample } from '../../lib/perceptron';
import type { CachedMlp } from '../../lib/model';
import { sumExpr } from './CalcPanel';
import MlpGraph, { type Pulse } from './MlpGraph';
import Note from './Note';
import RegionChart, { type RegionPoint } from './RegionChart';

const TYPE_NAME: Record<Lang, Record<PokeType, string>> = {
  pt: { fire: 'Fogo', water: 'Água', grass: 'Planta' },
  en: { fire: 'Fire', water: 'Water', grass: 'Grass' },
};
const PLANE: [number, number] = [35, 110];
const PLANE_TICKS = [40, 50, 60, 70, 80, 90, 100, 110];

const toPoints = (lang: Lang, info?: (s: Poke) => string): RegionPoint[] =>
  STARTERS.map((s) => ({ id: s.id, name: POKE_NAME[s.id], x: s.x1, y: s.x2, type: s.type, info: info?.(s) ?? `x1 = ${s.x1}, x2 = ${s.x2} · ${TYPE_NAME[lang][s.type]}` }));

function PokePicker({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  return (
    <div className="poke-pick" role="radiogroup" aria-label="Pokémon">
      {STARTERS.map((s) => (
        <button key={s.id} role="radio" aria-checked={value === s.id} className={`${s.type} ${value === s.id ? 'active' : ''}`} onClick={() => onChange(s.id)}>
          {POKE_NAME[s.id]}
        </button>
      ))}
    </div>
  );
}

export default function Lesson2() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [cached, setCached] = useState<Loaded<CachedMlp> | null>(null);
  useEffect(() => {
    loadCachedMlp().then(setCached);
  }, []);
  const jump = (id: string) => (e: React.MouseEvent) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView(); };

  return (
    <div className="lesson-body">
      <div className="parts">
        <a href="#l2-problema" onClick={jump('l2-problema')}><b>1</b>{pt ? 'O problema do tipo Planta' : 'The Grass-type problem'}</a>
        <a href="#l2-oculta" onClick={jump('l2-oculta')}><b>2</b>{pt ? 'A camada oculta' : 'The hidden layer'}</a>
        <a href="#l2-aprende" onClick={jump('l2-aprende')}><b>3</b>{pt ? 'Como o MLP aprende' : 'How the MLP learns'}</a>
      </div>
      <Problem />
      <HiddenLayer />
      <section className="section" id="l2-aprende" aria-labelledby="l2-h-aprende">
        <div className="section-head">
          <h2 id="l2-h-aprende">3 · {pt ? 'Como o MLP aprende: backpropagation' : 'How the MLP learns: backpropagation'}</h2>
          {cached && (
            <span className="label">
              <span className={`led ${cached.source === 'cache' ? 'led--ok' : ''}`} />{' '}
              {cached.source === 'cache'
                ? (pt ? `modelo em cache · ${cached.path}` : `cached model · ${cached.path}`)
                : (pt ? 'cache indisponível · treinado no navegador' : 'cache unavailable · trained in the browser')}
            </span>
          )}
        </div>
        {cached ? <Learning model={cached.model} /> : <div className="panel">{pt ? 'Carregando o modelo…' : 'Loading the model…'}</div>}
      </section>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// 1. The problem
// --------------------------------------------------------------------------- //
function Problem() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const v = classify({ w1: 22, w2: -22, theta: 0.5 }, 80, 83);
  const hull = fireWaterHull();
  const inside = STARTERS.filter((s) => s.type === 'grass' && insideConvex(hull, [s.x1, s.x2])).map((s) => POKE_NAME[s.id]);
  const grassVsRest = useMemo(() => {
    const samples: Sample[] = STARTERS.map((s) => ({ id: s.id, x1: s.x1, x2: s.x2, d: s.type === 'grass' ? 1 : 0 }));
    return train(samples, { w1: 0, w2: 0, theta: 0.5 }, 1, 100);
  }, []);
  const errs = grassVsRest.epochs.map((e) => e.errors);

  return (
    <section className="section" id="l2-problema" aria-labelledby="l2-h-problema">
      <div className="section-head">
        <h2 id="l2-h-problema">1 · {pt ? 'O problema: Venusaur não é Água' : 'The problem: Venusaur is not Water'}</h2>
      </div>
      <div className="grid grid-side">
        <article className="panel poke grass">
          <img src={assets.card('venusaur')} alt="Venusaur card" width={128} loading="lazy" />
          <div>
            <h3>Venusaur</h3>
            <span className="type grass">{TYPE_NAME[lang].grass}</span>
            <p className="statline">x1 · {pt ? 'Velocidade' : 'Speed'} <b>80</b> · x2 · {pt ? 'Defesa' : 'Defense'} <b>83</b></p>
            <div className="formula">{`net = 22·80 + (−22)·83 + 0,5·(−1)
    = 1760 − 1826 − 0,5 = ${num(v.net, lang)}  ⟹  y = ${v.y} (${TYPE_NAME[lang].water})`.replace(/0,5/g, num(0.5, lang))}</div>
            <p className="small" style={{ margin: 0 }}>
              {pt
                ? 'O perceptron da Aula 1 chama o Venusaur de Água. Ele não está "errando uma conta": com uma única saída 0 ou 1, ele só sabe dizer Fogo ou Água.'
                : 'The lesson 1 perceptron calls Venusaur Water. It is not "getting a sum wrong": with a single 0 or 1 output, it can only say Fire or Water.'}
            </p>
          </div>
        </article>
        <article className="panel">
          <h3>{pt ? 'Dois obstáculos' : 'Two obstacles'}</h3>
          <ul className="arrows">
            <li>
              <b>{pt ? 'Uma saída, duas respostas.' : 'One output, two answers.'}</b>{' '}
              {pt
                ? 'Para três tipos precisamos de três saídas, uma por tipo; ganha a que acender. Isso é fácil de resolver.'
                : 'For three types we need three outputs, one per type; the one that lights up wins. That part is easy.'}
            </li>
            <li>
              <b>{pt ? 'Planta fica no meio.' : 'Grass sits in the middle.'}</b>{' '}
              {pt
                ? 'Este é o problema de verdade. Olhe a diferença Velocidade − Defesa dos três iniciais de cada tipo: Fogo tem +22, Água tem −22, e Planta fica entre −4 e −3.'
                : 'This is the real problem. Look at Speed − Defense for the three starters of each type: Fire has +22, Water has −22, and Grass sits between −4 and −3.'}
            </li>
          </ul>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>Pokémon</th><th>{pt ? 'Tipo' : 'Type'}</th><th>x1</th><th>x2</th><th>x1 − x2</th></tr></thead>
              <tbody>
                {STARTERS.map((s) => (
                  <tr key={s.id}>
                    <td>{POKE_NAME[s.id]}</td><td className={`t-${s.type}`}>{TYPE_NAME[lang][s.type]}</td>
                    <td>{s.x1}</td><td>{s.x2}</td><td>{num(s.x1 - s.x2, lang)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </div>

      <div className="grid grid-2" style={{ marginTop: 20 }}>
        <article className="panel">
          <h3>{pt ? 'Os nove iniciais' : 'The nine starters'}</h3>
          <RegionChart x={PLANE} y={PLANE} ticks={PLANE_TICKS} xLabel={pt ? 'x1 · Velocidade' : 'x1 · Speed'} yLabel={pt ? 'x2 · Defesa' : 'x2 · Defense'}
            points={toPoints(lang)} polygon={hull} />
          <div className="legend">
            <span><i style={{ background: '#ff7a2f', borderRadius: '50%' }} />{TYPE_NAME[lang].fire}</span>
            <span><i style={{ background: '#2f7bff' }} />{TYPE_NAME[lang].water}</span>
            <span><i style={{ background: '#4cc46a' }} />{TYPE_NAME[lang].grass}</span>
            <span><i style={{ border: '1px dashed #a9b6c8' }} />{pt ? 'área cercada por Fogo e Água' : 'area enclosed by Fire and Water'}</span>
          </div>
        </article>
        <article className="panel">
          <h3>{pt ? 'Por que uma reta não basta' : 'Why one line is not enough'}</h3>
          <p>
            {pt
              ? `Para um único neurônio responder "Planta", precisamos de uma reta com todos os Planta de um lado e todos os Fogo e Água do outro. Mas ${inside.join(' e ')} estão dentro da área tracejada, cujos cantos são Charmander, Charizard, Blastoise e Squirtle. Qualquer reta que deixe esses quatro cantos do mesmo lado deixa a área inteira desse lado, e com ela ${inside.join(' e ')}.`
              : `For a single neuron to answer "Grass", we need a line with every Grass type on one side and every Fire and Water type on the other. But ${inside.join(' and ')} are inside the dashed area, whose corners are Charmander, Charizard, Blastoise and Squirtle. Any line that keeps those four corners on the same side keeps the whole area on that side, and ${inside.join(' and ')} with it.`}
          </p>
          <p>
            {pt
              ? `Teste prático: treinei um perceptron "Planta ou não" com a regra da Aula 1 por ${grassVsRest.epochs.length} épocas. Erros nas primeiras épocas: ${errs.slice(0, 8).join(', ')}… O menor valor em todas as épocas foi ${Math.min(...errs)}, nunca zero.`
              : `Hands-on check: I trained a "Grass or not" perceptron with the lesson 1 rule for ${grassVsRest.epochs.length} epochs. Errors in the first epochs: ${errs.slice(0, 8).join(', ')}… The lowest value across all epochs was ${Math.min(...errs)}, never zero.`}
          </p>
          <Note title={pt ? 'Linearmente separável' : 'Linearly separable'}>
            {pt
              ? 'Duas classes são linearmente separáveis quando existe uma reta (no plano) que deixa cada classe de um lado. Fogo contra Água é; Planta contra o resto não é. É o mesmo obstáculo do famoso problema XOR, que Minsky e Papert usaram em 1969 para mostrar o limite do perceptron de uma camada.'
              : 'Two classes are linearly separable when a line (in the plane) leaves each class on one side. Fire versus Water is; Grass versus the rest is not. It is the same obstacle as the famous XOR problem, which Minsky and Papert used in 1969 to show the limit of the single-layer perceptron.'}
          </Note>
        </article>
      </div>
    </section>
  );
}

// --------------------------------------------------------------------------- //
// 2. Hidden layer by hand
// --------------------------------------------------------------------------- //
function HiddenLayer() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [id, setId] = useState('venusaur');
  const [play, setPlay] = useState<{ id: number; kind: Pulse }>({ id: 0, kind: 'forward' });
  const s = STARTERS.find((q) => q.id === id)!;
  const f = stepForward(HAND_NET, s.x1, s.x2);
  const T = TYPE_NAME[lang];
  const outNames = [T.fire, T.water, T.grass];

  const handType = (x1: number, x2: number): PokeType => {
    const r = stepForward(HAND_NET, x1, x2);
    return r.type ?? TYPES[r.outputNet.indexOf(Math.max(...r.outputNet))];
  };
  const hiddenType = (h1: number, h2: number): PokeType => {
    const o = HAND_NET.output.map((n) => n.w[0] * h1 + n.w[1] * h2 - n.theta);
    return TYPES[o.indexOf(Math.max(...o))];
  };
  const jitter = [-0.05, 0, 0.05];
  const hiddenPoints: RegionPoint[] = STARTERS.map((q, i) => {
    const r = stepForward(HAND_NET, q.x1, q.x2);
    return { id: q.id, name: POKE_NAME[q.id], type: q.type, x: r.h[0] + jitter[i % 3], y: r.h[1] + jitter[(i + 1) % 3], info: `h1 = ${r.h[0]}, h2 = ${r.h[1]}` };
  });

  const hidRows = HAND_NET.hidden.map((n, j) => {
    const prods = [n.w[0] * s.x1, n.w[1] * s.x2, -n.theta];
    return { name: `h${j + 1}`, n, prods, net: f.hiddenNet[j], out: f.h[j] };
  });
  const outRows = HAND_NET.output.map((n, k) => {
    const prods = [n.w[0] * f.h[0], n.w[1] * f.h[1], -n.theta];
    return { name: outNames[k], n, prods, net: f.outputNet[k], out: f.o[k] };
  });

  return (
    <section className="section" id="l2-oculta" aria-labelledby="l2-h-oculta">
      <div className="section-head">
        <h2 id="l2-h-oculta">2 · {pt ? 'A ideia: uma camada oculta' : 'The idea: a hidden layer'}</h2>
        <span className="label">{pt ? 'pesos escolhidos à mão' : 'weights chosen by hand'}</span>
      </div>
      <div className="grid grid-side">
        <article className="panel">
          <h3>{pt ? 'Duas retas em vez de uma' : 'Two lines instead of one'}</h3>
          <p>
            {pt
              ? 'Se uma reta não separa o meio, usamos duas. Cada reta vira um neurônio da camada oculta, e uma segunda camada combina as respostas deles:'
              : 'If one line cannot separate the middle, we use two. Each line becomes a neuron of the hidden layer, and a second layer combines their answers:'}
          </p>
          <ul className="arrows">
            <li><span className="var">h1</span> · {pt ? 'detector de Fogo: responde 1 se Velocidade − Defesa ≥ 11' : 'Fire detector: answers 1 if Speed − Defense ≥ 11'}</li>
            <li><span className="var">h2</span> · {pt ? 'detector de Água: responde 1 se Defesa − Velocidade ≥ 11' : 'Water detector: answers 1 if Defense − Speed ≥ 11'}</li>
            <li>{T.fire} = h1 · {T.water} = h2 · {T.grass} = {pt ? 'nem h1 nem h2' : 'neither h1 nor h2'}</li>
          </ul>
          <div className="formula">{pt
            ? `h1:  w = (1, −1),   θ = 11    net = x1 − x2 − 11
h2:  w = (−1, 1),   θ = 11    net = x2 − x1 − 11
Fogo:    w = (1, 0),   θ = ${num(0.5, lang)}    net = h1 − ${num(0.5, lang)}
Água:    w = (0, 1),   θ = ${num(0.5, lang)}    net = h2 − ${num(0.5, lang)}
Planta:  w = (−1, −1), θ = ${num(-0.5, lang)}   net = ${num(0.5, lang)} − h1 − h2`
            : `h1:  w = (1, −1),   θ = 11    net = x1 − x2 − 11
h2:  w = (−1, 1),   θ = 11    net = x2 − x1 − 11
Fire:    w = (1, 0),   θ = 0.5    net = h1 − 0.5
Water:   w = (0, 1),   θ = 0.5    net = h2 − 0.5
Grass:   w = (−1, −1), θ = −0.5   net = 0.5 − h1 − h2`}</div>
          <Note title={pt ? 'Por que 11?' : 'Why 11?'}>
            {pt
              ? 'É metade de 22: a reta de h1 fica no meio do caminho entre os Fogo (Velocidade − Defesa = 22) e o ponto em que Velocidade = Defesa. Os Planta têm Defesa − Velocidade de 3 ou 4, então qualquer valor acima de 4 e até 22 funcionaria nestes nove Pokémon; 11 deixa folga dos dois lados.'
              : 'It is half of 22: the h1 line sits halfway between the Fire types (Speed − Defense = 22) and the point where Speed = Defense. The Grass types have Defense − Speed of 3 or 4, so any value above 4 and up to 22 would work for these nine Pokémon; 11 leaves room on both sides.'}
          </Note>
          <Note title={pt ? 'Por que "oculta"?' : 'Why "hidden"?'}>
            {pt
              ? 'Porque os dados só trazem as entradas e o tipo; ninguém diz o que h1 e h2 devem calcular. Aqui eu escolhi os pesos para você ver a ideia. Na parte 3, a rede descobre pesos parecidos sozinha.'
              : 'Because the data only has the inputs and the type; nobody says what h1 and h2 should compute. Here I chose the weights so you can see the idea. In part 3, the network finds similar weights on its own.'}
          </Note>
        </article>
        <article className="panel">
          <h3>{pt ? 'Passo a passo' : 'Step by step'}</h3>
          <PokePicker value={id} onChange={(v) => { setId(v); setPlay((p) => ({ id: p.id + 1, kind: 'forward' })); }} />
          <div className="calc">
            <div className="block on">
              <div className="title">1 · {pt ? 'Camada oculta' : 'Hidden layer'} · x1 = {s.x1}, x2 = {s.x2}</div>
              {hidRows.map((r) => (
                <div key={r.name} className="line">
                  {r.name}: net = {num(r.n.w[0], lang)}·{s.x1} + {par(r.n.w[1], lang)}·{s.x2} + {num(r.n.theta, lang)}·(−1) = {sumExpr(r.prods, lang)} ={' '}
                  <span className="res">{num(r.net, lang)}</span> ⟹ {r.name} = <span className="res">{r.out}</span>
                </div>
              ))}
            </div>
            <div className="block on">
              <div className="title">2 · {pt ? 'Camada de saída' : 'Output layer'} · h1 = {f.h[0]}, h2 = {f.h[1]}</div>
              {outRows.map((r) => (
                <div key={r.name} className="line">
                  {r.name}: net = {sumExpr(r.prods, lang)} = <span className="res">{num(r.net, lang)}</span> ⟹ <span className={r.out ? 'ok' : 'muted'}>{r.out}</span>
                </div>
              ))}
            </div>
            <div className={`line verdict ${f.type === s.type ? 'ok' : 'bad'}`}>
              {POKE_NAME[s.id]} → {f.type ? T[f.type] : '?'} {f.type === s.type ? '✓' : '✗'}
            </div>
          </div>
        </article>
      </div>

      <div className="panel" style={{ marginTop: 20 }}>
        <div className="player">
          <button className="btn btn--gold btn--sm" onClick={() => setPlay((p) => ({ id: p.id + 1, kind: 'forward' }))}>▶ {pt ? 'Propagar' : 'Forward'}</button>
          <span className="small muted">{pt ? 'Rede com os pesos escolhidos à mão, mostrando' : 'Network with the hand-picked weights, showing'} {POKE_NAME[s.id]}</span>
        </div>
        <MlpGraph net={HAND_NET} x={[s.x1, s.x2]} h={f.h} o={f.o} winner={f.type ? TYPES.indexOf(f.type) : null} play={play} digits={1} />
        <div className="legend" style={{ justifyContent: 'center' }}>
          <span><i style={{ background: '#29d3e8' }} />{pt ? 'peso positivo' : 'positive weight'}</span>
          <span><i style={{ background: '#7b4dff' }} />{pt ? 'peso negativo' : 'negative weight'}</span>
          <span><i style={{ border: '1px dashed #5b6b82' }} />{pt ? 'peso zero' : 'zero weight'}</span>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: 20 }}>
        <article className="panel">
          <h3>{pt ? 'No plano original' : 'In the original plane'}</h3>
          <RegionChart x={PLANE} y={PLANE} ticks={PLANE_TICKS} xLabel={pt ? 'x1 · Velocidade' : 'x1 · Speed'} yLabel={pt ? 'x2 · Defesa' : 'x2 · Defense'}
            points={toPoints(lang)} region={handType}
            segments={[
              { a: [46, 35], b: [110, 99], color: '#ff7a2f', label: 'h1: x1 − x2 = 11' },
              { a: [35, 46], b: [99, 110], color: '#6fa4ff', label: 'h2: x2 − x1 = 11' },
            ]} />
          <p className="small muted" style={{ margin: 0 }}>
            {pt ? 'As duas retas criam uma faixa no meio: a faixa é Planta.' : 'The two lines create a band in the middle: the band is Grass.'}
          </p>
        </article>
        <article className="panel">
          <h3>{pt ? 'No espaço da camada oculta' : 'In the hidden-layer space'}</h3>
          <RegionChart x={[-0.25, 1.25]} y={[-0.25, 1.25]} ticks={[0, 0.5, 1]} xLabel="h1" yLabel="h2" points={hiddenPoints} region={hiddenType} cells={30} />
          <p className="small muted" style={{ margin: 0 }}>
            {pt
              ? 'Cada Pokémon vira o par (h1, h2): Fogo em (1, 0), Água em (0, 1), Planta em (0, 0). Os pontos foram levemente afastados para não ficarem um em cima do outro.'
              : 'Each Pokémon becomes the pair (h1, h2): Fire at (1, 0), Water at (0, 1), Grass at (0, 0). The points are nudged apart slightly so they do not overlap.'}
          </p>
        </article>
      </div>
      <Note title={pt ? 'O truque do MLP' : 'The MLP trick'}>
        {pt
          ? 'A camada oculta cria novas variáveis. No plano (Velocidade, Defesa) o tipo Planta está no meio e nenhuma reta o isola; no plano (h1, h2) os três tipos viram três cantos, e aí retas simples separam tudo. Cada camada transforma o problema até ele ficar fácil para a camada seguinte.'
          : 'The hidden layer creates new variables. In the (Speed, Defense) plane Grass is in the middle and no line isolates it; in the (h1, h2) plane the three types become three corners, and simple lines separate everything. Each layer reshapes the problem until it is easy for the next one.'}
      </Note>
    </section>
  );
}

// --------------------------------------------------------------------------- //
// 3. Learning: backpropagation, one step by hand and the cached training run
// --------------------------------------------------------------------------- //
const flatNames = (net: Net) => ({
  hidden: net.hidden.map((_, j) => [`w${j + 1},1`, `w${j + 1},2`, `θ${j + 1}`]),
  output: net.output.map((_, k) => [`v${k + 1},1`, `v${k + 1},2`, `φ${k + 1}`]),
});

function Learning({ model }: { model: CachedMlp }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const T = TYPE_NAME[lang];
  const outNames = [T.fire, T.water, T.grass];
  const eta = model.config.eta;
  const venusaur = STARTERS.find((s) => s.id === 'venusaur')!;
  const init = model.init;
  const b = useMemo(() => backprop(init, venusaur), [init, venusaur]);
  const after = useMemo(() => sgdStep(init, b.grad, eta), [init, b, eta]);
  const fAfter = softForward(after, venusaur.x1, venusaur.x2);
  const lossAfter = loss(fAfter, 'grass');
  const names = flatNames(init);
  const [play, setPlay] = useState<{ id: number; kind: Pulse }>({ id: 0, kind: 'forward' });
  const f = b.forward;
  const d4 = (v: number) => fx(v, 4, lang);
  const p4 = (v: number) => pfx(v, 4, lang);

  return (
    <>
      <div className="grid grid-2">
        <article className="panel">
          <h3>{pt ? 'Por que trocar o degrau' : 'Why replace the step'}</h3>
          <p>
            {pt
              ? 'Para ajustar a camada oculta, a rede precisa saber como o erro muda quando cada peso muda um pouquinho: a derivada. O degrau é plano em quase todo lugar (derivada zero) e dá um salto em 0. Ele não diz para que lado andar.'
              : 'To adjust the hidden layer, the network needs to know how the error changes when each weight changes a little: the derivative. The step is flat almost everywhere (zero derivative) and jumps at 0. It does not say which way to go.'}
          </p>
          <ul className="arrows">
            <li><b>{pt ? 'Sigmoide' : 'Sigmoid'}</b> {pt ? 'na camada oculta' : 'in the hidden layer'}: <span className="var">σ(z) = 1 / (1 + e^(−z))</span>. {pt ? 'É um degrau suave, vai de 0 a 1, com derivada σ(z)·(1 − σ(z)).' : 'A smooth step from 0 to 1, with derivative σ(z)·(1 − σ(z)).'}</li>
            <li><b>Softmax</b> {pt ? 'na saída' : 'at the output'}: <span className="var">p_k = e^(z_k) / Σ e^(z_j)</span>. {pt ? 'Transforma as três saídas em probabilidades que somam 1.' : 'Turns the three outputs into probabilities that add up to 1.'}</li>
            <li><b>{pt ? 'Entropia cruzada' : 'Cross-entropy'}</b> {pt ? 'como erro' : 'as the error'}: <span className="var">L = −ln p_{pt ? 'certo' : 'correct'}</span>. {pt ? 'Vale 0 quando a rede dá 100% ao tipo certo e cresce rápido quando ela erra com confiança.' : 'It is 0 when the network gives 100% to the right type and grows fast when it is confidently wrong.'}</li>
          </ul>
          <Note title={pt ? 'Entradas divididas por 100' : 'Inputs divided by 100'}>
            {pt
              ? `Velocidade 80 vira ${num(0.8, lang)}. Com valores perto de 1 a sigmoide começa na parte inclinada, onde aprende; com 80 ela já começaria "saturada" em 0 ou 1, com derivada quase zero. É a mesma lição dos pesos que pularam na Aula 1.`
              : 'Speed 80 becomes 0.8. With values near 1 the sigmoid starts on its slope, where it learns; with 80 it would start "saturated" at 0 or 1, with a derivative close to zero. Same lesson as the jumping weights in lesson 1.'}
          </Note>
        </article>
        <article className="panel">
          <h3>{pt ? 'A regra do backpropagation' : 'The backpropagation rule'}</h3>
          <p>
            {pt
              ? 'Depois da propagação para a frente, o erro volta camada por camada. Com softmax e entropia cruzada, as contas ficam curtas:'
              : 'After the forward pass, the error goes back layer by layer. With softmax and cross-entropy, the formulas are short:'}
          </p>
          <div className="formula">{pt
            ? `saída:   δo_k = p_k − d_k
         ∂L/∂v_k,j = δo_k · h_j        ∂L/∂φ_k = δo_k · (−1)
oculta:  δh_j = (Σ_k v_k,j · δo_k) · h_j · (1 − h_j)
         ∂L/∂w_j,i = δh_j · x_i        ∂L/∂θ_j = δh_j · (−1)
ajuste:  peso ← peso − η · ∂L/∂peso`
            : `output:  δo_k = p_k − d_k
         ∂L/∂v_k,j = δo_k · h_j        ∂L/∂φ_k = δo_k · (−1)
hidden:  δh_j = (Σ_k v_k,j · δo_k) · h_j · (1 − h_j)
         ∂L/∂w_j,i = δh_j · x_i        ∂L/∂θ_j = δh_j · (−1)
update:  weight ← weight − η · ∂L/∂weight`}</div>
          <p className="small muted" style={{ margin: 0 }}>
            {pt
              ? 'w e θ são os pesos da camada oculta; v e φ, os da saída. d é a resposta desejada em formato "um ligado": Planta = (0, 0, 1).'
              : 'w and θ are the hidden-layer weights; v and φ, the output ones. d is the desired answer as "one-hot": Grass = (0, 0, 1).'}
          </p>
          <Note title={pt ? 'Parece a regra da Aula 1' : 'It looks like the lesson 1 rule'}>
            {pt
              ? 'Na saída, δo = p − d faz o papel do erro e = d − y (com o sinal trocado, porque aqui subtraímos o gradiente). A novidade é a camada oculta: o erro dela é a soma dos erros da saída, pesada pelas ligações, vezes a derivada da sigmoide. É isso que "propagar o erro para trás" quer dizer.'
              : 'At the output, δo = p − d plays the role of the error e = d − y (with the sign flipped, because here we subtract the gradient). What is new is the hidden layer: its error is the sum of the output errors, weighted by the connections, times the sigmoid derivative. That is what "propagating the error back" means.'}
          </Note>
        </article>
      </div>

      {/* one step by hand */}
      <div className="panel" style={{ marginTop: 20 }}>
        <h3>{pt ? `Um passo à mão: Venusaur com os pesos iniciais (η = ${num(eta, lang)})` : `One step by hand: Venusaur with the initial weights (η = ${num(eta, lang)})`}</h3>
        <div className="player">
          <button className="btn btn--gold btn--sm" onClick={() => setPlay((p) => ({ id: p.id + 1, kind: 'forward' }))}>▶ {pt ? 'Para a frente' : 'Forward'}</button>
          <button className="btn btn--sm" onClick={() => setPlay((p) => ({ id: p.id + 1, kind: 'backward' }))}>◀ {pt ? 'Erro para trás' : 'Error backward'}</button>
          <span className="small muted">{pt ? 'Pesos iniciais sorteados com semente fixa e arredondados a uma casa.' : 'Initial weights drawn with a fixed seed and rounded to one decimal.'}</span>
        </div>
        <div className="grid grid-side" style={{ alignItems: 'start' }}>
          <MlpGraph net={init} x={[venusaur.x1, venusaur.x2]} h={f.h} o={f.p} winner={f.p.indexOf(Math.max(...f.p))} play={play} digits={1} />
          <div className="calc">
            <div className="block on">
              <div className="title">1 · {pt ? 'Para a frente' : 'Forward'} · x = ({num(f.x[0], lang)}, {num(f.x[1], lang)})</div>
              {init.hidden.map((n, j) => (
                <div key={j} className="line">
                  z_h{j + 1} = {num(n.w[0], lang)}·{num(f.x[0], lang)} + {par(n.w[1], lang)}·{num(f.x[1], lang)} + {par(n.theta, lang)}·(−1) = {d4(f.zh[j])} ⟹ h{j + 1} = σ = <span className="res">{d4(f.h[j])}</span>
                </div>
              ))}
              {init.output.map((n, k) => (
                <div key={k} className="line">
                  z_{outNames[k]} = {par(n.w[0], lang)}·{d4(f.h[0])} + {par(n.w[1], lang)}·{d4(f.h[1])} + {par(n.theta, lang)}·(−1) = {d4(f.zo[k])}
                </div>
              ))}
              <div className="line">p = softmax = ({f.p.map(d4).join(' ; ')})</div>
              <div className="line">
                L = −ln p_{T.grass} = −ln {d4(f.p[2])} = <span className="bad">{d4(b.loss)}</span>{' '}
                <span className="muted">({pt ? `a rede acha mais provável ${outNames[f.p.indexOf(Math.max(...f.p))]}` : `the network thinks ${outNames[f.p.indexOf(Math.max(...f.p))]} is most likely`})</span>
              </div>
            </div>
            <div className="block on">
              <div className="title">2 · {pt ? 'Erro na saída' : 'Output error'} · d = (0, 0, 1)</div>
              {b.deltaOut.map((v, k) => (
                <div key={k} className="line">δo_{outNames[k]} = {d4(f.p[k])} − {b.d[k]} = <span className="res">{d4(v)}</span></div>
              ))}
            </div>
            <div className="block on">
              <div className="title">3 · {pt ? 'Erro na camada oculta' : 'Hidden-layer error'}</div>
              {b.deltaHidden.map((v, j) => (
                <div key={j} className="line">
                  δh{j + 1} = ({init.output.map((n, k) => `${par(n.w[j], lang)}·${p4(b.deltaOut[k])}`).join(' + ')}) · {d4(f.h[j])} · (1 − {d4(f.h[j])}) = <span className="res">{d4(v)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="table-wrap" style={{ marginTop: 12 }}>
          <table className="hud">
            <thead><tr><th>{pt ? 'Peso' : 'Weight'}</th><th>{pt ? 'antes' : 'before'}</th><th>∂L/∂{pt ? 'peso' : 'weight'}</th><th>{pt ? 'conta' : 'arithmetic'}</th><th>{pt ? 'depois' : 'after'}</th></tr></thead>
            <tbody>
              {(['hidden', 'output'] as const).flatMap((layer) =>
                init[layer].flatMap((n, j) => [...n.w, n.theta].map((w, i) => {
                  const g = i < n.w.length ? b.grad[layer][j].w[i] : b.grad[layer][j].theta;
                  const a = i < n.w.length ? after[layer][j].w[i] : after[layer][j].theta;
                  return (
                    <tr key={`${layer}${j}${i}`}>
                      <td>{names[layer][j][i]}</td>
                      <td>{num(w, lang)}</td>
                      <td>{d4(g)}</td>
                      <td className="muted">{num(w, lang)} − {num(eta, lang)}·{p4(g)}</td>
                      <td className="gold">{d4(a)}</td>
                    </tr>
                  );
                })))}
            </tbody>
          </table>
        </div>
        <p className="small" style={{ marginTop: 12 }}>
          {pt
            ? `Resultado: o erro do Venusaur caiu de ${d4(b.loss)} para ${d4(lossAfter)}, e a probabilidade de Planta foi de ${fx(100 * f.p[2], 1, lang)}% para ${fx(100 * fAfter.p[2], 1, lang)}%.`
            : `Result: Venusaur's error dropped from ${d4(b.loss)} to ${d4(lossAfter)}, and the Grass probability went from ${fx(100 * f.p[2], 1, lang)}% to ${fx(100 * fAfter.p[2], 1, lang)}%.`}
        </p>
        <Note title={pt ? 'Um Pokémon por vez ou todos juntos?' : 'One Pokémon at a time or all together?'}>
          {pt
            ? 'Um passo só com o Venusaur melhora muito o Venusaur, mas pode piorar os outros. No treino de verdade abaixo, cada época calcula esse gradiente para os nove Pokémon e usa a média (gradiente descendente em lote completo).'
            : 'A step with Venusaur alone helps Venusaur a lot, but may hurt the others. In the real training below, each epoch computes this gradient for all nine Pokémon and uses the average (full-batch gradient descent).'}
        </Note>
      </div>

      <Training model={model} />
    </>
  );
}

function Training({ model }: { model: CachedMlp }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const T = TYPE_NAME[lang];
  const [si, setSi] = useState(0);
  const [id, setId] = useState('venusaur');
  const snap = model.snapshots[si];
  const s = STARTERS.find((q) => q.id === id)!;
  const f = softForward(snap.net, s.x1, s.x2);
  const curve = model.curve.map((c) => ({ epoch: c.epoch, loss: c.loss, acc: 100 * c.accuracy }));
  const last = model.curve[model.curve.length - 1];

  // biggest rise of the loss after a lower value: shows that gradient descent is not monotonic with a large η
  const bump = useMemo(() => {
    let best = { epoch: 0, from: 0, fromEpoch: 0, to: 0, rise: 0 };
    let minSoFar = Infinity;
    let minEpoch = 0;
    for (const c of model.curve) {
      if (c.loss < minSoFar) { minSoFar = c.loss; minEpoch = c.epoch; }
      if (c.loss - minSoFar > best.rise) best = { epoch: c.epoch, from: minSoFar, fromEpoch: minEpoch, to: c.loss, rise: c.loss - minSoFar };
    }
    return best;
  }, [model.curve]);
  const firstFull = model.curve.find((c) => c.accuracy === 1)?.epoch;
  const fin = model.final.hidden;

  return (
    <>
      <div className="panel" style={{ marginTop: 20 }}>
        <h3>{pt ? 'O treino completo' : 'The full training run'}</h3>
        <p className="small" style={{ marginTop: 0 }}>
          {pt
            ? `Mesma rede (2 neurônios ocultos, 3 saídas), mesmos pesos iniciais, η = ${num(model.config.eta, lang)}, ${model.config.epochs} épocas com os nove Pokémon. O treino roda uma vez, fora do site, e o resultado fica em cache, como os modelos da minha dissertação.`
            : `Same network (2 hidden neurons, 3 outputs), same initial weights, η = ${num(model.config.eta, lang)}, ${model.config.epochs} epochs over the nine Pokémon. Training runs once, outside the site, and the result is cached, like the models of my dissertation.`}
        </p>
        <div className="range-row">
          <span className="label">{pt ? 'Época' : 'Epoch'}</span>
          <input type="range" min={0} max={model.snapshots.length - 1} step={1} value={si} onChange={(e) => setSi(Number(e.target.value))}
            aria-label={pt ? 'Época do treino' : 'Training epoch'} />
          <span className="mono gold">{snap.epoch}</span>
          <span className="mono small">L = {fx(snap.loss, 4, lang)} · {pt ? 'acertos' : 'correct'} {Math.round(snap.accuracy * STARTERS.length)}/{STARTERS.length}</span>
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={curve} margin={{ top: 10, right: 20, bottom: 18, left: 4 }}>
            <CartesianGrid stroke="#2a3547" strokeDasharray="3 3" />
            <XAxis dataKey="epoch" type="number" domain={[0, model.config.epochs]} stroke="#a9b6c8" fontSize={11}
              label={{ value: pt ? 'época' : 'epoch', position: 'insideBottom', offset: -8, fill: '#a9b6c8', fontSize: 11 }} />
            <YAxis yAxisId="l" scale="log" domain={['auto', 'auto']} stroke="#ffd25e" fontSize={11} width={52} tickFormatter={(v) => fx(v, 3, lang)} />
            <YAxis yAxisId="a" orientation="right" domain={[0, 100]} stroke="#29d3e8" fontSize={11} unit="%" width={44} />
            <Tooltip contentStyle={{ background: '#0d1b2a', border: '1px solid #29d3e8', fontFamily: 'var(--font-mono)', fontSize: 12 }}
              formatter={(v: number, n: string) => (n === 'acc' ? [`${fx(v, 0, lang)}%`, pt ? 'acertos' : 'accuracy'] : [fx(v, 4, lang), pt ? 'erro L' : 'loss L'])}
              labelFormatter={(e) => `${pt ? 'época' : 'epoch'} ${e}`} />
            <Legend formatter={(v) => (v === 'acc' ? (pt ? 'acertos (%)' : 'accuracy (%)') : (pt ? 'erro médio L (log)' : 'mean loss L (log)'))} />
            <ReferenceLine yAxisId="l" x={snap.epoch} stroke="#f5b52a" strokeDasharray="4 4" />
            <Line yAxisId="l" dataKey="loss" stroke="#ffd25e" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line yAxisId="a" dataKey="acc" stroke="#29d3e8" strokeWidth={1.6} dot={false} isAnimationActive={false} type="stepAfter" />
          </LineChart>
        </ResponsiveContainer>
        <Note title={pt ? 'O erro nem sempre cai' : 'The loss does not always go down'}>
          {pt
            ? `A rede acerta os nove pela primeira vez na época ${firstFull}. Depois o erro sobe de ${fx(bump.from, 4, lang)} (época ${bump.fromEpoch}) para ${fx(bump.to, 4, lang)} (época ${bump.epoch}) antes de voltar a cair. Com η = ${num(model.config.eta, lang)} o passo é grande e às vezes passa do ponto; um η menor seria mais estável, porém mais lento. No fim, o erro médio é ${fx(last.loss, 4, lang)} e a rede acerta os nove.`
            : `The network first gets all nine right at epoch ${firstFull}. Then the loss climbs from ${fx(bump.from, 4, lang)} (epoch ${bump.fromEpoch}) to ${fx(bump.to, 4, lang)} (epoch ${bump.epoch}) before falling again. With η = ${model.config.eta} the step is large and sometimes overshoots; a smaller η would be steadier but slower. At the end, the mean loss is ${fx(last.loss, 4, lang)} and the network gets all nine right.`}
        </Note>
      </div>

      <div className="grid grid-2" style={{ marginTop: 20 }}>
        <article className="panel">
          <h3>{pt ? `Regiões na época ${snap.epoch}` : `Regions at epoch ${snap.epoch}`}</h3>
          <RegionChart x={PLANE} y={PLANE} ticks={PLANE_TICKS} xLabel={pt ? 'x1 · Velocidade' : 'x1 · Speed'} yLabel={pt ? 'x2 · Defesa' : 'x2 · Defense'}
            points={toPoints(lang, (q) => {
              const p = softForward(snap.net, q.x1, q.x2).p;
              return `${T.fire} ${fx(100 * p[0], 0, lang)}% · ${T.water} ${fx(100 * p[1], 0, lang)}% · ${T.grass} ${fx(100 * p[2], 0, lang)}%`;
            })}
            region={(a, c) => predict(snap.net, a, c)} />
          <p className="small muted" style={{ margin: 0 }}>{pt ? 'Cor de fundo = tipo que a rede escolheria em cada ponto.' : 'Background colour = type the network would pick at each point.'}</p>
        </article>
        <article className="panel">
          <h3>{pt ? 'Probabilidades por Pokémon' : 'Probabilities per Pokémon'}</h3>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>Pokémon</th><th className="t-fire">{T.fire}</th><th className="t-water">{T.water}</th><th className="t-grass">{T.grass}</th><th></th></tr></thead>
              <tbody>
                {STARTERS.map((q) => {
                  const p = softForward(snap.net, q.x1, q.x2).p;
                  const ok = predict(snap.net, q.x1, q.x2) === q.type;
                  return (
                    <tr key={q.id} className={q.id === id ? 'current' : ''} onClick={() => setId(q.id)} style={{ cursor: 'pointer' }}>
                      <td className={`t-${q.type}`}>{POKE_NAME[q.id]}</td>
                      {p.map((v, k) => <td key={k}>{fx(100 * v, 1, lang)}%</td>)}
                      <td className={ok ? 'ok' : 'bad'}>{ok ? '✓' : '✗'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </article>
      </div>

      <div className="panel" style={{ marginTop: 20 }}>
        <h3>{pt ? `A rede na época ${snap.epoch}, mostrando ${POKE_NAME[s.id]}` : `The network at epoch ${snap.epoch}, showing ${POKE_NAME[s.id]}`}</h3>
        <PokePicker value={id} onChange={setId} />
        <MlpGraph net={snap.net} x={[s.x1, s.x2]} h={f.h} o={f.p} winner={f.p.indexOf(Math.max(...f.p))} digits={1} />
        <Note title={pt ? 'O que a camada oculta aprendeu' : 'What the hidden layer learned'}>
          {pt
            ? `Ninguém disse à rede para comparar Velocidade com Defesa, mas no fim do treino h1 tem pesos ${fx(fin[0].w[0], 1, lang)} e ${fx(fin[0].w[1], 1, lang)}, e h2 tem ${fx(fin[1].w[0], 1, lang)} e ${fx(fin[1].w[1], 1, lang)}: cada um mede Velocidade − Defesa, com sinais opostos, como os detectores de Fogo e de Água que montei à mão na parte 2. (Entradas divididas por ${SCALE}.)`
            : `Nobody told the network to compare Speed with Defense, but at the end of training h1 has weights ${fx(fin[0].w[0], 1, lang)} and ${fx(fin[0].w[1], 1, lang)}, and h2 has ${fx(fin[1].w[0], 1, lang)} and ${fx(fin[1].w[1], 1, lang)}: each one measures Speed − Defense, with opposite signs, like the Fire and Water detectors I built by hand in part 2. (Inputs divided by ${SCALE}.)`}
        </Note>
      </div>

      <article className="panel panel--gold tip" style={{ marginTop: 20 }}>
        <span className="label">★ {pt ? 'Ponto-chave' : 'Key point'}</span>
        <p style={{ margin: '8px 0 0' }}>
          {pt
            ? 'Um perceptron traça uma reta. Um MLP combina várias: a camada oculta transforma o problema até ele ficar separável por retas, e o backpropagation encontra os pesos levando o erro da saída de volta, camada por camada. Foi assim que o Venusaur deixou de ser Água e passou a ser Planta.'
            : 'A perceptron draws one line. An MLP combines several: the hidden layer reshapes the problem until lines can separate it, and backpropagation finds the weights by carrying the output error back, layer by layer. That is how Venusaur stopped being Water and became Grass.'}
        </p>
      </article>
    </>
  );
}
