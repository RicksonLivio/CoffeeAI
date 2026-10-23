import { useMemo, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, XAxis, YAxis } from 'recharts';
import { fx, num, pfx, useLang } from '../../i18n';
import { toDeep, type DeepNet } from '../../lib/deep';
import { backprop, HAND_NET, sgdStep, sigmoid, STARTERS, type Net } from '../../lib/mlp';
import NetGraph from './NetGraph';
import Note from './Note';

// --------------------------------------------------------------------------- //
// What a "smooth step" is
// --------------------------------------------------------------------------- //
export function SmoothSteps() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const data = useMemo(() => Array.from({ length: 161 }, (_, i) => {
    const z = -8 + i * 0.1;
    const s = sigmoid(z);
    return { z: Number(z.toFixed(1)), step: z >= 0 ? 1 : 0, k05: sigmoid(0.5 * z), k1: s, k3: sigmoid(3 * z), d: s * (1 - s) };
  }), []);
  const v = (z: number) => fx(sigmoid(z), 3, lang);
  const label: Record<string, string> = {
    step: pt ? 'degrau (Aula 1)' : 'step (lesson 1)',
    k05: pt ? 'σ(0,5·z)' : 'σ(0.5·z)',
    k1: 'σ(z)',
    k3: 'σ(3·z)',
    d: pt ? "derivada σ'(z) = σ(1 − σ)" : "derivative σ'(z) = σ(1 − σ)",
  };
  return (
    <div className="grid grid-side">
      <div>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data} margin={{ top: 10, right: 16, bottom: 22, left: 4 }}>
            <CartesianGrid stroke="#2a3547" strokeDasharray="3 3" />
            <XAxis dataKey="z" type="number" domain={[-8, 8]} ticks={[-8, -4, -2, 0, 2, 4, 8]} stroke="#a9b6c8" fontSize={11}
              label={{ value: pt ? 'z = soma ponderada do neurônio (net)' : 'z = the neuron’s weighted sum (net)', position: 'insideBottom', offset: -10, fill: '#a9b6c8', fontSize: 11 }} />
            <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} stroke="#a9b6c8" fontSize={11} width={40} tickFormatter={(t) => fx(t, 2, lang)} />
            <Legend verticalAlign="top" height={36} formatter={(k) => label[k] ?? k} />
            <Line dataKey="step" stroke="#a9b6c8" strokeWidth={2} dot={false} isAnimationActive={false} type="stepAfter" />
            <Line dataKey="k05" stroke="#7b4dff" dot={false} isAnimationActive={false} />
            <Line dataKey="k1" stroke="#ffd25e" strokeWidth={3} dot={false} isAnimationActive={false} />
            <Line dataKey="k3" stroke="#29d3e8" dot={false} isAnimationActive={false} />
            <Line dataKey="d" stroke="#e8312f" strokeDasharray="5 4" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div>
        <p>
          {pt
            ? 'O degrau da Aula 1 responde só 0 ou 1 e muda de uma vez em z = 0. A sigmoide faz a mesma pergunta ("a soma passou do limiar?"), mas responde com um número entre 0 e 1 que sobe aos poucos. Por isso é chamada de degrau suave.'
            : 'The lesson 1 step only answers 0 or 1 and switches at once at z = 0. The sigmoid asks the same question ("did the sum pass the threshold?"), but answers with a number between 0 and 1 that rises gradually. That is why it is called a smooth step.'}
        </p>
        <div className="formula">{`σ(z) = 1 / (1 + e^(−z))
σ(−4) = ${v(-4)}   σ(−2) = ${v(-2)}   σ(0) = ${v(0)}
σ(2)  = ${v(2)}   σ(4)  = ${v(4)}`}</div>
        <ul className="arrows small">
          <li>{pt ? 'Longe de 0 ela parece o degrau: perto de 0 ou de 1.' : 'Far from 0 it looks like the step: close to 0 or 1.'}</li>
          <li>{pt ? 'Perto de 0 ela é inclinada, e a inclinação (linha vermelha) chega a 0,25 em z = 0. O degrau tem inclinação 0 em todo lugar.' : 'Near 0 it slopes, and the slope (red line) reaches 0.25 at z = 0. The step has slope 0 everywhere.'}</li>
          <li>{pt ? 'Multiplicar z por k muda a rampa: σ(3·z) é quase um degrau, σ(0,5·z) é bem suave. Na rede, esse k vem do tamanho dos pesos.' : 'Multiplying z by k changes the ramp: σ(3·z) is almost a step, σ(0.5·z) is very smooth. In the network, that k comes from the size of the weights.'}</li>
        </ul>
        <Note title={pt ? 'Por que isso importa' : 'Why it matters'}>
          {pt
            ? 'A inclinação diz quanto a saída do neurônio muda quando a soma muda um pouco. O backpropagation precisa exatamente disso para saber quanto corrigir cada peso. Com o degrau, a resposta seria sempre "nada".'
            : 'The slope says how much the neuron’s output changes when the sum changes a little. Backpropagation needs exactly that to know how much to correct each weight. With the step, the answer would always be "nothing".'}
        </Note>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Every hidden node is a perceptron; where the activation goes and why it matters
// --------------------------------------------------------------------------- //
/** Grass output of the hand-made net when the activations are removed (identity): constant for every Pokémon. */
export function linearGrass(x1: number, x2: number): number {
  const h = HAND_NET.hidden.map((n) => n.w[0] * x1 + n.w[1] * x2 - n.theta);
  const g = HAND_NET.output[2];
  return g.w[0] * h[0] + g.w[1] * h[1] - g.theta;
}

export function PerceptronNodes() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const constant = linearGrass(80, 83);
  const rows = pt
    ? [
      ['Degrau', 'Aula 1 e parte 2', '1 se net ≥ 0; senão 0', 'resposta sim/não, fácil de fazer à mão; sem derivada útil'],
      ['Sigmoide', 'camadas ocultas (partes 3 e 5)', '1 / (1 + e^(−net))', 'degrau suave entre 0 e 1, com derivada: deixa o backpropagation funcionar'],
      ['Softmax', 'camada de saída (partes 3 e 5)', 'e^(z_k) / Σ e^(z_j)', 'transforma as três saídas em probabilidades que somam 1'],
      ['ReLU', 'redes modernas (não usada aqui)', 'max(0, net)', 'inclinação 1 quando ativa: o erro enfraquece menos em redes profundas'],
    ]
    : [
      ['Step', 'lesson 1 and part 2', '1 if net ≥ 0; else 0', 'yes/no answer, easy by hand; no useful derivative'],
      ['Sigmoid', 'hidden layers (parts 3 and 5)', '1 / (1 + e^(−net))', 'smooth step between 0 and 1, with a derivative: makes backpropagation work'],
      ['Softmax', 'output layer (parts 3 and 5)', 'e^(z_k) / Σ e^(z_j)', 'turns the three outputs into probabilities that add up to 1'],
      ['ReLU', 'modern networks (not used here)', 'max(0, net)', 'slope 1 when active: the error fades less in deep networks'],
    ];
  return (
    <div className="panel" style={{ marginTop: 20 }}>
      <h3>{pt ? 'Cada neurônio oculto é um perceptron' : 'Every hidden neuron is a perceptron'}</h3>
      <div className="grid grid-2">
        <div>
          <p>
            {pt
              ? 'Cada círculo da camada oculta (e da saída) é o perceptron da Aula 1: faz a soma ponderada das entradas com o bias e passa o resultado pela função de ativação. O h1 da parte 2, com w = (1, −1) e θ = 11, é exatamente um perceptron que pergunta "Velocidade − Defesa ≥ 11?".'
              : 'Each circle of the hidden layer (and of the output) is the lesson 1 perceptron: it takes the weighted sum of its inputs with the bias and passes the result through the activation function. The h1 of part 2, with w = (1, −1) and θ = 11, is exactly a perceptron asking "Speed − Defense ≥ 11?".'}
          </p>
          <div className="formula">{pt
            ? `1. soma:     net = w1·x1 + w2·x2 + θ·(−1)
              (soma ponderada + bias)
2. ativação: a = f(net)   ← aqui entra a função
3. saída:    a vai para a próxima camada`
            : `1. sum:        net = w1·x1 + w2·x2 + θ·(−1)
                (weighted sum + bias)
2. activation: a = f(net)   ← the function goes here
3. output:     a goes to the next layer`}</div>
          <p className="small">
            {pt
              ? 'O MLP é uma equipe de perceptrons: os da primeira camada leem os dados, os seguintes leem as respostas dos anteriores. A ativação fica dentro de cada neurônio, logo depois da soma.'
              : 'The MLP is a team of perceptrons: those in the first layer read the data, the next ones read the answers of the previous ones. The activation sits inside each neuron, right after the sum.'}
          </p>
        </div>
        <div>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>{pt ? 'Ativação' : 'Activation'}</th><th>{pt ? 'onde' : 'where'}</th><th>{pt ? 'fórmula' : 'formula'}</th><th>{pt ? 'para quê' : 'what for'}</th></tr></thead>
              <tbody>{rows.map((r) => <tr key={r[0]}>{r.map((c, i) => <td key={i} className={i === 0 ? 'gold' : ''}>{c}</td>)}</tr>)}</tbody>
            </table>
          </div>
          <Note title={pt ? 'Sem ativação, as camadas se juntam numa só' : 'Without activation, the layers merge into one'}>
            {pt
              ? `Tire o degrau da rede da parte 2 (use f(net) = net). A saída Planta vira 0,5 − (x1 − x2 − 11) − (x2 − x1 − 11) = ${num(constant, lang)} para qualquer Pokémon: as duas retas se cancelam e a faixa some. Em geral, empilhar camadas sem ativação dá outra conta linear, igual a uma camada só. É a ativação, por não ser uma reta, que deixa a rede dobrar o espaço e desenhar curvas.`
              : `Remove the step from the part 2 network (use f(net) = net). The Grass output becomes 0.5 − (x1 − x2 − 11) − (x2 − x1 − 11) = ${num(constant, lang)} for every Pokémon: the two lines cancel out and the band disappears. In general, stacking layers without activation gives another linear formula, the same as a single layer. It is the activation, because it is not a straight line, that lets the network bend the space and draw curves.`}
          </Note>
        </div>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Softmax: how the three outputs become numbers that add up to 1
// --------------------------------------------------------------------------- //
export function SoftmaxExplain({ z, who, context }: { z: number[]; who: string; context: string }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const names = pt ? ['Fogo', 'Água', 'Planta'] : ['Fire', 'Water', 'Grass'];
  const e = z.map(Math.exp);
  const S = e.reduce((a, b) => a + b, 0);
  const p = e.map((v) => v / S);
  const win = p.indexOf(Math.max(...p));
  const big = Math.max(...z);
  return (
    <div className="panel" style={{ marginTop: 20 }}>
      <h3>{pt ? `Da saída ternária às probabilidades: softmax (${context})` : `From a three-way output to probabilities: softmax (${context})`}</h3>
      <div className="grid grid-2">
        <div>
          <p>
            {pt
              ? 'Na parte 2 cada saída era um degrau: 0 ou 1. O Venusaur saía (0, 0, 1), uma resposta ternária no formato "um ligado". Esse formato não diz o quanto a rede tem certeza, e nada impede duas saídas ligadas ao mesmo tempo.'
              : 'In part 2 each output was a step: 0 or 1. Venusaur came out as (0, 0, 1), a three-way answer in "one-hot" form. That form does not say how sure the network is, and nothing stops two outputs from being on at once.'}
          </p>
          <p>
            {pt ? 'O softmax resolve as duas coisas em quatro passos:' : 'Softmax fixes both in four steps:'}
          </p>
          <ol className="rules small" style={{ paddingLeft: 20 }}>
            <li>{pt ? 'pega as somas z de cada saída (podem ser negativas);' : 'takes each output’s sum z (it can be negative);'}</li>
            <li>{pt ? 'calcula e^z, que é sempre positivo e cresce rápido;' : 'computes e^z, which is always positive and grows fast;'}</li>
            <li>{pt ? 'soma os três: S = e^z_Fogo + e^z_Água + e^z_Planta;' : 'adds the three: S = e^z_Fire + e^z_Water + e^z_Grass;'}</li>
            <li>{pt ? 'divide cada um por S: p_k = e^z_k / S.' : 'divides each one by S: p_k = e^z_k / S.'}</li>
          </ol>
          <p className="small" style={{ margin: 0 }}>
            {pt
              ? 'Como cada p é uma fatia de S, as três somam S/S = 1. A saída deixa de ser um "liga/desliga" e vira uma divisão de 100% entre os tipos, que competem entre si: se um sobe, os outros descem.'
              : 'Since each p is a slice of S, the three add up to S/S = 1. The output stops being an on/off and becomes a split of 100% among the types, which compete: if one goes up, the others go down.'}
          </p>
        </div>
        <div>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>{pt ? 'Saída' : 'Output'}</th><th>z</th><th>e^z</th><th>p = e^z / S</th></tr></thead>
              <tbody>
                {names.map((n, k) => (
                  <tr key={n} className={k === win ? 'current' : ''}>
                    <td>{n}</td><td>{fx(z[k], 4, lang)}</td><td>{fx(e[k], 4, lang)}</td><td className={k === win ? 'gold' : ''}>{fx(p[k], 4, lang)} ({fx(100 * p[k], 1, lang)}%)</td>
                  </tr>
                ))}
                <tr><td>{pt ? 'soma' : 'sum'}</td><td></td><td>S = {fx(S, 4, lang)}</td><td className="ok">{fx(p.reduce((a, b) => a + b, 0), 4, lang)}</td></tr>
              </tbody>
            </table>
          </div>
          <p className="small">
            {pt
              ? `${who}, rede treinada: a maior probabilidade é ${names[win]} (${fx(100 * p[win], 1, lang)}%), e essa é a resposta da rede.`
              : `${who}, trained network: the largest probability is ${names[win]} (${fx(100 * p[win], 1, lang)}%), and that is the network’s answer.`}
          </p>
          <Note title={pt ? 'Detalhe de implementação' : 'Implementation detail'}>
            {pt
              ? `Subtrair o mesmo número de todos os z não muda p, porque ele aparece em cima e embaixo da fração. O código subtrai o maior z (${fx(big, 4, lang)}) antes do e^z para não estourar o número em redes grandes.`
              : `Subtracting the same number from every z does not change p, because it shows up in the top and bottom of the fraction. The code subtracts the largest z (${fx(big, 4, lang)}) before e^z so the number does not overflow in large networks.`}
          </Note>
        </div>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Following one error correction from the output to the first weight
// --------------------------------------------------------------------------- //
export function ErrorTrace({ init, eta }: { init: Net; eta: number }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [step, setStep] = useState(0);
  const v = STARTERS.find((s) => s.id === 'venusaur')!;
  const b = useMemo(() => backprop(init, v), [init, v]);
  const after = useMemo(() => sgdStep(init, b.grad, eta), [init, b, eta]);
  const f = b.forward;
  const d4 = (x: number) => fx(x, 4, lang);
  const p4 = (x: number) => pfx(x, 4, lang);
  const names = pt ? ['Fogo', 'Água', 'Planta'] : ['Fire', 'Water', 'Grass'];
  const gV = b.grad.output[2].w[0];
  const gW = b.grad.hidden[0].w[0];
  const vOld = init.output[2].w[0];
  const wOld = init.hidden[0].w[0];

  const steps = [
    {
      t: pt ? 'O erro aparece na saída' : 'The error shows up at the output',
      text: pt
        ? `A rede deu ${fx(100 * f.p[2], 1, lang)}% para Planta, e o certo era 100%. Erro da saída Planta: δ_Planta = p − d = ${d4(f.p[2])} − 1 = ${d4(b.deltaOut[2])}. Negativo: a saída Planta precisa subir.`
        : `The network gave Grass ${fx(100 * f.p[2], 1, lang)}%, and the right answer was 100%. Error of the Grass output: δ_Grass = p − d = ${d4(f.p[2])} − 1 = ${d4(b.deltaOut[2])}. Negative: the Grass output must go up.`,
      edges: [], nodes: ['1:2'], deltas: [null, b.deltaOut] as (number[] | null)[], net: init,
    },
    {
      t: pt ? 'Culpa do peso que chega na saída' : 'Blame on the weight that reaches the output',
      text: pt
        ? `O peso v3,1 liga h1 à saída Planta. Sua parte da culpa é δ_Planta × o que ele carregou: ∂L/∂v3,1 = ${p4(b.deltaOut[2])} · h1 = ${p4(b.deltaOut[2])} · ${d4(f.h[0])} = ${d4(gV)}. O valor fica guardado; v3,1 ainda vale ${num(vOld, lang)}.`
        : `Weight v3,1 links h1 to the Grass output. Its share of the blame is δ_Grass × what it carried: ∂L/∂v3,1 = ${p4(b.deltaOut[2])} · h1 = ${p4(b.deltaOut[2])} · ${d4(f.h[0])} = ${d4(gV)}. The value is stored; v3,1 is still ${num(vOld, lang)}.`,
      edges: ['1:2:0'], nodes: ['1:2', '0:0'], deltas: [null, b.deltaOut] as (number[] | null)[], net: init,
    },
    {
      t: pt ? 'O erro atravessa para a camada oculta' : 'The error crosses into the hidden layer',
      text: pt
        ? `h1 manda sinal para as três saídas, então recebe o erro das três, cada um pesado pela ligação (com os pesos antigos): δh1 = (${init.output.map((n, k) => `${p4(n.w[0])}·${p4(b.deltaOut[k])}`).join(' + ')}) · h1(1 − h1) = ${d4(b.deltaHidden[0])}.`
        : `h1 sends signal to the three outputs, so it gets the error of all three, each weighted by the connection (old weights): δh1 = (${init.output.map((n, k) => `${p4(n.w[0])}·${p4(b.deltaOut[k])}`).join(' + ')}) · h1(1 − h1) = ${d4(b.deltaHidden[0])}.`,
      edges: ['1:0:0', '1:1:0', '1:2:0'], nodes: ['0:0', '1:0', '1:1', '1:2'], deltas: [b.deltaHidden, b.deltaOut] as (number[] | null)[], net: init,
    },
    {
      t: pt ? 'Culpa do primeiro peso' : 'Blame on the first weight',
      text: pt
        ? `O peso w1,1 liga a Velocidade a h1: ∂L/∂w1,1 = δh1 · x1 = ${p4(b.deltaHidden[0])} · ${fx(f.x[0], 2, lang)} = ${d4(gW)}. Também fica guardado. O mesmo acontece, ao mesmo tempo, com os outros 13 pesos.`
        : `Weight w1,1 links Speed to h1: ∂L/∂w1,1 = δh1 · x1 = ${p4(b.deltaHidden[0])} · ${fx(f.x[0], 2, lang)} = ${d4(gW)}. It is stored too. The same happens, at the same time, to the other 13 weights.`,
      edges: ['0:0:0'], nodes: ['-1:0', '0:0'], deltas: [b.deltaHidden, b.deltaOut] as (number[] | null)[], net: init,
    },
    {
      t: pt ? 'Só agora todos os pesos mudam' : 'Only now do all weights change',
      text: pt
        ? `Com todas as culpas calculadas, cada peso anda contra o seu gradiente (η = ${num(eta, lang)}): v3,1 = ${num(vOld, lang)} − ${num(eta, lang)}·${p4(gV)} = ${d4(after.output[2].w[0])} e w1,1 = ${num(wOld, lang)} − ${num(eta, lang)}·${p4(gW)} = ${d4(after.hidden[0].w[0])}. A correção que começou na saída Planta chegou ao primeiro peso.`
        : `With all the blame computed, every weight moves against its gradient (η = ${num(eta, lang)}): v3,1 = ${num(vOld, lang)} − ${num(eta, lang)}·${p4(gV)} = ${d4(after.output[2].w[0])} and w1,1 = ${num(wOld, lang)} − ${num(eta, lang)}·${p4(gW)} = ${d4(after.hidden[0].w[0])}. The correction that started at the Grass output reached the first weight.`,
      edges: ['1:2:0', '0:0:0'], nodes: [], deltas: undefined, net: after,
    },
  ];
  const s = steps[step];
  const deep: DeepNet = toDeep(s.net);
  const fShown = step === 4 ? { h: f.h, p: f.p } : f;

  return (
    <div className="panel" style={{ marginTop: 20 }}>
      <h3>{pt ? 'Exemplo: a correção de um erro, da saída até o primeiro peso' : 'Example: one error correction, from the output to the first weight'}</h3>
      <div className="steps">
        {steps.map((q, i) => (
          <button key={i} className={`step-pill ${i === step ? 'active' : i < step ? 'done' : ''}`} onClick={() => setStep(i)}>{i + 1}. {q.t}</button>
        ))}
      </div>
      <div className="grid grid-side" style={{ alignItems: 'start' }}>
        <NetGraph net={deep} x={[v.x1, v.x2]} acts={[fShown.h, fShown.p]} winner={null} digits={step === 4 ? 2 : 1}
          highlight={{ edges: s.edges, nodes: s.nodes }} deltas={s.deltas} changed={step === 4} />
        <div className="calc">
          <div className="block on">
            <div className="title">{step + 1} · {s.t}</div>
            <div className="line">{s.text}</div>
          </div>
          <div className="player">
            <button className="btn btn--ghost btn--sm" onClick={() => setStep((x) => Math.max(0, x - 1))} disabled={step === 0}>◀ {pt ? 'Voltar' : 'Back'}</button>
            <button className="btn btn--gold btn--sm" onClick={() => setStep((x) => Math.min(steps.length - 1, x + 1))} disabled={step === steps.length - 1}>{pt ? 'Próximo' : 'Next'} ▶</button>
          </div>
          <p className="small muted" style={{ margin: 0 }}>
            {pt ? `Venusaur, pesos iniciais, saídas: ${names.join(', ')}. Em vermelho, o caminho do erro.` : `Venusaur, initial weights, outputs: ${names.join(', ')}. In red, the path of the error.`}
          </p>
        </div>
      </div>
    </div>
  );
}
