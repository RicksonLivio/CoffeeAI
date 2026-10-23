import { useMemo, useState, type ReactNode } from 'react';
import { fx, useLang } from '../../i18n';
import { descend, eig2, linearGrad, linearHessian, linearJ, type DescentPoint } from '../../lib/landscape';
import { countWeights, EXAMPLE_DATA, EXAMPLE_W, linearGradient, numericGradient } from '../../lib/linear';
import Landscape, { type Runner, type SurfaceSpec } from './Landscape';
import Note from './Note';
import Tex from './Tex';

const R = String.raw;

function Card({ kind, label, children }: { kind: 'def' | 'thm' | 'proof' | 'ex' | 'warn' | 'obs'; label: string; children: ReactNode }) {
  return (
    <div className={`nb-card ${kind}`}>
      <span className="nb-kind">{label}</span>
      {children}
    </div>
  );
}

function Page({ id, n, title, children }: { id: string; n: number; title: string; children: ReactNode }) {
  return (
    <section className="section" id={id} aria-labelledby={`${id}-h`}>
      <div className="section-head">
        <h2 id={`${id}-h`}>{n} · {title}</h2>
        <span className="nb-num">PÁG. {String(n).padStart(2, '0')}</span>
      </div>
      <div className="panel nb-page">{children}</div>
    </section>
  );
}

// valley of global minima w1·w2 = 1 inside [−2, 2]²
const hyperbola = (sign: 1 | -1): [number, number][] =>
  Array.from({ length: 61 }, (_, i) => {
    const a = sign * (0.5 + (1.5 * i) / 60);
    return [a, 1 / a] as [number, number];
  });

export default function Lesson3() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const t = (p: string, e: string) => (pt ? p : e);
  const jump = (id: string) => (ev: React.MouseEvent) => { ev.preventDefault(); document.getElementById(id)?.scrollIntoView(); };

  // the worked example, computed by the same code the tests check
  const ex = useMemo(() => descend(0.5, 0.5, 0.1, 8, linearJ, linearGrad), []);
  const g0 = linearGrad(0.5, 0.5);
  const bp = useMemo(() => linearGradient(EXAMPLE_W, EXAMPLE_DATA), []);
  const fd = useMemo(() => numericGradient(EXAMPLE_W, EXAMPLE_DATA), []);
  const diff = Math.max(...bp.flatMap((M, l) => M.flatMap((r, i) => r.map((v, j) => Math.abs(v - fd.grad[l][i][j])))));
  const P = countWeights(EXAMPLE_W);
  const eig0 = eig2(linearHessian(0, 0));
  const eig1 = eig2(linearHessian(1, 1));
  const n = (v: number, d = 4) => fx(v, d, lang);

  const spec: SurfaceSpec = useMemo(() => ({
    f: linearJ,
    grad: linearGrad,
    domain: [-2, 2],
    zMin: 0,
    zMax: 2.5,
    markers: [
      { a: 0, b: 0, label: ['ponto de sela (0, 0)', 'saddle point (0, 0)'] },
      { a: 1, b: 1, label: ['mínimo global', 'global minimum'] },
      { a: -1, b: -1, label: ['mínimo global', 'global minimum'] },
    ],
    axes: ['w₁', 'w₂'],
    curve: [hyperbola(1), hyperbola(-1)],
    verdict: (p: DescentPoint[], ptb: boolean) => {
      const e = p[p.length - 1];
      if (e.loss < 1e-6) return ptb ? `mínimo global: J = 0, w₁·w₂ = ${fx(e.a * e.b, 3, 'pt')}` : `global minimum: J = 0, w₁·w₂ = ${fx(e.a * e.b, 3, 'en')}`;
      if (Math.hypot(e.a, e.b) < 1e-6) return ptb ? 'parada no ponto de sela (∇J = 0)' : 'stopped at the saddle point (∇J = 0)';
      return ptb ? 'ainda descendo' : 'still going down';
    },
  }), []);
  const runners: Runner[] = useMemo(() => [
    { eta: 0.1, color: '#e8312f', label: t('começa em (0, 0)', 'starts at (0, 0)'), start: [0, 0] },
    { eta: 0.1, color: '#7df0ff', label: t('começa em (0,001; 0,001)', 'starts at (0.001, 0.001)'), start: [0.001, 0.001] },
    { eta: 0.1, color: '#b49bff', label: t('começa em (−1,4; 1,6)', 'starts at (−1.4, 1.6)'), start: [-1.4, 1.6] },
    { eta: 0.1, color: '#ffd25e', label: t('seu ponto (clique no mapa)', 'your point (click the map)') },
  ], [lang]); // eslint-disable-line react-hooks/exhaustive-deps
  const [start, setStart] = useState<[number, number]>([-1.5, -0.2]);
  const firstBelow = (p: DescentPoint[]) => p.findIndex((q) => q.loss < 1e-3);
  const escape = useMemo(() => [descend(0.001, 0.001, 0.1, 150, linearJ, linearGrad), descend(0.05, 0.02, 0.1, 150, linearJ, linearGrad)].map(firstBelow), []);

  return (
    <div className="lesson-body">
      <div className="parts">
        <a href="#l3-problema" onClick={jump('l3-problema')}><b>1</b>{t('O problema', 'The problem')}</a>
        <a href="#l3-uma" onClick={jump('l3-uma')}><b>2</b>{t('Uma camada: gradiente', 'One layer: gradient')}</a>
        <a href="#l3-bp" onClick={jump('l3-bp')}><b>3</b>{t('Backpropagation', 'Backpropagation')}</a>
        <a href="#l3-paisagem" onClick={jump('l3-paisagem')}><b>4</b>{t('A paisagem', 'The landscape')}</a>
        <a href="#l3-resumo" onClick={jump('l3-resumo')}><b>5</b>{t('Resumo', 'Summary')}</a>
      </div>

      <p className="lead" style={{ marginTop: 18 }}>
        {t('Como aprendemos os pesos de forma eficiente? No fim, tudo se resume a minimizar uma função de erro. Este caderno faz a conta completa, com derivadas parciais e regra da cadeia, e depois olha para o relevo dessa função: ela é convexa? Tem mínimos ruins? Onde ficam as selas?',
          'How do we learn the weights efficiently? In the end it all comes down to minimizing a loss function. This notebook does the full calculation, with partial derivatives and the chain rule, and then looks at the shape of that function: is it convex? Does it have bad minima? Where are the saddles?')}
      </p>

      {/* ------------------------------------------------------------------ */}
      <Page id="l3-problema" n={1} title={t('O problema: minimizar o erro', 'The problem: minimize the loss')}>
        <Card kind="def" label={t('Definição 1 · rede linear profunda', 'Definition 1 · deep linear network')}>
          <p>{t('Uma rede com L camadas sem função de ativação multiplica a entrada por uma matriz por camada:', 'A network with L layers and no activation function multiplies the input by one matrix per layer:')}</p>
          <Tex block>{R`f(x) = W_L^{\top} \cdots W_2^{\top} W_1^{\top} x, \qquad W_\ell \in \R^{d_{\ell-1} \times d_\ell}, \quad x \in \R^{d_0}`}</Tex>
        </Card>
        <p>{t('Com N exemplos (x_i, y_i), o erro quadrático total é:', 'With N examples (x_i, y_i), the total squared error is:')}</p>
        <Tex block>{R`\gold{J(W_1,\dots,W_L)} = \sum_{i=1}^{N} \bigl\| W_L^{\top} \cdots W_2^{\top} W_1^{\top} x_i - y_i \bigr\|^2 \tag{1}`}</Tex>
        <div className="table-wrap">
          <table className="hud">
            <thead><tr><th>{t('Símbolo', 'Symbol')}</th><th>{t('Significado', 'Meaning')}</th></tr></thead>
            <tbody>
              <tr><td><Tex>{R`N`}</Tex></td><td>{t('número de exemplos (na Aula 2, os 9 Pokémon)', 'number of examples (in lesson 2, the 9 Pokémon)')}</td></tr>
              <tr><td><Tex>{R`x_i,\ y_i`}</Tex></td><td>{t('entrada e resposta certa do exemplo i', 'input and right answer of example i')}</td></tr>
              <tr><td><Tex>{R`d_\ell`}</Tex></td><td>{t('número de neurônios da camada ℓ (d₀ = entradas, d_L = saídas)', 'number of neurons in layer ℓ (d₀ = inputs, d_L = outputs)')}</td></tr>
              <tr><td><Tex>{R`W_\ell`}</Tex></td><td>{t('matriz de pesos da camada ℓ; a coluna j tem os pesos do neurônio j', 'weight matrix of layer ℓ; column j holds the weights of neuron j')}</td></tr>
              <tr><td><Tex>{R`h_\ell`}</Tex></td><td>{t('saída da camada ℓ, com h₀ = x', 'output of layer ℓ, with h₀ = x')}</td></tr>
              <tr><td><Tex>{R`\|v\|^2 = \textstyle\sum_k v_k^2`}</Tex></td><td>{t('quadrado do comprimento do vetor (as barras da fórmula)', 'squared length of the vector (the bars in the formula)')}</td></tr>
            </tbody>
          </table>
        </div>
        <Note title={t('Por que estudar a rede sem ativação?', 'Why study the network without activation?')}>
          {t('O produto das matrizes é uma matriz só (a Aula 2 mostrou que camadas sem ativação "se juntam numa só"), então a rede só representa funções lineares. Mesmo assim, J como função dos fatores W₁, …, W_L não é convexa. É o laboratório mais simples onde a teoria consegue provar como o relevo do erro se comporta.',
            'The product of the matrices is a single matrix (lesson 2 showed that layers without activation "merge into one"), so the network only represents linear functions. Even so, J as a function of the factors W₁, …, W_L is not convex. It is the simplest laboratory where theory can prove how the loss landscape behaves.')}
        </Note>
      </Page>

      {/* ------------------------------------------------------------------ */}
      <Page id="l3-uma" n={2} title={t('Uma camada: gradiente descendente', 'One layer: gradient descent')}>
        <p>{t('Com L = 1 sobra uma matriz W. Chamando o resíduo de r_i = Wᵀx_i − y_i:', 'With L = 1 there is one matrix W. Calling the residual r_i = Wᵀx_i − y_i:')}</p>
        <Tex block>{R`J(W) = \sum_{i=1}^{N} \|r_i\|^2 = \sum_{i=1}^{N} \sum_{k} \Bigl( \sum_{j} W_{jk}\, x_{ij} - y_{ik} \Bigr)^{2}`}</Tex>
        <Card kind="proof" label={t('Derivação, peso a peso', 'Derivation, weight by weight')}>
          <p>{t('Derivamos em relação a um peso W_ab. Só os termos com k = b dependem dele, e dentro da soma em j só o termo j = a:', 'We differentiate with respect to one weight W_ab. Only the terms with k = b depend on it, and inside the sum over j only the term j = a:')}</p>
          <Tex block>{R`\frac{\partial J}{\partial W_{ab}} = \sum_{i=1}^{N} 2\Bigl(\sum_j W_{jb}\,x_{ij} - y_{ib}\Bigr) \cdot \frac{\partial}{\partial W_{ab}}\Bigl(\sum_j W_{jb}\,x_{ij}\Bigr) = \sum_{i=1}^{N} 2\, r_{ib}\, x_{ia}`}</Tex>
          <p>{t('Juntando todos os a e b numa matriz:', 'Putting every a and b in one matrix:')}</p>
          <Tex block>{R`\gold{\nabla_W J = 2 \sum_{i=1}^{N} x_i\, r_i^{\top}} \tag{2}`}</Tex>
          <span className="qed">∎</span>
        </Card>
        <p>{t('O gradiente descendente anda contra esse gradiente, com passo η:', 'Gradient descent moves against that gradient, with step η:')}</p>
        <Tex block>{R`W \leftarrow W - \eta\, \nabla_W J \tag{3}`}</Tex>
        <Card kind="thm" label={t('Fato · uma camada é convexa', 'Fact · one layer is convex')}>
          <p>
            {t('J(W) é uma função quadrática de W, com Hessiana', 'J(W) is a quadratic function of W, with Hessian')}{' '}
            <Tex>{R`H = 2\sum_i x_i x_i^{\top}`}</Tex>{' '}
            {t('(repetida para cada saída), que é semidefinida positiva. Logo J é convexa: todo mínimo local é global, e o gradiente descendente converge para um deles se', '(repeated for each output), which is positive semidefinite. So J is convex: every local minimum is global, and gradient descent converges to one of them if')}{' '}
            <Tex>{R`0 < \eta < 2/\lambda_{\max}(H)`}</Tex>.
          </p>
        </Card>
        <Note title={t('A mesma forma da Aula 1', 'The same shape as lesson 1')}>
          {t('Para um exemplo só, o passo é ΔW = −2η · x · rᵀ: "erro × entrada", como o Δw = η·e·x do perceptron.', 'For a single example, the step is ΔW = −2η · x · rᵀ: "error × input", like the perceptron’s Δw = η·e·x.')}
        </Note>
      </Page>

      {/* ------------------------------------------------------------------ */}
      <Page id="l3-bp" n={3} title={t('Várias camadas: backpropagation', 'Several layers: backpropagation')}>
        <p>
          {t('Com várias camadas o método continua sendo gradiente descendente. O que muda é como calcular ∇J sem se perder: isso é o backpropagation (Rumelhart, Hinton e Williams, 1986), ou seja, ',
            'With several layers the method is still gradient descent. What changes is how to compute ∇J without getting lost: that is backpropagation (Rumelhart, Hinton and Williams, 1986), that is, ')}
          <b className="gold">{t('contabilidade muito cuidadosa + regra da cadeia', 'very careful book-keeping + the chain rule')}</b>.
        </p>
        <Card kind="def" label={t('Definição 2 · o que guardar na ida (a contabilidade)', 'Definition 2 · what to store on the way forward (the book-keeping)')}>
          <Tex block>{R`h_0 = x, \qquad z_\ell = W_\ell^{\top} h_{\ell-1}, \qquad h_\ell = \sigma(z_\ell), \qquad \ell = 1,\dots,L`}</Tex>
          <p className="small">{t('Na rede linear, σ(z) = z e h_ℓ = z_ℓ. Cada h_ℓ e z_ℓ é calculado uma vez, guardado, e reaproveitado na volta.', 'In the linear network, σ(z) = z and h_ℓ = z_ℓ. Each h_ℓ and z_ℓ is computed once, stored, and reused on the way back.')}</p>
        </Card>
        <Card kind="def" label={t('Definição 3 · o erro de cada camada', 'Definition 3 · the error of each layer')}>
          <Tex block>{R`\cy{\delta_\ell} := \frac{\partial J}{\partial z_\ell} \in \R^{d_\ell}`}</Tex>
          <p className="small">{t('É o mesmo δ das Aulas 1 e 2: quanto o erro muda quando a soma de cada neurônio muda um pouco.', 'It is the same δ as in lessons 1 and 2: how much the loss changes when each neuron’s sum changes a little.')}</p>
        </Card>
        <Card kind="proof" label={t('Passo 1 · regra da cadeia para um peso', 'Step 1 · chain rule for one weight')}>
          <p>{t('O peso (W_ℓ)_ab só entra na soma do neurônio b da camada ℓ, (z_ℓ)_b = Σ_a (W_ℓ)_ab (h_ℓ₋₁)_a. Então:', 'The weight (W_ℓ)_ab only enters the sum of neuron b in layer ℓ, (z_ℓ)_b = Σ_a (W_ℓ)_ab (h_ℓ₋₁)_a. So:')}</p>
          <Tex block>{R`\frac{\partial J}{\partial (W_\ell)_{ab}} = \frac{\partial J}{\partial (z_\ell)_b} \cdot \frac{\partial (z_\ell)_b}{\partial (W_\ell)_{ab}} = \cy{(\delta_\ell)_b}\,(h_{\ell-1})_a \tag{4}`}</Tex>
        </Card>
        <Card kind="proof" label={t('Passo 2 · regra da cadeia de uma camada para a anterior', 'Step 2 · chain rule from one layer to the previous one')}>
          <p>{t('A soma (z_ℓ)_j afeta o erro através de todos os neurônios k da camada seguinte:', 'The sum (z_ℓ)_j affects the loss through every neuron k of the next layer:')}</p>
          <Tex block>{R`(\delta_\ell)_j = \sum_{k} \frac{\partial J}{\partial (z_{\ell+1})_k} \cdot \frac{\partial (z_{\ell+1})_k}{\partial (h_\ell)_j} \cdot \frac{\partial (h_\ell)_j}{\partial (z_\ell)_j} = \sigma'\bigl((z_\ell)_j\bigr) \sum_k (W_{\ell+1})_{jk}\, (\delta_{\ell+1})_k`}</Tex>
          <Tex block>{R`\cy{\delta_\ell = \bigl(W_{\ell+1}\,\delta_{\ell+1}\bigr) \odot \sigma'(z_\ell)} \tag{5}`}</Tex>
          <p className="small">{t('⊙ é o produto elemento a elemento. Na rede linear σ′ = 1 e δ_ℓ = W_ℓ₊₁ δ_ℓ₊₁; com sigmoide, σ′ = σ(1 − σ), como na Aula 2.', '⊙ is the element-wise product. In the linear network σ′ = 1 and δ_ℓ = W_ℓ₊₁ δ_ℓ₊₁; with a sigmoid, σ′ = σ(1 − σ), as in lesson 2.')}</p>
        </Card>
        <Card kind="proof" label={t('Passo 3 · o começo e o fim', 'Step 3 · the start and the end')}>
          <p>{t('Na saída, com erro quadrático e saída linear, J = ‖h_L − y‖², então:', 'At the output, with squared error and a linear output, J = ‖h_L − y‖², so:')}</p>
          <Tex block>{R`\delta_L = 2\,(h_L - y) = 2\,r \tag{6}`}</Tex>
          <p>{t('Somando (4) sobre os exemplos, cada matriz de gradiente é um produto externo:', 'Summing (4) over the examples, each gradient matrix is an outer product:')}</p>
          <Tex block>{R`\gold{\nabla_{W_\ell} J = \sum_{i=1}^{N} h_{\ell-1,i}\; \delta_{\ell,i}^{\top}} \tag{7}`}</Tex>
          <p>{t('Abrindo a recursão (5) na rede linear, aparece a fórmula fechada:', 'Unrolling recursion (5) in the linear network gives the closed form:')}</p>
          <Tex block>{R`\nabla_{W_\ell} J = 2 \sum_{i=1}^{N} \underbrace{\bigl(W_{\ell-1}^{\top}\cdots W_1^{\top} x_i\bigr)}_{h_{\ell-1,i}\ \text{(${t('ida', 'forward')})}}\ \underbrace{\bigl(W_{\ell+1}\cdots W_L\, r_i\bigr)^{\top}}_{\delta_{\ell,i}^{\top}/2\ \text{(${t('volta', 'backward')})}} \tag{8}`}</Tex>
          <span className="qed">∎</span>
        </Card>

        <Card kind="ex" label={t('Exemplo · a menor rede profunda, com números', 'Example · the smallest deep network, with numbers')}>
          <p>{t('Dois pesos escalares, f(x) = w₂·w₁·x, um exemplo (x, y) = (1, 1), começando em w₁ = w₂ = 0,5 e η = 0,1:', 'Two scalar weights, f(x) = w₂·w₁·x, one example (x, y) = (1, 1), starting at w₁ = w₂ = 0.5 and η = 0.1:')}</p>
          <Tex block>{R`\begin{aligned}
\text{${t('ida', 'forward')}:}\quad & h_1 = w_1 x = 0{,}5, \qquad \hat y = w_2 h_1 = 0{,}25, \qquad r = \hat y - y = -0{,}75, \qquad J = r^2 = 0{,}5625\\
\text{${t('volta', 'backward')}:}\quad & \cy{\delta_2} = 2r = -1{,}5, \qquad \frac{\partial J}{\partial w_2} = \delta_2\, h_1 = \gold{-0{,}75}, \qquad \cy{\delta_1} = w_2\,\delta_2 = -0{,}75, \qquad \frac{\partial J}{\partial w_1} = \delta_1\, x = \gold{-0{,}75}\\
\text{${t('passo', 'step')}:}\quad & w_1 \leftarrow 0{,}5 - 0{,}1\cdot(-0{,}75) = 0{,}575, \qquad w_2 \leftarrow 0{,}575
\end{aligned}`.replace(/\{,\}/g, pt ? '{,}' : '.')}</Tex>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>{t('passo', 'step')}</th><th>w₁</th><th>w₂</th><th>J</th><th>∂J/∂w₁</th><th>∂J/∂w₂</th></tr></thead>
              <tbody>
                {ex.map((q, k) => {
                  const g = linearGrad(q.a, q.b);
                  return <tr key={k}><td>{k}</td><td>{n(q.a)}</td><td>{n(q.b)}</td><td>{n(q.loss)}</td><td>{n(g[0])}</td><td>{n(g[1])}</td></tr>;
                })}
              </tbody>
            </table>
          </div>
          <p className="small" style={{ margin: 0 }}>{t(`Conferência: ∂J/∂w₁ = ${n(g0[0])} e ∂J/∂w₂ = ${n(g0[1])} no ponto inicial, calculados pelo mesmo código do site.`, `Check: ∂J/∂w₁ = ${n(g0[0])} and ∂J/∂w₂ = ${n(g0[1])} at the start, computed by the same code as the site.`)}</p>
        </Card>

        <Card kind="thm" label={t('Por que é eficiente', 'Why it is efficient')}>
          <p>{t('A alternativa ingênua é a diferença finita: mexer um peso de cada vez e medir o erro.', 'The naive alternative is the finite difference: nudge one weight at a time and measure the loss.')}</p>
          <Tex block>{R`\frac{\partial J}{\partial \theta_p} \approx \frac{J(\theta + \varepsilon e_p) - J(\theta - \varepsilon e_p)}{2\varepsilon} \quad\Rightarrow\quad 2P \text{ ${t('avaliações de J para P pesos', 'evaluations of J for P weights')}}`}</Tex>
          <p>
            {t(`O backpropagation faz uma ida e uma volta, e cada volta custa um múltiplo pequeno e fixo da ida, qualquer que seja P. Na rede linear de exemplo (${P} pesos, 3 camadas, 4 exemplos), a diferença finita precisou de ${fd.evaluations} avaliações de J; o backpropagation, de uma ida e uma volta. Diferença máxima entre os dois gradientes: ${diff.toExponential(1).replace('.', ',')}.`,
              `Backpropagation does one forward and one backward pass, and the backward pass costs a small fixed multiple of the forward one, whatever P is. In the example linear network (${P} weights, 3 layers, 4 examples), finite differences needed ${fd.evaluations} evaluations of J; backpropagation, one forward and one backward pass. Largest difference between the two gradients: ${diff.toExponential(1)}.`)}
          </p>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>{t('Rede', 'Network')}</th><th>{t('pesos P', 'weights P')}</th><th>{t('diferença finita', 'finite differences')}</th><th>backpropagation</th></tr></thead>
              <tbody>
                <tr><td>{t('Aula 2, uma camada oculta', 'Lesson 2, one hidden layer')}</td><td>15</td><td>{t('30 avaliações', '30 evaluations')}</td><td>{t('1 ida + 1 volta', '1 forward + 1 backward')}</td></tr>
                <tr><td>{t('Aula 2, duas camadas ocultas', 'Lesson 2, two hidden layers')}</td><td>21</td><td>{t('42 avaliações', '42 evaluations')}</td><td>{t('1 ida + 1 volta', '1 forward + 1 backward')}</td></tr>
                <tr><td>{t('rede com 10⁸ pesos', 'network with 10⁸ weights')}</td><td>10⁸</td><td>{t('2 × 10⁸ avaliações', '2 × 10⁸ evaluations')}</td><td>{t('1 ida + 1 volta', '1 forward + 1 backward')}</td></tr>
              </tbody>
            </table>
          </div>
          <p className="small" style={{ margin: 0 }}>
            {t('A ideia de calcular derivadas "de trás para a frente" (modo reverso da diferenciação automática) é anterior: Linnainmaa (1970) e Werbos (1974). O artigo de 1986 na Nature mostrou que ela faz redes com camadas ocultas aprenderem representações úteis.',
              'The idea of computing derivatives "from back to front" (reverse-mode automatic differentiation) is older: Linnainmaa (1970) and Werbos (1974). The 1986 Nature paper showed that it makes networks with hidden layers learn useful representations.')}
          </p>
        </Card>
      </Page>

      {/* ------------------------------------------------------------------ */}
      <Page id="l3-paisagem" n={4} title={t('A paisagem de otimização', 'The optimization landscape')}>
        <p>{t('Três perguntas: J é convexa? O gradiente descendente acha o mínimo global? O que são os pontos de sela?', 'Three questions: is J convex? Does gradient descent find the global minimum? What are saddle points?')}</p>
        <Card kind="def" label={t('Definição 4 · pontos críticos', 'Definition 4 · critical points')}>
          <ul className="arrows small" style={{ margin: 0 }}>
            <li><b>{t('Ponto crítico', 'Critical point')}</b>: <Tex>{R`\nabla J(\theta) = 0`}</Tex>.</li>
            <li><b>{t('Mínimo local', 'Local minimum')}</b>: {t('J(θ) ≤ J(θ′) para todo θ′ perto de θ. ', 'J(θ) ≤ J(θ′) for every θ′ near θ. ')}<b>{t('Global', 'Global')}</b>: {t('para todo θ′.', 'for every θ′.')}</li>
            <li><b>{t('Ponto de sela', 'Saddle point')}</b>: {t('crítico, mas com direções em que J sobe e direções em que J desce. Se a Hessiana tem um autovalor negativo, é uma sela "estrita"; se não tem, uma sela "ruim" (degenerada), que as derivadas de segunda ordem não denunciam.', 'critical, but with directions where J goes up and directions where J goes down. If the Hessian has a negative eigenvalue, it is a "strict" saddle; if not, a "bad" (degenerate) saddle, which second-order derivatives do not reveal.')}</li>
          </ul>
        </Card>

        <p>{t('A menor rede profunda do exemplo já mostra tudo:', 'The smallest deep network from the example already shows everything:')}</p>
        <Tex block>{R`J(w_1, w_2) = (w_2\, w_1 - 1)^2, \qquad \nabla J = 2\,(w_1 w_2 - 1)\,\begin{pmatrix} w_2 \\ w_1 \end{pmatrix} \tag{9}`}</Tex>
        <div className="grid grid-2">
          <Card kind="proof" label={t('Não é convexa', 'It is not convex')}>
            <Tex block>{R`\begin{aligned} & J(1,1) = J(-1,-1) = 0 \\ & J\Bigl(\tfrac{(1,1)+(-1,-1)}{2}\Bigr) = J(0,0) = 1 > 0 \end{aligned}`}</Tex>
            <p className="small" style={{ margin: 0 }}>{t('Numa função convexa, o valor no ponto médio nunca passa da média dos valores nas pontas. Aqui passa.', 'In a convex function, the value at the midpoint never exceeds the average of the values at the ends. Here it does.')}</p>
          </Card>
          <Card kind="proof" label={t('Os pontos críticos', 'The critical points')}>
            <p className="small">{t('Por (9), ∇J = 0 exige w₁w₂ = 1 ou w₁ = w₂ = 0. A Hessiana é', 'By (9), ∇J = 0 requires w₁w₂ = 1 or w₁ = w₂ = 0. The Hessian is')}</p>
            <Tex block>{R`\nabla^2 J = 2\begin{pmatrix} w_2^2 & 2w_1w_2 - 1 \\ 2w_1w_2 - 1 & w_1^2 \end{pmatrix}`}</Tex>
            <ul className="arrows small" style={{ margin: 0 }}>
              <li>{t(`Em (0, 0): autovalores ${n(eig0[0], 0)} e ${n(eig0[1], 0)}, sinais opostos: sela estrita.`, `At (0, 0): eigenvalues ${n(eig0[0], 0)} and ${n(eig0[1], 0)}, opposite signs: strict saddle.`)}</li>
              <li>{t(`Em w₁w₂ = 1, por exemplo (1, 1): autovalores ${n(eig1[0], 0)} e ${n(eig1[1], 0)}. J = 0 é o menor valor possível: todo ponto da hipérbole é mínimo global, e o autovalor 0 é a direção ao longo do vale.`, `On w₁w₂ = 1, e.g. (1, 1): eigenvalues ${n(eig1[0], 0)} and ${n(eig1[1], 0)}. J = 0 is the smallest possible value: every point of the hyperbola is a global minimum, and the 0 eigenvalue is the direction along the valley.`)}</li>
            </ul>
          </Card>
        </div>

        <Landscape runners={runners} start={start} onStart={setStart} steps={150} spec={spec} />
        <p className="small" style={{ marginTop: 8 }}>
          {t(`A linha tracejada dourada é o vale de mínimos globais w₁w₂ = 1. Quem começa exatamente em (0, 0) nunca sai: lá ∇J = 0. Começando em (0,001; 0,001), o gradiente é minúsculo e a bola demora ${escape[0]} passos para chegar a J < 0,001; de (0,05; 0,02), ${escape[1]} passos. É assim que selas atrasam o treino sem prendê-lo. Clique em qualquer outro ponto: o resultado é sempre um mínimo global.`,
            `The dashed gold line is the valley of global minima w₁w₂ = 1. Whoever starts exactly at (0, 0) never leaves: there ∇J = 0. Starting at (0.001, 0.001), the gradient is tiny and the ball takes ${escape[0]} steps to reach J < 0.001; from (0.05, 0.02), ${escape[1]} steps. That is how saddles slow training down without trapping it. Click any other point: the result is always a global minimum.`)}
        </p>

        <Card kind="thm" label={t('Teorema (Baldi e Hornik, 1989) · duas camadas lineares', 'Theorem (Baldi and Hornik, 1989) · two linear layers')}>
          <p>{t('Para a rede linear de duas camadas com erro quadrático,', 'For the two-layer linear network with squared error,')} <Tex>{R`J(W_1, W_2) = \sum_i \|W_2^{\top} W_1^{\top} x_i - y_i\|^2`}</Tex>{t(', sob hipóteses técnicas sobre os dados (matrizes de covariância de posto completo e autovalores distintos):', ', under technical assumptions on the data (full-rank covariance matrices and distinct eigenvalues):')}</p>
          <ul className="arrows" style={{ margin: 0 }}>
            <li>{t('todo mínimo local é global;', 'every local minimum is global;')}</li>
            <li>{t('todo ponto crítico é um mínimo global ou um ponto de sela.', 'every critical point is either a global minimum or a saddle point.')}</li>
          </ul>
          <p className="small" style={{ marginTop: 8 }}>{t('Os mínimos globais correspondem à análise de componentes principais (PCA) dos dados.', 'The global minima correspond to principal component analysis (PCA) of the data.')}</p>
        </Card>
        <Card kind="thm" label={t('Teorema (Kawaguchi, 2016) · L camadas lineares', 'Theorem (Kawaguchi, 2016) · L linear layers')}>
          <p>{t('Para a rede linear profunda (1), com qualquer profundidade e quaisquer larguras (sob hipóteses de posto sobre os dados):', 'For the deep linear network (1), with any depth and any widths (under rank assumptions on the data):')}</p>
          <ol className="rules small" style={{ paddingLeft: 20, margin: 0 }}>
            <li>{t('J é não convexa e não côncava;', 'J is non-convex and non-concave;')}</li>
            <li>{t('todo mínimo local é global;', 'every local minimum is global;')}</li>
            <li>{t('todo ponto crítico que não é mínimo global é um ponto de sela;', 'every critical point that is not a global minimum is a saddle point;')}</li>
            <li>{t('com uma camada oculta não existem selas ruins; com mais de uma, existem selas sem autovalor negativo na Hessiana.', 'with one hidden layer there are no bad saddles; with more than one, there are saddles with no negative Hessian eigenvalue.')}</li>
          </ol>
          <p className="small" style={{ marginTop: 8 }}>
            {t('Exemplo do item 4: com três pesos escalares, J(a, b, c) = (abc − 1)². Em (0, 0, 0) o gradiente e a Hessiana são zero, mas J(t, t, t) = (t³ − 1)² fica abaixo de 1 para t > 0 e acima para t < 0: é uma sela que a Hessiana não denuncia.',
              'Example of item 4: with three scalar weights, J(a, b, c) = (abc − 1)². At (0, 0, 0) the gradient and the Hessian are zero, but J(t, t, t) = (t³ − 1)² is below 1 for t > 0 and above for t < 0: a saddle the Hessian does not reveal.')}
          </p>
        </Card>
        <Card kind="warn" label={t('Atenção · e as redes não lineares?', 'Careful · what about nonlinear networks?')}>
          <p>
            {t('É comum ler que o mesmo vale para L camadas com sigmoide entre elas. Não é bem assim. Kawaguchi (2016) estendeu os quatro enunciados a redes não lineares só por redução ao caso linear, sob uma "hipótese de independência" tomada de Choromanska et al. (2015), que o próprio autor reconhece ser irrealista. Não é um teorema geral para sigmoide.',
              'It is common to read that the same holds for L layers with sigmoids between them. Not quite. Kawaguchi (2016) extended the four statements to nonlinear networks only by reduction to the linear case, under an "independence assumption" taken from Choromanska et al. (2015), which the author himself acknowledges is unrealistic. It is not a general theorem for sigmoids.')}
          </p>
          <p>
            {t('Em geral, é falso: Yun, Sra e Jadbabaie (2019) provam que, mesmo com uma única camada oculta e a menor não linearidade, o erro empírico tem mínimos locais espúrios na maioria dos casos; Safran e Shamir (2018) mostram que mínimos espúrios são comuns em redes ReLU de duas camadas. "Sem mínimos locais ruins" é uma propriedade das redes lineares.',
              'In general it is false: Yun, Sra and Jadbabaie (2019) prove that, even with a single hidden layer and the slightest nonlinearity, the empirical risk has spurious local minima in most cases; Safran and Shamir (2018) show that spurious minima are common in two-layer ReLU networks. "No bad local minima" is a property of linear networks.')}
          </p>
          <p className="small" style={{ margin: 0 }}>
            {t('Isso combina com o que vimos na Aula 2: com η = 10 a rede com sigmoide travou num platô, e o relevo de brinquedo da parte 4 tem um mínimo local de verdade.',
              'That matches what we saw in lesson 2: with η = 10 the sigmoid network got stuck on a plateau, and the toy landscape of part 4 has a real local minimum.')}
          </p>
        </Card>
      </Page>

      {/* ------------------------------------------------------------------ */}
      <Page id="l3-resumo" n={5} title={t('Resumo: como aprendemos os pesos', 'Summary: how we learn the weights')}>
        <div className="table-wrap">
          <table className="hud">
            <thead><tr><th>{t('Caso', 'Case')}</th><th>{t('Convexa?', 'Convex?')}</th><th>{t('Mínimos locais ruins?', 'Bad local minima?')}</th><th>{t('Selas', 'Saddles')}</th><th>{t('Fonte', 'Source')}</th></tr></thead>
            <tbody>
              <tr><td>{t('1 camada linear, erro quadrático', '1 linear layer, squared error')}</td><td className="ok">{t('sim', 'yes')}</td><td className="ok">{t('não', 'no')}</td><td>{t('não há', 'none')}</td><td>{t('quadrática', 'quadratic')}</td></tr>
              <tr><td>{t('2 camadas lineares', '2 linear layers')}</td><td className="bad">{t('não', 'no')}</td><td className="ok">{t('não', 'no')}</td><td>{t('só estritas', 'strict only')}</td><td>Baldi & Hornik (1989)</td></tr>
              <tr><td>{t('L camadas lineares', 'L linear layers')}</td><td className="bad">{t('não', 'no')}</td><td className="ok">{t('não', 'no')}</td><td>{t('podem ser ruins (L ≥ 3)', 'can be bad (L ≥ 3)')}</td><td>Kawaguchi (2016)</td></tr>
              <tr><td>{t('não linear (sigmoide, ReLU)', 'nonlinear (sigmoid, ReLU)')}</td><td className="bad">{t('não', 'no')}</td><td className="bad">{t('podem existir', 'they can exist')}</td><td>{t('sim', 'yes')}</td><td>Yun, Sra & Jadbabaie (2019); Safran & Shamir (2018)</td></tr>
            </tbody>
          </table>
        </div>
        <ol className="rules" style={{ paddingLeft: 20, marginTop: 14 }}>
          <li>{t('Escrevemos o erro J dos pesos e queremos o seu mínimo.', 'We write the loss J of the weights and want its minimum.')}</li>
          <li>{t('Andamos contra o gradiente: W ← W − η∇J (equação 3).', 'We move against the gradient: W ← W − η∇J (equation 3).')}</li>
          <li>{t('O gradiente de todas as camadas sai de uma ida (guardando h e z) e uma volta (δ, equações 4 a 7), com custo que não cresce com o número de pesos: isso torna o treino eficiente.', 'The gradient of every layer comes from one forward pass (storing h and z) and one backward pass (δ, equations 4 to 7), at a cost that does not grow with the number of weights: that is what makes training efficient.')}</li>
          <li>{t('Nas redes lineares o relevo é bom: sem mínimos ruins, só selas, que atrasam mas raramente prendem (com pesos iniciais aleatórios, o gradiente descendente quase nunca para numa sela estrita; Lee et al., 2016).', 'In linear networks the landscape is benign: no bad minima, only saddles, which slow down but rarely trap (with random initial weights, gradient descent almost never stops at a strict saddle; Lee et al., 2016).')}</li>
          <li>{t('Nas redes não lineares não há essa garantia; na prática, ajudam a escolha de η, os pesos iniciais, o momento, o Adam e repetir o treino.', 'In nonlinear networks there is no such guarantee; in practice, the choice of η, the initial weights, momentum, Adam and repeated runs help.')}</li>
        </ol>
        <p className="small muted" style={{ margin: 0 }}>
          {t('Referências: Rumelhart, Hinton e Williams, Nature 323 (1986); Baldi e Hornik, Neural Networks 2 (1989); Kawaguchi, NeurIPS (2016); Choromanska et al., AISTATS (2015); Safran e Shamir, ICML (2018); Yun, Sra e Jadbabaie, ICLR (2019); Lee, Simchowitz, Jordan e Recht, COLT (2016); Linnainmaa (1970); Werbos (1974).',
            'References: Rumelhart, Hinton and Williams, Nature 323 (1986); Baldi and Hornik, Neural Networks 2 (1989); Kawaguchi, NeurIPS (2016); Choromanska et al., AISTATS (2015); Safran and Shamir, ICML (2018); Yun, Sra and Jadbabaie, ICLR (2019); Lee, Simchowitz, Jordan and Recht, COLT (2016); Linnainmaa (1970); Werbos (1974).')}
        </p>
      </Page>
    </div>
  );
}
