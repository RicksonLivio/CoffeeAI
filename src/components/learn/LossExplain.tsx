import { useMemo } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { fx, useLang } from '../../i18n';
import { POKE_NAME, softForward, STARTERS, TYPES, type Net } from '../../lib/mlp';
import Note from './Note';

const COLOR = { fire: '#ff7a2f', water: '#2f7bff', grass: '#4cc46a' } as const;

/** What the loss is, what cross-entropy is, why the logarithm, and the calculation with the real numbers. */
export default function LossExplain({ init, final, eta }: { init: Net; final: Net; eta: number }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const t = (a: string, b: string) => (pt ? a : b);
  const T = pt ? ['Fogo', 'Água', 'Planta'] : ['Fire', 'Water', 'Grass'];
  const d4 = (v: number) => fx(v, 4, lang);

  const rows = useMemo(() => STARTERS.map((s) => {
    const p = softForward(init, s.x1, s.x2).p;
    const pc = p[TYPES.indexOf(s.type)];
    return { s, pc, L: -Math.log(pc), sq: (1 - pc) ** 2 };
  }), [init]);
  const sumL = rows.reduce((a, r) => a + r.L, 0);
  const meanL = sumL / rows.length;
  const prodP = rows.reduce((a, r) => a * r.pc, 1);
  const finalMean = STARTERS.reduce((a, s) => a - Math.log(softForward(final, s.x1, s.x2).p[TYPES.indexOf(s.type)]), 0) / STARTERS.length;
  const uniform = -Math.log(1 / 3);
  const curve = useMemo(() => Array.from({ length: 100 }, (_, i) => {
    const p = 0.01 + (0.99 * i) / 99;
    return { p, ce: -Math.log(p), sq: (1 - p) ** 2 };
  }), []);
  const examples = [0.99, 0.9, 0.5, 0.1, 0.01];

  return (
    <div className="panel panel--gold" style={{ marginTop: 20 }}>
      <h3>{t('O que é a loss e o que é a entropia cruzada', 'What the loss is and what cross-entropy is')}</h3>
      <div className="grid grid-2">
        <div>
          <p>
            <b className="gold">{t('Loss (função de erro).', 'Loss (error function).')}</b>{' '}
            {t('É um único número que mede o quanto a rede está errando nos dados. Vale 0 quando ela acerta tudo com total certeza e cresce quanto pior ela vai. Treinar é procurar os pesos que deixam esse número o menor possível. Na Aula 1 bastava contar erros (e = d − y); aqui a saída é uma probabilidade, então a loss precisa levar em conta o quanto a rede tem certeza.',
              'It is a single number that measures how wrong the network is on the data. It is 0 when it gets everything right with full certainty and grows the worse it does. Training means looking for the weights that make this number as small as possible. In lesson 1 counting mistakes was enough (e = d − y); here the output is a probability, so the loss has to take into account how sure the network is.')}
          </p>
          <p>
            <b className="gold">{t('Entropia cruzada.', 'Cross-entropy.')}</b>{' '}
            {t('Compara a resposta certa d (em formato "um ligado") com as probabilidades p da rede:', 'Compares the right answer d (in "one-hot" form) with the network’s probabilities p:')}
          </p>
          <div className="formula">{t(
            `L = −Σ_k d_k · ln p_k = −ln p_certo        (um Pokémon)
L̄ = (1/N) · Σ_i −ln p_certo,i                (média nos N = 9)`,
            `L = −Σ_k d_k · ln p_k = −ln p_correct      (one Pokémon)
L̄ = (1/N) · Σ_i −ln p_correct,i              (mean over N = 9)`)}</div>
          <p className="small">
            {t('Como só a posição certa de d vale 1, sobra o logaritmo da probabilidade que a rede deu ao tipo certo, com sinal trocado. As probabilidades dadas aos tipos errados não aparecem na conta, mas entram indiretamente: pelo softmax, para p_certo subir, as outras precisam descer.',
              'Since only the right position of d is 1, what remains is the logarithm of the probability the network gave the right type, with the sign flipped. The probabilities given to wrong types do not appear in the sum, but they come in indirectly: through the softmax, for p_correct to go up, the others must go down.')}
          </p>

          <h3>{t('Por que o logaritmo?', 'Why the logarithm?')}</h3>
          <ol className="rules small" style={{ paddingLeft: 20 }}>
            <li>
              <b>{t('Transforma produto em soma.', 'It turns a product into a sum.')}</b>{' '}
              {t(`A chance de a rede dar a resposta certa para os nove ao mesmo tempo é o produto das nove probabilidades. Com os pesos iniciais, esse produto é ${prodP.toExponential(2).replace('.', pt ? ',' : '.')}: um número minúsculo, difícil de comparar e de derivar. O logaritmo troca o produto por uma soma, ln(Π p) = Σ ln p, e −Σ ln p = ${d4(sumL)} = 9 × ${d4(meanL)}. Minimizar a entropia cruzada é o mesmo que maximizar essa chance (máxima verossimilhança).`,
                `The chance that the network gives the right answer for all nine at once is the product of the nine probabilities. With the initial weights that product is ${prodP.toExponential(2)}: a tiny number, hard to compare and to differentiate. The logarithm swaps the product for a sum, ln(Π p) = Σ ln p, and −Σ ln p = ${d4(sumL)} = 9 × ${d4(meanL)}. Minimizing cross-entropy is the same as maximizing that chance (maximum likelihood).`)}
            </li>
            <li>
              <b>{t('Pune o erro confiante.', 'It punishes confident mistakes.')}</b>{' '}
              {t('−ln p vale 0 quando p = 1 e cresce sem limite quando p vai para 0. Dar 1% ao tipo certo custa muito mais do que dar 50%. O erro quadrático (1 − p)² nunca passa de 1 e trata quase igual um erro leve e um erro grosseiro.',
                '−ln p is 0 when p = 1 and grows without limit as p goes to 0. Giving the right type 1% costs far more than giving it 50%. The squared error (1 − p)² never goes above 1 and treats a mild mistake and a gross one almost the same.')}
            </li>
            <li>
              <b>{t('Deixa o gradiente simples.', 'It keeps the gradient simple.')}</b>{' '}
              {t('Junto com o softmax, a derivada da entropia cruzada em relação a cada soma de saída é só δ = p − d. O sinal de correção fica grande quando a rede erra feio e não some por causa do softmax, o que ajuda o treino a sair do lugar.',
                'Together with the softmax, the derivative of cross-entropy with respect to each output sum is just δ = p − d. The correction signal is large when the network is badly wrong and is not killed by the softmax, which helps training get moving.')}
            </li>
          </ol>
        </div>

        <div>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={curve} margin={{ top: 10, right: 16, bottom: 22, left: 4 }}>
              <CartesianGrid stroke="#2a3547" strokeDasharray="3 3" />
              <XAxis dataKey="p" type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} stroke="#a9b6c8" fontSize={11} tickFormatter={(v) => fx(v, 2, lang)}
                label={{ value: t('p_certo: probabilidade dada ao tipo certo', 'p_correct: probability given to the right type'), position: 'insideBottom', offset: -10, fill: '#a9b6c8', fontSize: 11 }} />
              <YAxis domain={[0, 5]} stroke="#a9b6c8" fontSize={11} width={36} />
              <Tooltip contentStyle={{ background: '#0d1b2a', border: '1px solid #29d3e8', fontFamily: 'var(--font-mono)', fontSize: 12 }}
                formatter={(v: number) => fx(v, 3, lang)} labelFormatter={(p) => `p = ${fx(Number(p), 2, lang)}`} />
              <Legend verticalAlign="top" height={30} formatter={(k) => (k === 'ce' ? t('entropia cruzada −ln p', 'cross-entropy −ln p') : t('erro quadrático (1 − p)²', 'squared error (1 − p)²'))} />
              <Line dataKey="ce" stroke="#ffd25e" strokeWidth={3} dot={false} isAnimationActive={false} />
              <Line dataKey="sq" stroke="#7b4dff" strokeDasharray="5 4" dot={false} isAnimationActive={false} />
              {rows.map((r) => <ReferenceDot key={r.s.id} x={r.pc} y={r.L} r={5} fill={COLOR[r.s.type]} stroke="#e6ecf4" />)}
            </LineChart>
          </ResponsiveContainer>
          <p className="small muted" style={{ marginTop: 0 }}>{t('Pontos: os nove Pokémon com os pesos iniciais, cada um na altura da sua loss.', 'Dots: the nine Pokémon with the initial weights, each at the height of its loss.')}</p>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>p_{t('certo', 'correct')}</th>{examples.map((p) => <th key={p}>{fx(p, 2, lang)}</th>)}</tr></thead>
              <tbody>
                <tr><td>−ln p</td>{examples.map((p) => <td key={p}>{fx(-Math.log(p), 3, lang)}</td>)}</tr>
                <tr><td>(1 − p)²</td>{examples.map((p) => <td key={p}>{fx((1 - p) ** 2, 3, lang)}</td>)}</tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <h3 style={{ marginTop: 18 }}>{t('A conta completa, com os pesos iniciais', 'The full calculation, with the initial weights')}</h3>
      <div className="table-wrap">
        <table className="hud">
          <thead><tr><th>Pokémon</th><th>{t('tipo certo', 'right type')}</th><th>p_{t('certo', 'correct')}</th><th>ln p_{t('certo', 'correct')}</th><th>L = −ln p_{t('certo', 'correct')}</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.s.id}>
                <td className={`t-${r.s.type}`}>{POKE_NAME[r.s.id]}</td><td>{T[TYPES.indexOf(r.s.type)]}</td>
                <td>{d4(r.pc)}</td><td>{d4(Math.log(r.pc))}</td><td className="gold">{d4(r.L)}</td>
              </tr>
            ))}
            <tr><td colSpan={4}>{t('soma', 'sum')}</td><td>{d4(sumL)}</td></tr>
            <tr><td colSpan={4}><b>{t('média L̄ = soma / 9', 'mean L̄ = sum / 9')}</b></td><td className="gold"><b>{d4(meanL)}</b></td></tr>
          </tbody>
        </table>
      </div>
      <div className="grid grid-2" style={{ marginTop: 10 }}>
        <Note title={t('Como a loss interfere no treino', 'How the loss drives training')}>
          {t(`A loss é a função que o treino tenta diminuir: todo ajuste de peso vem da derivada dela. Cada Pokémon empurra os pesos com força proporcional ao seu erro δ = p − d. O Venusaur, com p_certo = ${d4(rows[8].pc)}, tem L = ${d4(rows[8].L)} e puxa muito; um Pokémon que a rede já acerta com 99% quase não puxa. Trocar a loss muda o que a rede aprende a priorizar.`,
            `The loss is the function training tries to reduce: every weight update comes from its derivative. Each Pokémon pushes the weights with a force proportional to its error δ = p − d. Venusaur, with p_correct = ${d4(rows[8].pc)}, has L = ${d4(rows[8].L)} and pulls hard; a Pokémon the network already gets right with 99% barely pulls. Changing the loss changes what the network learns to prioritize.`)}
        </Note>
        <Note title={t('Como queremos minimizar a loss', 'How we want to minimize the loss')}>
          {t(`Não existe fórmula pronta para os melhores pesos de uma rede com sigmoide, então descemos a loss aos poucos: calculamos o gradiente ∇L (o lado em que ela sobe) e damos um passo no sentido contrário, peso ← peso − η·∂L/∂peso, com η = ${fx(eta, 1, lang)}. Um palpite ao acaso, 1/3 para cada tipo, daria L = −ln(1/3) = ${d4(uniform)}; os pesos iniciais dão ${d4(meanL)}, um pouco pior. Depois do treino, a média cai para ${d4(finalMean)}.`,
            `There is no ready formula for the best weights of a sigmoid network, so we walk the loss down little by little: we compute the gradient ∇L (the direction where it rises) and take a step the other way, weight ← weight − η·∂L/∂weight, with η = ${fx(eta, 1, lang)}. A random guess, 1/3 per type, would give L = −ln(1/3) = ${d4(uniform)}; the initial weights give ${d4(meanL)}, slightly worse. After training, the mean drops to ${d4(finalMean)}.`)}
        </Note>
      </div>
    </div>
  );
}
