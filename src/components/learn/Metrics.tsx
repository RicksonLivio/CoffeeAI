import { fx, pfx, useLang } from '../../i18n';
import { report } from '../../lib/metrics';
import { POKE_NAME, sigmoid, STARTERS, TYPES, type Poke } from '../../lib/mlp';
import Note from './Note';

const NAMES = { pt: ['Fogo', 'Água', 'Planta'], en: ['Fire', 'Water', 'Grass'] };

// --------------------------------------------------------------------------- //
// "Three categories that end up as probabilities: how is the error computed and propagated?"
// --------------------------------------------------------------------------- //
export function ThreeClassError({ p, h, v }: { p: number[]; h: number[]; v: number[][] }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const N = NAMES[lang];
  const d = [0, 0, 1];
  const delta = p.map((q, k) => q - d[k]);
  const dh = h.map((hj, j) => v.reduce((acc, row, k) => acc + row[j] * delta[k], 0) * hj * (1 - hj));
  const d4 = (x: number) => fx(x, 4, lang);
  return (
    <div className="panel panel--gold" style={{ marginTop: 20 }}>
      <h3>{pt ? 'Se temos 3 categorias que no fim são probabilidades, como calculamos o erro e como ele é propagado?' : 'If we have 3 categories that end up as probabilities, how do we compute the error and how is it propagated?'}</h3>
      <div className="grid grid-2">
        <div>
          <p>
            {pt
              ? '1. A resposta certa vira um vetor "um ligado": para o Venusaur, d = (0, 0, 1). A rede devolve p = (p_Fogo, p_Água, p_Planta), que soma 1.'
              : '1. The right answer becomes a "one-hot" vector: for Venusaur, d = (0, 0, 1). The network returns p = (p_Fire, p_Water, p_Grass), which adds up to 1.'}
          </p>
          <p>
            {pt
              ? '2. O erro é a entropia cruzada: L = −Σ d_k · ln p_k. Como só um d_k vale 1, sobra L = −ln p_certo. Se a rede dá 100% ao tipo certo, L = 0; se dá quase 0%, L dispara. É um número só para as três saídas.'
              : '2. The error is the cross-entropy: L = −Σ d_k · ln p_k. Since only one d_k is 1, it becomes L = −ln p_correct. If the network gives the right type 100%, L = 0; if it gives almost 0%, L shoots up. It is one number for the three outputs.'}
          </p>
          <p>
            {pt
              ? '3. Derivando L em relação à soma z_k de cada saída, através do softmax, a conta se simplifica para δ_k = p_k − d_k. As três saídas recebem erro, não só a certa: as erradas recebem δ positivo (devem baixar) e a certa recebe δ negativo (deve subir).'
              : '3. Differentiating L with respect to each output’s sum z_k, through the softmax, the math simplifies to δ_k = p_k − d_k. All three outputs get an error, not only the right one: the wrong ones get a positive δ (must go down) and the right one a negative δ (must go up).'}
          </p>
          <div className="formula">{pt
            ? `softmax liga as três saídas:
  ∂p_i/∂z_k = p_i·(1[i=k] − p_k)
regra da cadeia:
  ∂L/∂z_k = −Σ_i (d_i/p_i)·p_i·(1[i=k] − p_k)
          = p_k·Σ_i d_i − d_k
          = p_k − d_k        (porque Σ d_i = 1)`
            : `softmax couples the three outputs:
  ∂p_i/∂z_k = p_i·(1[i=k] − p_k)
chain rule:
  ∂L/∂z_k = −Σ_i (d_i/p_i)·p_i·(1[i=k] − p_k)
          = p_k·Σ_i d_i − d_k
          = p_k − d_k        (because Σ d_i = 1)`}</div>
          <p className="small" style={{ margin: 0 }}>
            {pt
              ? '4. Propagação: cada neurônio oculto recebe a soma dos três δ, pesada pelas suas ligações com cada saída, vezes a inclinação da sua sigmoide: δh_j = (Σ_k v_k,j · δ_k) · h_j(1 − h_j). Daí para trás é a mesma regra, camada por camada.'
              : '4. Propagation: each hidden neuron gets the sum of the three δ, weighted by its links to each output, times the slope of its sigmoid: δh_j = (Σ_k v_k,j · δ_k) · h_j(1 − h_j). From there back it is the same rule, layer by layer.'}
          </p>
        </div>
        <div>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>{pt ? 'Saída' : 'Output'}</th><th>p</th><th>d</th><th>δ = p − d</th><th>{pt ? 'leitura' : 'meaning'}</th></tr></thead>
              <tbody>
                {N.map((n, k) => (
                  <tr key={n}>
                    <td>{n}</td><td>{d4(p[k])}</td><td>{d[k]}</td><td className={delta[k] < 0 ? 'ok' : 'bad'}>{d4(delta[k])}</td>
                    <td className="muted">{delta[k] < 0 ? (pt ? 'subir' : 'go up') : (pt ? 'baixar' : 'go down')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small">
            {pt
              ? `Venusaur com os pesos iniciais: L = −ln ${d4(p[2])} = ${d4(-Math.log(p[2]))}. A soma dos três δ é ${d4(delta.reduce((a, b) => a + b, 0))}: o que sai de uma probabilidade entra nas outras.`
              : `Venusaur with the initial weights: L = −ln ${d4(p[2])} = ${d4(-Math.log(p[2]))}. The three δ add up to ${d4(delta.reduce((a, b) => a + b, 0))}: what one probability loses, the others gain.`}
          </p>
          <div className="formula">{h.map((_, j) => `δh${j + 1} = (${v.map((row, k) => `${pfx(row[j], 1, lang)}·${pfx(delta[k], 4, lang)}`).join(' + ')})\n      · ${d4(h[j])}·(1 − ${d4(h[j])}) = ${d4(dh[j])}`).join('\n')}</div>
          <Note title={pt ? 'Por que não usar só d − y em cada saída?' : 'Why not just d − y at each output?'}>
            {pt
              ? 'Porque o softmax amarra as saídas: subir p_Planta obriga p_Fogo e p_Água a descer. A entropia cruzada com softmax leva isso em conta e, de quebra, dá um erro de saída com a mesma cara do da Aula 1 (previsto menos desejado).'
              : 'Because the softmax ties the outputs together: raising p_Grass forces p_Fire and p_Water down. Cross-entropy with softmax takes that into account and, as a bonus, gives an output error that looks like the lesson 1 one (predicted minus desired).'}
          </Note>
        </div>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// "How do the activation functions guarantee a classification between 0 and 1?"
// --------------------------------------------------------------------------- //
export function ZeroOneRange() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const ex = [-10, -2, 0, 2, 10];
  return (
    <div className="panel" style={{ marginTop: 20 }}>
      <h3>{pt ? 'Como as funções de ativação garantem valores entre 0 e 1?' : 'How do activation functions guarantee values between 0 and 1?'}</h3>
      <div className="grid grid-2">
        <div>
          <ul className="arrows">
            <li>
              <b>{pt ? 'Degrau' : 'Step'}</b>: {pt ? 'só existem dois resultados, 0 ou 1. A garantia é imediata, mas não há meio-termo.' : 'there are only two results, 0 or 1. The guarantee is immediate, but there is no middle ground.'}
            </li>
            <li>
              <b>{pt ? 'Sigmoide' : 'Sigmoid'}</b>: σ(z) = 1/(1 + e^(−z)). {pt
                ? 'e^(−z) é sempre maior que 0, então o denominador é sempre maior que 1 e σ fica sempre abaixo de 1; e como tudo é positivo, σ fica sempre acima de 0. Ela chega muito perto de 0 ou de 1, mas nunca encosta.'
                : 'e^(−z) is always greater than 0, so the denominator is always greater than 1 and σ stays below 1; and since everything is positive, σ stays above 0. It gets very close to 0 or 1, but never touches them.'}
            </li>
            <li>
              <b>Softmax</b>: p_k = e^(z_k)/Σ e^(z_j). {pt
                ? 'Cada e^z é positivo e a soma de baixo inclui o de cima, então cada p está entre 0 e 1, e os três somam exatamente 1.'
                : 'Each e^z is positive and the sum below includes the one on top, so each p is between 0 and 1, and the three add up to exactly 1.'}
            </li>
          </ul>
          <Note title={pt ? 'A camada de saída no grafo' : 'The output layer in the graph'}>
            {pt
              ? 'A última camada tem 3 neurônios, um por tipo (Fogo, Água, Planta). Cada um calcula a sua soma z_k com os seus pesos; o softmax é aplicado à camada inteira, juntando as três somas, por isso aparece no cabeçalho "SAÍDA · softmax". O número dentro de cada círculo da saída já é a probabilidade p_k, e a classe prevista é a de maior p (círculo com a borda acesa).'
              : 'The last layer has 3 neurons, one per type (Fire, Water, Grass). Each computes its own sum z_k with its weights; the softmax is applied to the whole layer, joining the three sums, which is why the header says "OUTPUT · softmax". The number inside each output circle is already the probability p_k, and the predicted class is the one with the largest p (the circle with the lit border).'}
          </Note>
          <Note title={pt ? 'Garantir o intervalo não é garantir o acerto' : 'Guaranteeing the range is not guaranteeing the right answer'}>
            {pt
              ? 'As ativações garantem que a saída tem forma de probabilidade. Quem faz a probabilidade ficar alta no tipo certo é o treino. A classe escolhida é a de maior p (com três classes, não existe um "limiar de 0,5" fixo).'
              : 'Activations guarantee the output has the shape of a probability. It is training that makes the probability high for the right type. The chosen class is the one with the largest p (with three classes there is no fixed "0.5 threshold").'}
          </Note>
        </div>
        <div>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>z</th>{ex.map((z) => <th key={z}>{z}</th>)}</tr></thead>
              <tbody>
                <tr><td>{pt ? 'degrau' : 'step'}</td>{ex.map((z) => <td key={z}>{z >= 0 ? 1 : 0}</td>)}</tr>
                <tr><td>σ(z)</td>{ex.map((z) => <td key={z}>{fx(sigmoid(z), 5, lang)}</td>)}</tr>
              </tbody>
            </table>
          </div>
          <p className="small">
            {pt
              ? 'Nas camadas ocultas, os valores entre 0 e 1 da sigmoide não são probabilidades de classe: são "quanto o neurônio acendeu". Só a camada de saída, com o softmax, fala em chances de cada tipo.'
              : 'In the hidden layers, the sigmoid’s values between 0 and 1 are not class probabilities: they say "how much the neuron lit up". Only the output layer, with the softmax, talks about the chance of each type.'}
          </p>
          <Note title={pt ? 'E se a tarefa fosse regressão?' : 'What if the task were regression?'}>
            {pt
              ? 'Para prever um número direto (por exemplo, a Velocidade de um Pokémon), a última camada teria um único neurônio com ativação linear (identidade, f(z) = z), que devolve qualquer valor real em vez de probabilidades. O erro passaria a ser, por exemplo, o erro quadrático médio, e o erro da saída seria δ = ŷ − y. As camadas ocultas continuariam com ativações não lineares.'
              : 'To predict a number directly (for example, a Pokémon’s Speed), the last layer would have a single neuron with a linear activation (identity, f(z) = z), returning any real value instead of probabilities. The error would then be, for example, the mean squared error, and the output error would be δ = ŷ − y. The hidden layers would keep non-linear activations.'}
          </Note>
        </div>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Accuracy, precision, recall/sensitivity, specificity, F1, AUC
// --------------------------------------------------------------------------- //
export function MetricsReport({ probs, title }: { probs: (s: Poke) => number[]; title: string }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const N = NAMES[lang];
  const r = report(STARTERS, probs);
  const pc = (x: number) => (Number.isNaN(x) ? '—' : `${fx(100 * x, 1, lang)}%`);
  const wrong = STARTERS.filter((s) => {
    const p = probs(s);
    return TYPES[p.indexOf(Math.max(...p))] !== s.type;
  });
  return (
    <div className="panel" style={{ marginTop: 20 }}>
      <h3>{title}</h3>
      <div className="grid grid-2">
        <div>
          <span className="label">{pt ? 'Matriz de confusão (linhas: tipo real · colunas: previsão)' : 'Confusion matrix (rows: true type · columns: prediction)'}</span>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th></th>{N.map((n) => <th key={n}>{n}</th>)}</tr></thead>
              <tbody>
                {r.confusion.map((row, i) => (
                  <tr key={i}><td className={`t-${TYPES[i]}`}>{N[i]}</td>{row.map((v, j) => <td key={j} className={i === j ? (v ? 'ok' : '') : v ? 'bad' : 'muted'}>{v}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small">
            {pt ? `Acurácia: ${r.confusion.reduce((a, row, i) => a + row[i], 0)}/${r.n} = ` : `Accuracy: ${r.confusion.reduce((a, row, i) => a + row[i], 0)}/${r.n} = `}<b className="gold">{pc(r.accuracy)}</b>
            {wrong.length ? (pt ? ` · erros: ${wrong.map((s) => POKE_NAME[s.id]).join(', ')}` : ` · mistakes: ${wrong.map((s) => POKE_NAME[s.id]).join(', ')}`) : ''}
          </p>
          <Note title={pt ? 'Como ler cada métrica (um tipo contra os outros)' : 'How to read each metric (one type against the others)'}>
            {pt
              ? 'Precisão: dos que a rede chamou de X, quantos eram X. Recall (= sensibilidade): dos que eram X, quantos a rede encontrou. Especificidade: dos que não eram X, quantos a rede não chamou de X. F1: média harmônica de precisão e recall, que só é alta se as duas forem. AUC: chance de um X real receber nota de X maior que um não-X; mede a ordenação das probabilidades, não o acerto final.'
              : 'Precision: of those the network called X, how many were X. Recall (= sensitivity): of those that were X, how many the network found. Specificity: of those that were not X, how many it did not call X. F1: harmonic mean of precision and recall, high only if both are. AUC: chance that a real X gets a higher X score than a non-X; it measures how the probabilities rank, not the final answer.'}
          </Note>
        </div>
        <div>
          <div className="table-wrap">
            <table className="hud">
              <thead><tr><th>{pt ? 'Tipo' : 'Type'}</th><th>{pt ? 'Precisão' : 'Precision'}</th><th>Recall</th><th>{pt ? 'Especif.' : 'Specif.'}</th><th>F1</th><th>AUC</th></tr></thead>
              <tbody>
                {r.perClass.map((c, i) => (
                  <tr key={c.type}><td className={`t-${c.type}`}>{N[i]}</td><td>{pc(c.precision)}</td><td>{pc(c.recall)}</td><td>{pc(c.specificity)}</td><td>{pc(c.f1)}</td><td>{fx(c.auc, 2, lang)}</td></tr>
                ))}
                <tr><td>{pt ? 'média (macro)' : 'macro average'}</td><td>{pc(r.macro.precision)}</td><td>{pc(r.macro.recall)}</td><td></td><td>{pc(r.macro.f1)}</td><td>{fx(r.macro.auc, 2, lang)}</td></tr>
              </tbody>
            </table>
          </div>
          <p className="small muted">
            {pt
              ? 'TP, FP, FN, TN por tipo: ' + r.perClass.map((c, i) => `${N[i]} ${c.tp}/${c.fp}/${c.fn}/${c.tn}`).join(' · ')
              : 'TP, FP, FN, TN per type: ' + r.perClass.map((c, i) => `${N[i]} ${c.tp}/${c.fp}/${c.fn}/${c.tn}`).join(' · ')}
          </p>
          <Note title={pt ? 'Cuidado com estes números' : 'Be careful with these numbers'}>
            {pt
              ? 'Eles são calculados nos mesmos nove Pokémon usados no treino, com três por tipo. Medem o quanto a rede aprendeu estes exemplos, não o quanto acertaria Pokémon novos; para isso seria preciso separar dados de teste. Com só nove exemplos, cada erro muda a acurácia em 11 pontos percentuais.'
              : 'They are computed on the same nine Pokémon used for training, three per type. They measure how well the network learned these examples, not how well it would do on new Pokémon; that would need separate test data. With only nine examples, each mistake moves accuracy by 11 percentage points.'}
          </Note>
        </div>
      </div>
    </div>
  );
}

/** Short note for the training chart: which curve is which, and where AUC fits. */
export function CurveNote() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  return (
    <Note title={pt ? 'Alguma dessas curvas é a AUC?' : 'Is either of these curves the AUC?'}>
      {pt
        ? 'Não. A linha amarela é o erro médio L (entropia cruzada) a cada época, em escala log; a azul é a acurácia, a fração dos nove que a rede acertou. A AUC é outra coisa: é a área sob a curva ROC, que se desenha variando o limiar de decisão para um tipo e marcando a taxa de verdadeiros positivos contra a de falsos positivos. Ela não aparece neste gráfico; o valor da AUC de cada tipo, na época escolhida, está no relatório de métricas logo abaixo.'
        : 'No. The yellow line is the mean loss L (cross-entropy) at each epoch, on a log scale; the blue one is accuracy, the fraction of the nine the network got right. AUC is something else: the area under the ROC curve, drawn by sweeping the decision threshold for one type and plotting the true positive rate against the false positive rate. It is not in this chart; the AUC of each type, at the chosen epoch, is in the metrics report below.'}
    </Note>
  );
}
