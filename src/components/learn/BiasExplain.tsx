import { useMemo } from 'react';
import { fx, num, useLang } from '../../i18n';
import { deepCurve, deepForward, initDeep } from '../../lib/deep';
import { HAND_NET, STARTERS, stepForward } from '../../lib/mlp';
import { buildCachedMlp } from '../../lib/model';
import { LESSON_INIT, trainStep, CHARIZARD } from '../../lib/perceptron';
import Note from './Note';

/**
 * Why θ = 11 is positive, what x0 = −1 is, why biases come out negative,
 * and whether we really need h1 and h2. Every number is computed from the library.
 */
export default function BiasExplain() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const t = (a: string, b: string) => (pt ? a : b);
  const v = STARTERS.find((s) => s.id === 'venusaur')!;
  const c = STARTERS.find((s) => s.id === 'charizard')!;
  const fV = stepForward(HAND_NET, v.x1, v.x2);
  const fC = stepForward(HAND_NET, c.x1, c.x2);
  const step1 = trainStep(LESSON_INIT, CHARIZARD, 1, 1, 0);
  const trainedTheta = buildCachedMlp().final.hidden.map((n) => n.theta);
  // a single hidden neuron (2 → 1 → 3), same data, η = 2
  const one = useMemo(() => {
    const r = deepCurve(initDeep(2, [1, 3]), STARTERS, 2, 600, 1);
    const hOf = (type: string) => {
      const hs = STARTERS.filter((s) => s.type === type).map((s) => deepForward(r.final, s.x1, s.x2).a[1][0]);
      return hs.reduce((a, b) => a + b, 0) / hs.length;
    };
    return { first: r.curve.find((q) => q.accuracy === 1)?.epoch, acc: r.curve[r.curve.length - 1].accuracy, h: { fire: hOf('fire'), water: hOf('water'), grass: hOf('grass') } };
  }, []);
  const h1n = HAND_NET.hidden[0];

  return (
    <div className="panel panel--gold" style={{ marginTop: 20 }}>
      <h3>{t('O bias θ, a entrada x₀ = −1 e o sinal: por que θ = 11 é positivo?', 'The bias θ, the input x₀ = −1 and the sign: why is θ = 11 positive?')}</h3>
      <div className="grid grid-2">
        <div>
          <p>
            <b>{t('x₀ não é peso: é uma entrada.', 'x₀ is not a weight: it is an input.')}</b>{' '}
            {t('É uma entrada fixa, sempre igual a −1, que todo neurônio recebe além das entradas de verdade. Ela não é aprendida e não muda. O peso é θ, que fica na ligação dessa entrada até o neurônio, e esse sim é aprendido como os outros.',
              'It is a fixed input, always −1, that every neuron receives besides the real inputs. It is not learned and never changes. The weight is θ, sitting on the link from that input to the neuron, and that one is learned like the others.')}
          </p>
          <div className="formula">{t(
            `net = w1·x1 + w2·x2 + θ·x0      com x0 = −1 (entrada fixa)
    = w1·x1 + w2·x2 − θ

h1: w = (1, −1), θ = 11
net_h1 = x1 − x2 + 11·(−1) = x1 − x2 − 11
h1 = 1  ⟺  x1 − x2 − 11 ≥ 0  ⟺  x1 − x2 ≥ 11`,
            `net = w1·x1 + w2·x2 + θ·x0      with x0 = −1 (fixed input)
    = w1·x1 + w2·x2 − θ

h1: w = (1, −1), θ = 11
net_h1 = x1 − x2 + 11·(−1) = x1 − x2 − 11
h1 = 1  ⟺  x1 − x2 − 11 ≥ 0  ⟺  x1 − x2 ≥ 11`)}</div>
          <p>
            {t('Por isso θ = 11 é positivo: com a entrada −1, θ funciona como um limiar, a "altura da barra" que a soma w·x precisa passar para o neurônio acender. Queremos que h1 acenda só quando a Velocidade passa a Defesa por pelo menos 11, então o limiar é +11. O sinal de menos vem da entrada x₀ = −1, não do peso.',
              'That is why θ = 11 is positive: with the −1 input, θ works as a threshold, the "height of the bar" the sum w·x must clear for the neuron to light up. We want h1 to light up only when Speed beats Defense by at least 11, so the threshold is +11. The minus sign comes from the input x₀ = −1, not from the weight.')}
          </p>
          <ul className="arrows small">
            <li>{t(`Charizard: net_h1 = ${c.x1} − ${c.x2} − 11 = ${num(fC.hiddenNet[0], lang)} ≥ 0 ⟹ h1 = ${fC.h[0]}.`, `Charizard: net_h1 = ${c.x1} − ${c.x2} − 11 = ${num(fC.hiddenNet[0], lang)} ≥ 0 ⟹ h1 = ${fC.h[0]}.`)}</li>
            <li>{t(`Venusaur: net_h1 = ${v.x1} − ${v.x2} − 11 = ${num(fV.hiddenNet[0], lang)} < 0 ⟹ h1 = ${fV.h[0]}.`, `Venusaur: net_h1 = ${v.x1} − ${v.x2} − 11 = ${num(fV.hiddenNet[0], lang)} < 0 ⟹ h1 = ${fV.h[0]}.`)}</li>
          </ul>
          <Note title={t('A mesma coisa em outra notação: o bias b', 'The same thing in another notation: the bias b')}>
            {t(`Muitos livros e o PyTorch usam uma entrada fixa +1 com um peso b: net = w·x + b. As duas formas são a mesma conta com b = −θ. Para h1, b = −${num(h1n.theta, lang)}: o bias "aparece negativo" porque é o limiar com o sinal trocado. Nos códigos em PyTorch desta aula, por isso, fazemos bias = −θ.`,
              `Many books and PyTorch use a fixed +1 input with a weight b: net = w·x + b. Both are the same calculation with b = −θ. For h1, b = −${num(h1n.theta, lang)}: the bias "shows up negative" because it is the threshold with the sign flipped. That is why the PyTorch code in this lesson sets bias = −θ.`)}
          </Note>
        </div>
        <div>
          <h3 style={{ marginTop: 0 }}>{t('Por que o bias fica negativo?', 'Why does the bias become negative?')}</h3>
          <ul className="arrows small">
            <li>
              <b>{t('O termo que entra na soma é −θ.', 'The term that enters the sum is −θ.')}</b>{' '}
              {t('Com θ positivo, a contribuição θ·x₀ = θ·(−1) é negativa: ela puxa a soma para baixo, e o neurônio só acende se as outras entradas compensarem. Com θ = 11, a contribuição é −11.',
                'With a positive θ, the contribution θ·x₀ = θ·(−1) is negative: it pulls the sum down, and the neuron only lights up if the other inputs make up for it. With θ = 11, the contribution is −11.')}
            </li>
            <li>
              <b>{t('Um θ negativo é um limiar abaixo de zero.', 'A negative θ is a threshold below zero.')}</b>{' '}
              {t(`Aí θ·(−1) soma algo positivo, e o neurônio fica "ligado por padrão". É o caso da saída Planta da rede à mão: θ = ${num(HAND_NET.output[2].theta, lang)}, net = 0,5 − h1 − h2. Sem nenhum detector aceso (Venusaur: h = (0, 0)), net = ${num(fV.outputNet[2], lang)} ≥ 0 e Planta acende.`,
                `Then θ·(−1) adds something positive, and the neuron is "on by default". That is the Grass output of the hand-made network: θ = ${num(HAND_NET.output[2].theta, lang)}, net = 0.5 − h1 − h2. With no detector lit (Venusaur: h = (0, 0)), net = ${num(fV.outputNet[2], lang)} ≥ 0 and Grass lights up.`)}
            </li>
            <li>
              <b>{t('No treino, θ desce quando o neurônio devia ter acendido.', 'During training, θ goes down when the neuron should have fired.')}</b>{' '}
              {t(`Pela regra da Aula 1, Δθ = η·e·x₀ = −η·e. No Charizard, o perceptron disse 0 e o certo era 1 (e = 1): Δθ = ${num(step1.delta.theta, lang)}, e θ foi de ${num(step1.before.theta, lang)} para ${num(step1.after.theta, lang)}. Baixar o limiar facilita acender da próxima vez; se ele cai abaixo de 0, fica negativo.`,
                `By the lesson 1 rule, Δθ = η·e·x₀ = −η·e. For Charizard, the perceptron said 0 and the answer was 1 (e = 1): Δθ = ${num(step1.delta.theta, lang)}, and θ went from ${num(step1.before.theta, lang)} to ${num(step1.after.theta, lang)}. Lowering the threshold makes it easier to fire next time; if it drops below 0, it becomes negative.`)}
            </li>
            <li>
              <b>{t('Na rede treinada, o sinal é o que o treino precisar.', 'In the trained network, the sign is whatever training needs.')}</b>{' '}
              {t(`Os neurônios ocultos da parte 3 aprenderam θ = (${trainedTheta.map((x) => fx(x, 2, lang)).join(' ; ')}). Esse sinal não tem significado sozinho: o treino ajusta θ junto com os pesos w, e os dois juntos decidem onde fica a rampa da sigmoide.`,
                `The hidden neurons of part 3 learned θ = (${trainedTheta.map((x) => fx(x, 2, lang)).join(' ; ')}). That sign means nothing on its own: training adjusts θ together with the weights w, and both together decide where the sigmoid ramp sits.`)}
            </li>
          </ul>

          <h3>{t('Precisamos mesmo de h1 e h2?', 'Do we really need h1 and h2?')}</h3>
          <ul className="arrows small">
            <li>
              <b>{t('Com degrau, sim: dois é o mínimo.', 'With a step, yes: two is the minimum.')}</b>{' '}
              {t('Um neurônio com degrau só responde 0 ou 1, ou seja, só separa dois grupos, e o tipo Planta fica numa faixa entre duas retas. Cada reta é um neurônio: precisamos de duas para fechar a faixa.',
                'A step neuron only answers 0 or 1, so it only separates two groups, and Grass sits in a band between two lines. Each line is a neuron: we need two to close the band.')}
            </li>
            <li>
              <b>{t('Com sigmoide, para estes nove, até um bastaria.', 'With a sigmoid, for these nine, even one would do.')}</b>{' '}
              {one.acc === 1
                ? t(`Testei uma rede 2 → 1 → 3 com os mesmos dados e η = 2: ela acerta os nove a partir da época ${one.first}. Depois de 600 épocas, o único neurônio oculto vale em média ${fx(one.h.water, 2, lang)} para Água, ${fx(one.h.grass, 2, lang)} para Planta e ${fx(one.h.fire, 2, lang)} para Fogo: a sigmoide dá a Planta um valor no meio, e a saída separa esses três níveis.`,
                  `I tested a 2 → 1 → 3 network with the same data and η = 2: it gets all nine right from epoch ${one.first}. After 600 epochs, the single hidden neuron averages ${fx(one.h.water, 2, lang)} for Water, ${fx(one.h.grass, 2, lang)} for Grass and ${fx(one.h.fire, 2, lang)} for Fire: the sigmoid gives Grass a middle value, and the output separates those three levels.`)
                : t('Numa rede 2 → 1 → 3, o treino não chegou a acertar os nove neste teste.', 'In a 2 → 1 → 3 network, training did not reach all nine in this test.')}
            </li>
            <li>
              {t('Então h1 e h2 são a escolha didática: cada um vira uma pergunta clara ("é Fogo?", "é Água?"), dá para fazer a conta à mão e é a menor rede que funciona também com degrau. Em problemas reais, o número de neurônios ocultos é escolhido testando; mais neurônios dão mais formas possíveis, e mais chance de decorar os dados.',
                'So h1 and h2 are the teaching choice: each becomes a clear question ("is it Fire?", "is it Water?"), the math can be done by hand, and it is the smallest network that also works with a step. In real problems, the number of hidden neurons is chosen by testing; more neurons allow more shapes, and more risk of memorizing the data.')}
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
