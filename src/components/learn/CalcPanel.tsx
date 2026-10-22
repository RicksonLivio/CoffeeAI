import { num, par, useLang, type Lang } from '../../i18n';
import type { TrainStep } from '../../lib/perceptron';

const NAMES: Record<string, string> = { charizard: 'Charizard', blastoise: 'Blastoise', venusaur: 'Venusaur' };
export const pokeName = (id: string) => NAMES[id] ?? id;

/** "0 + 0 − 0.5", the way the lesson writes a sum of products. */
export function sumExpr(values: number[], lang: Lang): string {
  return values
    .map((v, i) => (i === 0 ? num(v, lang) : v < 0 ? ` − ${num(-v, lang)}` : ` + ${num(v, lang)}`))
    .join('');
}

interface Props {
  step: TrainStep;
  eta: number;
  phase: number;
}

/** Right-hand column of the lesson: the same step written out as arithmetic. */
export default function CalcPanel({ step, eta, phase }: Props) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const s = step.sample;
  const b = step.before;
  const cls = (p: number) => `block ${phase === p ? 'on' : phase < p ? 'pending' : ''}`;
  const want = s.d === 1 ? (pt ? 'Fogo' : 'Fire') : pt ? 'Água' : 'Water';
  const got = step.y === 1 ? (pt ? 'Fogo' : 'Fire') : pt ? 'Água' : 'Water';

  const rows: [string, number, number, number][] = [
    ['w1', b.w1, step.after.w1, s.x1],
    ['w2', b.w2, step.after.w2, s.x2],
    ['θ', b.theta, step.after.theta, -1],
  ];

  return (
    <div className="calc" aria-live="polite">
      <div className={cls(0)}>
        <div className="title">{pt ? 'Dados' : 'Data'} · {pokeName(s.id)}</div>
        <div className="line">
          x1 = {num(s.x1, lang)} ({pt ? 'Velocidade' : 'Speed'}), x2 = {num(s.x2, lang)} ({pt ? 'Defesa' : 'Defense'}), d = {s.d} ({want})
        </div>
        <div className="line muted">
          {pt ? 'Pesos atuais' : 'Current weights'}: w1 = {num(b.w1, lang)}, w2 = {num(b.w2, lang)}, θ = {num(b.theta, lang)}
        </div>
      </div>

      <div className={cls(1)}>
        <div className="title">1 · {pt ? 'Somatória (net)' : 'Weighted sum (net)'}</div>
        <div className="line">net = (w1 · x1) + (w2 · x2) + (θ · (−1))</div>
        <div className="line">
          net = ({num(b.w1, lang)} · {num(s.x1, lang)}) + ({num(b.w2, lang)} · {num(s.x2, lang)}) + ({num(b.theta, lang)} · −1)
        </div>
        <div className="line">
          net = {sumExpr(step.products, lang)} = <span className="res">{num(step.net, lang)}</span>
        </div>
      </div>

      <div className={cls(2)}>
        <div className="title">2 · {pt ? 'Função degrau' : 'Step function'}</div>
        <div className="line">
          net = {num(step.net, lang)} {step.net >= 0 ? '≥ 0' : '< 0'} ⟹ <span className="res">y = {step.y}</span>{' '}
          ({pt ? `o perceptron achou que era ${got}` : `the perceptron thinks it is ${got}`})
        </div>
      </div>

      <div className={cls(3)}>
        <div className="title">3 · {pt ? 'Erro' : 'Error'}</div>
        <div className="line">
          e = d − y = {s.d} − {step.y} = <span className={step.e === 0 ? 'ok' : 'bad'}>{num(step.e, lang)}</span>
        </div>
        <div className={`line verdict ${step.e === 0 ? 'ok' : 'bad'}`}>
          {step.e === 0
            ? pt ? 'Acertou! Nenhum ajuste necessário.' : 'Correct! No update needed.'
            : pt ? `Errou! O desejado era ${s.d} e deu ${step.y}.` : `Wrong! The target was ${s.d} and the output was ${step.y}.`}
        </div>
      </div>

      <div className={cls(4)}>
        <div className="title">4 · {pt ? 'Atualização dos pesos' : 'Weight update'} (<span style={{ textTransform: 'none' }}>η = {num(eta, lang)}</span>)</div>
        {step.e === 0 ? (
          <div className="line">
            Δw = η · 0 · x = 0 ⟹ {pt ? 'os pesos continuam' : 'weights stay'} w1 = {num(b.w1, lang)}, w2 = {num(b.w2, lang)}, θ = {num(b.theta, lang)}
          </div>
        ) : (
          rows.map(([n, before, after, x]) => (
            <div key={n} className="line">
              {n} = {num(before, lang)} + ({num(eta, lang)} · {par(step.e, lang)} · {par(x, lang)}) ={' '}
              <span className="res">{num(after, lang)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
