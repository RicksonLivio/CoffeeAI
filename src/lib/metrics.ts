/**
 * Classification metrics for the three types, one-vs-rest for each class.
 *   TP: was the class and the network said it      FP: was not, the network said it
 *   FN: was the class, the network said another    TN: was not, the network did not say it
 * precision = TP/(TP+FP), recall = sensitivity = TP/(TP+FN), specificity = TN/(TN+FP),
 * F1 = 2·precision·recall/(precision+recall), accuracy = correct/total.
 * AUC (one-vs-rest): probability that a random Pokémon of the class gets a higher score
 * for that class than a random Pokémon of another class (ties count 1/2).
 */
import { TYPES, type Poke, type PokeType } from './mlp.ts';

export interface ClassMetrics {
  type: PokeType;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  support: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  auc: number;
}

export interface Report {
  /** confusion[i][j] = number of Pokémon of true class i predicted as class j */
  confusion: number[][];
  perClass: ClassMetrics[];
  accuracy: number;
  macro: { precision: number; recall: number; f1: number; auc: number };
  n: number;
}

const safe = (a: number, b: number) => (b === 0 ? 0 : a / b);

/** Mann–Whitney form of the ROC AUC. */
export function auc(scores: number[], positive: boolean[]): number {
  const pos = scores.filter((_, i) => positive[i]);
  const neg = scores.filter((_, i) => !positive[i]);
  if (!pos.length || !neg.length) return NaN;
  let s = 0;
  for (const p of pos) for (const q of neg) s += p > q ? 1 : p === q ? 0.5 : 0;
  return s / (pos.length * neg.length);
}

/** `probs(s)` returns the three class probabilities for a Pokémon. */
export function report(data: Poke[], probs: (s: Poke) => number[]): Report {
  const P = data.map(probs);
  const pred = P.map((p) => p.indexOf(Math.max(...p)));
  const truth = data.map((s) => TYPES.indexOf(s.type));
  const confusion = TYPES.map((_, i) => TYPES.map((__, j) => data.filter((_s, k) => truth[k] === i && pred[k] === j).length));
  const perClass: ClassMetrics[] = TYPES.map((type, c) => {
    const tp = confusion[c][c];
    const fp = TYPES.reduce((acc, _, i) => acc + (i !== c ? confusion[i][c] : 0), 0);
    const fn = TYPES.reduce((acc, _, j) => acc + (j !== c ? confusion[c][j] : 0), 0);
    const tn = data.length - tp - fp - fn;
    const precision = safe(tp, tp + fp);
    const recall = safe(tp, tp + fn);
    return {
      type, tp, fp, fn, tn, support: tp + fn, precision, recall,
      specificity: safe(tn, tn + fp),
      f1: safe(2 * precision * recall, precision + recall),
      auc: auc(P.map((p) => p[c]), truth.map((t) => t === c)),
    };
  });
  const mean = (f: (m: ClassMetrics) => number) => perClass.reduce((a, m) => a + f(m), 0) / perClass.length;
  return {
    confusion,
    perClass,
    accuracy: safe(pred.filter((p, k) => p === truth[k]).length, data.length),
    macro: { precision: mean((m) => m.precision), recall: mean((m) => m.recall), f1: mean((m) => m.f1), auc: mean((m) => m.auc) },
    n: data.length,
  };
}
