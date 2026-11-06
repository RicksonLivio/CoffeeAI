/**
 * Pokédex mini-game, phase 1: the trainer captures Pokémon by running the perceptron of lesson 1.
 *
 * The encounters are the lesson's training run, in order (η = 1, initial weights (0, 0, 0.5),
 * Charizard then Blastoise, two epochs), plus a bonus Venusaur with the final weights.
 * The Pokédex loads the Ball the perceptron predicts (y = 1 → Fire Ball, y = 0 → Water Ball);
 * the capture works only when that prediction matches the real type (e = d − y = 0).
 * Wrong options are the usual mistakes, so each one teaches something.
 */
import {
  LESSON_ETA, LESSON_INIT, LESSON_SAMPLES, X0, clean, classify, train, type Label, type Weights,
} from './perceptron.ts';

export type PokeId = 'charizard' | 'blastoise' | 'venusaur';
export type PokeType = 'fire' | 'water' | 'grass';
export type Ball = 'fire' | 'water';

/** Venusaur is not in the training data (base Speed 80, Defense 83), same as "Try it yourself". */
export const VENUSAUR = { id: 'venusaur' as const, x1: 80, x2: 83 };

export interface Encounter {
  /** 1-based position in the game */
  n: number;
  /** training epoch (1 or 2); null for the bonus encounter */
  epoch: number | null;
  pokemon: PokeId;
  type: PokeType;
  x1: number;
  x2: number;
  before: Weights;
  /** w1·x1, w2·x2, θ·x0 */
  products: [number, number, number];
  net: number;
  y: Label;
  ball: Ball;
  caught: boolean;
  /** desired output; null for Venusaur, which has no 0/1 label */
  d: Label | null;
  /** e = d − y; null for Venusaur */
  e: number | null;
  after: Weights;
  /** the correct net plus two common mistakes, in a fixed shuffled order */
  netOptions: number[];
  /** only when e ≠ 0: the correct update plus two common mistakes */
  updateOptions: Weights[];
}

export const ERROR_OPTIONS = [-1, 0, 1];

const typeOf = (d: Label): PokeType => (d === 1 ? 'fire' : 'water');
const ballOf = (y: Label): Ball => (y === 1 ? 'fire' : 'water');

/** Moves the first k items to the end, so the right answer is not always in the same place. */
function rotate<T>(items: T[], k: number): T[] {
  const s = ((k % items.length) + items.length) % items.length;
  return [...items.slice(s), ...items.slice(0, s)];
}

function uniqueBy<T>(items: T[], key: (t: T) => string): T[] {
  const seen = new Set<string>();
  return items.filter((t) => {
    const k = key(t);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/** Correct net, then "added θ instead of θ·(−1)" and "forgot the bias". */
export function netChoices(w: Weights, x1: number, x2: number, n: number): number[] {
  const a = clean(w.w1 * x1 + w.w2 * x2);
  const right = clean(a + w.theta * X0);
  const extra = [clean(a + w.theta), a, clean(right + 1), clean(right - 1)];
  return rotate(uniqueBy([right, ...extra], String).slice(0, 3), n);
}

const wKey = (w: Weights) => `${w.w1}|${w.w2}|${w.theta}`;

/** Correct w + η·e·x, then "subtracted the update" and "forgot that x0 = −1". */
export function updateChoices(w: Weights, x1: number, x2: number, eta: number, e: number, n: number): Weights[] {
  const d = { w1: eta * e * x1, w2: eta * e * x2, theta: eta * e * X0 };
  const right = { w1: clean(w.w1 + d.w1), w2: clean(w.w2 + d.w2), theta: clean(w.theta + d.theta) };
  const minus = { w1: clean(w.w1 - d.w1), w2: clean(w.w2 - d.w2), theta: clean(w.theta - d.theta) };
  const noX0 = { ...right, theta: clean(w.theta - d.theta) };
  return rotate(uniqueBy([right, minus, noX0], wKey), n + 1);
}

/** The whole game: the lesson's four training steps and the bonus Venusaur. */
export function buildGame(eta = LESSON_ETA): Encounter[] {
  const run = train(LESSON_SAMPLES, LESSON_INIT, eta);
  const out: Encounter[] = run.steps.map((s, i) => {
    const n = i + 1;
    return {
      n,
      epoch: s.epoch,
      pokemon: s.sample.id as PokeId,
      type: typeOf(s.sample.d),
      x1: s.sample.x1,
      x2: s.sample.x2,
      before: s.before,
      products: s.products,
      net: s.net,
      y: s.y,
      ball: ballOf(s.y),
      caught: s.e === 0,
      d: s.sample.d,
      e: s.e,
      after: s.after,
      netOptions: netChoices(s.before, s.sample.x1, s.sample.x2, n),
      updateOptions: s.e === 0 ? [] : updateChoices(s.before, s.sample.x1, s.sample.x2, eta, s.e, n),
    };
  });

  const w = run.final;
  const n = out.length + 1;
  const { net, y } = classify(w, VENUSAUR.x1, VENUSAUR.x2);
  out.push({
    n,
    epoch: null,
    pokemon: 'venusaur',
    type: 'grass',
    x1: VENUSAUR.x1,
    x2: VENUSAUR.x2,
    before: w,
    products: [clean(w.w1 * VENUSAUR.x1), clean(w.w2 * VENUSAUR.x2), clean(w.theta * X0)],
    net,
    y,
    ball: ballOf(y),
    caught: false,
    d: null,
    e: null,
    after: w,
    netOptions: netChoices(w, VENUSAUR.x1, VENUSAUR.x2, n),
    updateOptions: [],
  });
  return out;
}

/** Questions answered in each encounter: net and Ball always, error and update when it breaks free. */
export function questionCount(enc: Encounter[]): number {
  return enc.reduce((s, c) => s + 2 + (c.e !== null && c.e !== 0 ? 2 : 0), 0);
}

export const XP_PER_ANSWER = 10;
