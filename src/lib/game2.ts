/**
 * Pokédex mini-game, phase 2: the trainer catches Grass types with the MLP of lesson 2.
 *
 * Part A, "hand-made brain": the step network of lesson 2 (HAND_NET) meets Charizard, Blastoise and Venusaur.
 * The player works out the net of h1 and h2 and picks one of three Balls (Fire = h1, Water = h2, Grass = neither).
 * Part B, "a brain that learns": the sigmoid/softmax network with the lesson's initial weights (seed 2) meets
 * another Venusaur, loads the Water Ball and fails; the player does one backpropagation step by hand
 * (loss, output error, order of the calculation, update of the Grass bias with η = 2), and the next Venusaur is caught.
 * Wrong options are the usual mistakes, so each one teaches something.
 */
import {
  backprop, HAND_NET, initNet, sgdStep, softForward, STARTERS, stepForward, trainMlp, TYPES, type Net, type PokeType,
} from './mlp.ts';
import { MLP_CONFIG } from './model.ts';
import { clean } from './perceptron.ts';

export type Ball3 = PokeType;
export type Poke2 = 'charizard' | 'blastoise' | 'venusaur';

/** Rounds to 2 decimals, the precision the game asks for. */
export const r2 = (v: number) => clean(Math.round(v * 100) / 100);
const r3 = (v: number) => clean(Math.round(v * 1000) / 1000);

function rotate<T>(items: T[], k: number): T[] {
  const s = ((k % items.length) + items.length) % items.length;
  return [...items.slice(s), ...items.slice(0, s)];
}

function unique<T>(items: T[], key: (t: T) => string = String): T[] {
  const seen = new Set<string>();
  return items.filter((t) => {
    const k = key(t);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

const stats = (id: Poke2) => STARTERS.find((s) => s.id === id)!;

// --------------------------------------------------------------------------- //
// Part A: the hand-made step network
// --------------------------------------------------------------------------- //

export interface HandEncounter {
  kind: 'hand';
  n: number;
  pokemon: Poke2;
  type: PokeType;
  x1: number;
  x2: number;
  /** net of h1 = x1 − x2 − 11 and of h2 = x2 − x1 − 11 */
  hiddenNet: [number, number];
  h: [number, number];
  /** nets of the Fire, Water and Grass outputs */
  outputNet: [number, number, number];
  ball: Ball3;
  caught: boolean;
  /** correct net, "forgot the bias" and "added θ instead of θ·(−1)" */
  h1Options: number[];
  h2Options: number[];
}

/** Correct net plus the two common mistakes of lesson 1, in a rotated order. */
export function hiddenChoices(w: number[], theta: number, x1: number, x2: number, n: number): number[] {
  const a = clean(w[0] * x1 + w[1] * x2);
  const right = clean(a - theta);
  return rotate(unique([right, a, clean(a + theta), clean(right - 1)]).slice(0, 3), n);
}

export const HAND_POKEMON: Poke2[] = ['charizard', 'blastoise', 'venusaur'];

export function buildHand(): HandEncounter[] {
  return HAND_POKEMON.map((id, i) => {
    const s = stats(id);
    const f = stepForward(HAND_NET, s.x1, s.x2);
    const ball = f.type!;
    const [n1, n2] = HAND_NET.hidden;
    return {
      kind: 'hand',
      n: i + 1,
      pokemon: id,
      type: s.type,
      x1: s.x1,
      x2: s.x2,
      hiddenNet: [f.hiddenNet[0], f.hiddenNet[1]],
      h: [f.h[0], f.h[1]],
      outputNet: [f.outputNet[0], f.outputNet[1], f.outputNet[2]],
      ball,
      caught: ball === s.type,
      h1Options: hiddenChoices(n1.w, n1.theta, s.x1, s.x2, i),
      h2Options: hiddenChoices(n2.w, n2.theta, s.x1, s.x2, i + 1),
    };
  });
}

// --------------------------------------------------------------------------- //
// Part B: the network that learns (sigmoid hidden layer, softmax, cross-entropy)
// --------------------------------------------------------------------------- //

export type Vec3 = [number, number, number];

export interface LearnEncounter {
  kind: 'learn';
  n: number;
  pokemon: 'venusaur';
  /** 1: initial weights, breaks free; 2: after one backpropagation step, caught */
  attempt: 1 | 2;
  x1: number;
  x2: number;
  net: Net;
  /** hidden outputs and softmax probabilities, rounded to 2 decimals for display */
  h: [number, number];
  p: Vec3;
  ball: Ball3;
  caught: boolean;
  /** only for attempt 1 */
  train: Training | null;
}

export interface Training {
  /** L = −ln p_Grass */
  loss: number;
  lossOptions: number[];
  /** δo = p − d, d = (0, 0, 1) */
  delta: Vec3;
  deltaOptions: Vec3[];
  /** δh = (Σ v·δo)·h(1 − h), shown after the order question */
  deltaHidden: [number, number];
  /** the order question: the right answer is 'right' */
  orderOptions: OrderChoice[];
  eta: number;
  /** Grass output bias: φ before, ∂L/∂φ = δ·(−1), φ after = φ − η·∂L/∂φ */
  phi: number;
  gradPhi: number;
  phiAfter: number;
  phiOptions: number[];
  /** loss and Grass probability after the step (exact network, rounded) */
  lossAfter: number;
  grassAfter: number;
}

export type OrderChoice = 'right' | 'output-first' | 'hidden-first';

const GRASS = TYPES.indexOf('grass');
const ballOf = (p: number[]): Ball3 => TYPES[p.indexOf(Math.max(...p))];

export function learnChoices(eta: number, phi: number, gradPhi: number, n: number) {
  const right = r2(phi - eta * gradPhi);
  return rotate(unique([right, r2(phi + eta * gradPhi), r2(phi - gradPhi)]), n);
}

export function buildLearn(startN: number): LearnEncounter[] {
  const s = stats('venusaur');
  const eta = MLP_CONFIG.eta;
  const init = initNet(MLP_CONFIG.seed);
  const b = backprop(init, s);
  const after = sgdStep(init, b.grad, eta);
  const fa = softForward(after, s.x1, s.x2);

  const p = b.forward.p.map(r2) as Vec3;
  const delta = b.deltaOut.map(r2) as Vec3;
  const loss = r2(b.loss);
  const phi = init.output[GRASS].theta;
  const gradPhi = r2(-delta[GRASS]);
  const train: Training = {
    loss,
    lossOptions: rotate([loss, p[GRASS], r2(-loss)], 1),
    delta,
    // right, "d − p" (sign flipped) and "forgot to subtract d" (δ = p)
    deltaOptions: rotate([delta, delta.map((v) => r2(-v)) as Vec3, p], 2),
    deltaHidden: [r3(b.deltaHidden[0]), r3(b.deltaHidden[1])],
    orderOptions: ['output-first', 'right', 'hidden-first'],
    eta,
    phi,
    gradPhi,
    phiAfter: r2(phi - eta * gradPhi),
    phiOptions: learnChoices(eta, phi, gradPhi, startN),
    lossAfter: r2(-Math.log(fa.p[GRASS])),
    grassAfter: Math.round(100 * fa.p[GRASS]),
  };
  const first: LearnEncounter = {
    kind: 'learn', n: startN, pokemon: 'venusaur', attempt: 1, x1: s.x1, x2: s.x2, net: init,
    h: [r2(b.forward.h[0]), r2(b.forward.h[1])], p, ball: ballOf(b.forward.p), caught: ballOf(b.forward.p) === 'grass', train,
  };
  const second: LearnEncounter = {
    kind: 'learn', n: startN + 1, pokemon: 'venusaur', attempt: 2, x1: s.x1, x2: s.x2, net: after,
    h: [r2(fa.h[0]), r2(fa.h[1])], p: fa.p.map(r2) as Vec3, ball: ballOf(fa.p), caught: ballOf(fa.p) === 'grass', train: null,
  };
  return [first, second];
}

export type Encounter2 = HandEncounter | LearnEncounter;

/** The whole game: three encounters with the hand-made network, then two Venusaurs with the network that learns. */
export function buildGame2(): Encounter2[] {
  const hand = buildHand();
  return [...hand, ...buildLearn(hand.length + 1)];
}

/** Hand: h1, h2, Ball. Learn: Ball, plus loss, δ, order and update when it breaks free. */
export function questionCount2(enc: Encounter2[]): number {
  return enc.reduce((s, c) => s + (c.kind === 'hand' ? 3 : 1 + (c.train ? 4 : 0)), 0);
}

export const XP_PER_ANSWER = 10;

/** The lesson's full run (η = 2, 1500 epochs, full batch), summarized for the end screen. */
export function trainingSummary() {
  const r = trainMlp(STARTERS, initNet(MLP_CONFIG.seed), MLP_CONFIG.eta, MLP_CONFIG.epochs, [], 1);
  const v = stats('venusaur');
  return {
    epochs: MLP_CONFIG.epochs,
    firstAllRight: r.curve.find((c) => c.accuracy === 1)!.epoch,
    finalLoss: r.curve[r.curve.length - 1].loss,
    finalAccuracy: r.curve[r.curve.length - 1].accuracy,
    venusaurGrass: softForward(r.final, v.x1, v.x2).p[GRASS],
  };
}
