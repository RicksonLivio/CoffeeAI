/**
 * Multilayer perceptron (MLP) for lesson 2: Fire, Water and Grass from Speed and Defense.
 *
 * Two hidden neurons and three output neurons, the same layout in both parts of the lesson:
 *   Part A, "by hand": step activations and weights we choose ourselves (integer arithmetic).
 *   Part B, "learning": sigmoid hidden layer, softmax output, cross-entropy loss, trained with
 *   backpropagation and gradient descent.
 *
 * Convention from lesson 1: every neuron has a fixed input x0 = −1 with weight θ, so
 *   net = Σ wᵢ·xᵢ + θ·(−1) = Σ wᵢ·xᵢ − θ.
 */
import { activate, clean, X0, type Label } from './perceptron.ts';

export type PokeType = 'fire' | 'water' | 'grass';
export const TYPES: PokeType[] = ['fire', 'water', 'grass'];

export interface Poke {
  id: string;
  /** x1: base Speed */
  x1: number;
  /** x2: base Defense */
  x2: number;
  type: PokeType;
}

/** The three starter families, with base Speed and Defense from the main-series games. */
export const STARTERS: Poke[] = [
  { id: 'charmander', x1: 65, x2: 43, type: 'fire' },
  { id: 'charmeleon', x1: 80, x2: 58, type: 'fire' },
  { id: 'charizard', x1: 100, x2: 78, type: 'fire' },
  { id: 'squirtle', x1: 43, x2: 65, type: 'water' },
  { id: 'wartortle', x1: 58, x2: 80, type: 'water' },
  { id: 'blastoise', x1: 78, x2: 100, type: 'water' },
  { id: 'bulbasaur', x1: 45, x2: 49, type: 'grass' },
  { id: 'ivysaur', x1: 60, x2: 63, type: 'grass' },
  { id: 'venusaur', x1: 80, x2: 83, type: 'grass' },
];

export const POKE_NAME: Record<string, string> = {
  charmander: 'Charmander', charmeleon: 'Charmeleon', charizard: 'Charizard',
  squirtle: 'Squirtle', wartortle: 'Wartortle', blastoise: 'Blastoise',
  bulbasaur: 'Bulbasaur', ivysaur: 'Ivysaur', venusaur: 'Venusaur',
};

/** One neuron: weights for its inputs and the bias weight θ (input −1). */
export interface Neuron {
  w: number[];
  theta: number;
}

export interface Net {
  hidden: Neuron[];
  output: Neuron[];
}

export function neuronNet(n: Neuron, x: number[]): number {
  return n.w.reduce((s, wi, i) => s + wi * x[i], 0) + n.theta * X0;
}

// --------------------------------------------------------------------------- //
// Part A: step activations, weights chosen by hand
// --------------------------------------------------------------------------- //

/**
 * h1 "Fire detector": fires when Speed − Defense ≥ 11 (half of the gap of 22 that every Fire starter has).
 * h2 "Water detector": fires when Defense − Speed ≥ 11.
 * Outputs: Fire = h1, Water = h2, Grass = neither (net = 0.5 − h1 − h2).
 */
export const HAND_NET: Net = {
  hidden: [
    { w: [1, -1], theta: 11 },
    { w: [-1, 1], theta: 11 },
  ],
  output: [
    { w: [1, 0], theta: 0.5 },
    { w: [0, 1], theta: 0.5 },
    { w: [-1, -1], theta: -0.5 },
  ],
};

export interface StepForward {
  x: number[];
  hiddenNet: number[];
  h: Label[];
  outputNet: number[];
  o: Label[];
  /** predicted type, or null if the outputs are not exactly one-hot */
  type: PokeType | null;
}

export function stepForward(net: Net, x1: number, x2: number): StepForward {
  const x = [x1, x2];
  const hiddenNet = net.hidden.map((n) => clean(neuronNet(n, x)));
  const h = hiddenNet.map(activate);
  const outputNet = net.output.map((n) => clean(neuronNet(n, h)));
  const o = outputNet.map(activate);
  const on = o.reduce<number>((s, v) => s + v, 0);
  return { x, hiddenNet, h, outputNet, o, type: on === 1 ? TYPES[o.indexOf(1)] : null };
}

// --------------------------------------------------------------------------- //
// Part B: sigmoid + softmax, trained with backpropagation
// --------------------------------------------------------------------------- //

/** Inputs are divided by 100 so the sigmoid starts away from saturation (Speed 80 → 0.8). */
export const SCALE = 100;

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export interface SoftForward {
  x: number[];
  zh: number[];
  h: number[];
  zo: number[];
  p: number[];
}

export function softForward(net: Net, x1: number, x2: number): SoftForward {
  const x = [x1 / SCALE, x2 / SCALE];
  const zh = net.hidden.map((n) => neuronNet(n, x));
  const h = zh.map(sigmoid);
  const zo = net.output.map((n) => neuronNet(n, h));
  return { x, zh, h, zo, p: softmax(zo) };
}

export function oneHot(t: PokeType): number[] {
  return TYPES.map((k) => (k === t ? 1 : 0));
}

/** Cross-entropy L = −ln p_correct. */
export function loss(f: SoftForward, t: PokeType): number {
  return -Math.log(Math.max(1e-300, f.p[TYPES.indexOf(t)]));
}

export interface Backprop {
  forward: SoftForward;
  loss: number;
  d: number[];
  /** output error δo = p − d (softmax with cross-entropy) */
  deltaOut: number[];
  /** hidden error δh = (Σ_k w_kj·δo_k) · h(1 − h) */
  deltaHidden: number[];
  /** ∂L/∂w and ∂L/∂θ for every neuron, same shape as Net */
  grad: Net;
}

/** Forward pass, then the gradient of the loss for one sample, layer by layer from the output back. */
export function backprop(net: Net, s: Poke): Backprop {
  const f = softForward(net, s.x1, s.x2);
  const d = oneHot(s.type);
  const deltaOut = f.p.map((p, k) => p - d[k]);
  const deltaHidden = f.h.map((h, j) =>
    net.output.reduce((acc, n, k) => acc + n.w[j] * deltaOut[k], 0) * h * (1 - h));
  // the bias input is −1, so ∂L/∂θ = δ·(−1)
  const grad: Net = {
    output: net.output.map((_, k) => ({ w: f.h.map((h) => deltaOut[k] * h), theta: deltaOut[k] * X0 })),
    hidden: net.hidden.map((_, j) => ({ w: f.x.map((x) => deltaHidden[j] * x), theta: deltaHidden[j] * X0 })),
  };
  return { forward: f, loss: loss(f, s.type), d, deltaOut, deltaHidden, grad };
}

const mapNet = (a: Net, b: Net, fn: (x: number, y: number) => number): Net => ({
  hidden: a.hidden.map((n, j) => ({ w: n.w.map((v, i) => fn(v, b.hidden[j].w[i])), theta: fn(n.theta, b.hidden[j].theta) })),
  output: a.output.map((n, k) => ({ w: n.w.map((v, i) => fn(v, b.output[k].w[i])), theta: fn(n.theta, b.output[k].theta) })),
});

/** Gradient-descent step w ← w − η·∂L/∂w. */
export function sgdStep(net: Net, grad: Net, eta: number): Net {
  return mapNet(net, grad, (w, g) => w - eta * g);
}

export function meanGrad(net: Net, data: Poke[]): { grad: Net; loss: number } {
  let acc: Net | null = null;
  let L = 0;
  for (const s of data) {
    const b = backprop(net, s);
    L += b.loss;
    acc = acc ? mapNet(acc, b.grad, (x, y) => x + y) : b.grad;
  }
  const n = data.length;
  return { grad: mapNet(acc!, acc!, (x) => x / n), loss: L / n };
}

export function predict(net: Net, x1: number, x2: number): PokeType {
  const p = softForward(net, x1, x2).p;
  return TYPES[p.indexOf(Math.max(...p))];
}

export function accuracy(net: Net, data: Poke[]): number {
  return data.filter((s) => predict(net, s.x1, s.x2) === s.type).length / data.length;
}

/** Small deterministic PRNG (mulberry32) so the initial weights are the same on every machine. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Initial weights uniform in [−scale, scale], rounded to one decimal so the first step can be followed by hand. */
export function initNet(seed: number, nHidden = 2, scale = 1): Net {
  const r = rng(seed);
  const u = () => clean(Math.round((r() * 2 - 1) * scale * 10) / 10);
  return {
    hidden: Array.from({ length: nHidden }, () => ({ w: [u(), u()], theta: u() })),
    output: TYPES.map(() => ({ w: Array.from({ length: nHidden }, u), theta: u() })),
  };
}

export interface Snapshot {
  epoch: number;
  loss: number;
  accuracy: number;
  net: Net;
}

export interface MlpTrainResult {
  curve: { epoch: number; loss: number; accuracy: number }[];
  snapshots: Snapshot[];
  final: Net;
}

/** Full-batch gradient descent: every epoch averages the gradient over all nine Pokémon. */
export function trainMlp(
  data: Poke[], init: Net, eta: number, epochs: number, snapshotAt: number[], curveEvery = 10,
): MlpTrainResult {
  let net = init;
  const curve: MlpTrainResult['curve'] = [];
  const snapshots: Snapshot[] = [];
  const snapSet = new Set(snapshotAt);
  for (let e = 0; e <= epochs; e++) {
    const { grad, loss: L } = meanGrad(net, data);
    if (e % curveEvery === 0 || e === epochs) curve.push({ epoch: e, loss: L, accuracy: accuracy(net, data) });
    if (snapSet.has(e)) snapshots.push({ epoch: e, loss: L, accuracy: accuracy(net, data), net });
    if (e < epochs) net = sgdStep(net, grad, eta);
  }
  return { curve, snapshots, final: net };
}

// --------------------------------------------------------------------------- //
// Geometry: why one line is not enough
// --------------------------------------------------------------------------- //

export type Pt = [number, number];

/** Convex hull (Andrew's monotone chain), counter-clockwise, without repeated points. */
export function convexHull(points: Pt[]): Pt[] {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Pt[] = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper: Pt[] = [];
  for (const q of [...p].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

/** Strictly inside a convex polygon given counter-clockwise. */
export function insideConvex(poly: Pt[], q: Pt): boolean {
  return poly.every((a, i) => {
    const b = poly[(i + 1) % poly.length];
    return (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) > 0;
  });
}

/** Region enclosed by the Fire and Water starters. Any line that leaves all of them on one side leaves this whole region there too. */
export function fireWaterHull(data: Poke[] = STARTERS): Pt[] {
  return convexHull(data.filter((s) => s.type !== 'grass').map((s) => [s.x1, s.x2] as Pt));
}
