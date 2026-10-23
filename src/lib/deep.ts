/**
 * Network with any number of hidden layers, for lesson 2 (epoch player, two hidden layers).
 *
 * layers[0] reads the inputs (Speed/100, Defense/100); every following layer reads the previous one;
 * the last layer is the softmax output (Fire, Water, Grass). Hidden layers use the sigmoid.
 * Each neuron has a bias weight θ on a fixed input −1, as in lessons 1 and 2.
 *
 * Backpropagation (cross-entropy + softmax):
 *   δ_out = p − d
 *   δ_l[j] = (Σ_k W_{l+1}[k][j] · δ_{l+1}[k]) · a_l[j] · (1 − a_l[j])      (from the last hidden layer back)
 *   ∂L/∂W_l[j][i] = δ_l[j] · a_{l−1}[i]       ∂L/∂θ_l[j] = δ_l[j] · (−1)
 * All the δ are computed with the current (old) weights; only then every weight is updated at once.
 */
import { clean, X0 } from './perceptron.ts';
import { oneHot, rng, SCALE, sigmoid, softmax, TYPES, type Net, type Neuron, type Poke, type PokeType } from './mlp.ts';

export interface DeepNet {
  layers: Neuron[][];
}

export const toDeep = (n: Net): DeepNet => ({ layers: [n.hidden, n.output] });

export interface DeepForward {
  /** a[0] = scaled input; a[l + 1] = output of layer l (the last one is p) */
  a: number[][];
  /** z[l] = net input of layer l */
  z: number[][];
  p: number[];
}

const net = (n: Neuron, x: number[]) => n.w.reduce((s, w, i) => s + w * x[i], 0) + n.theta * X0;

export function deepForward(m: DeepNet, x1: number, x2: number): DeepForward {
  const a: number[][] = [[x1 / SCALE, x2 / SCALE]];
  const z: number[][] = [];
  m.layers.forEach((layer, l) => {
    const zl = layer.map((n) => net(n, a[l]));
    z.push(zl);
    a.push(l === m.layers.length - 1 ? softmax(zl) : zl.map(sigmoid));
  });
  return { a, z, p: a[a.length - 1] };
}

export const deepLoss = (f: DeepForward, t: PokeType) => -Math.log(Math.max(1e-300, f.p[TYPES.indexOf(t)]));

export interface DeepBackprop {
  forward: DeepForward;
  loss: number;
  d: number[];
  /** delta[l] = error of layer l, same indexing as layers (last = output) */
  delta: number[][];
  grad: DeepNet;
}

export function deepBackprop(m: DeepNet, s: Poke): DeepBackprop {
  const f = deepForward(m, s.x1, s.x2);
  const d = oneHot(s.type);
  const L = m.layers.length;
  const delta: number[][] = new Array(L);
  delta[L - 1] = f.p.map((p, k) => p - d[k]);
  for (let l = L - 2; l >= 0; l--) {
    const act = f.a[l + 1];
    delta[l] = act.map((h, j) => m.layers[l + 1].reduce((acc, n, k) => acc + n.w[j] * delta[l + 1][k], 0) * h * (1 - h));
  }
  const grad: DeepNet = {
    layers: m.layers.map((layer, l) => layer.map((_, j) => ({ w: f.a[l].map((x) => delta[l][j] * x), theta: delta[l][j] * X0 }))),
  };
  return { forward: f, loss: deepLoss(f, s.type), d, delta, grad };
}

export const mapDeep = (a: DeepNet, b: DeepNet, fn: (x: number, y: number) => number): DeepNet => ({
  layers: a.layers.map((layer, l) => layer.map((n, j) => ({
    w: n.w.map((v, i) => fn(v, b.layers[l][j].w[i])),
    theta: fn(n.theta, b.layers[l][j].theta),
  }))),
});

export const flatDeep = (m: DeepNet) => m.layers.flatMap((layer) => layer.flatMap((n) => [...n.w, n.theta]));

export interface EpochStep {
  epoch: number;
  /** weights at the start of the epoch */
  before: DeepNet;
  /** one backprop per Pokémon, all with the `before` weights */
  samples: DeepBackprop[];
  /** mean loss and mean gradient over the data */
  loss: number;
  grad: DeepNet;
  gradNorm: number;
  accuracy: number;
  /** weights after w ← w − η · grad */
  after: DeepNet;
}

export function deepPredict(m: DeepNet, x1: number, x2: number): PokeType {
  const p = deepForward(m, x1, x2).p;
  return TYPES[p.indexOf(Math.max(...p))];
}

export const deepAccuracy = (m: DeepNet, data: Poke[]) => data.filter((s) => deepPredict(m, s.x1, s.x2) === s.type).length / data.length;

/** One full-batch epoch: forward + backward for every Pokémon with the same weights, mean gradient, one update. */
export function deepEpoch(m: DeepNet, data: Poke[], eta: number, epoch: number): EpochStep {
  const samples = data.map((s) => deepBackprop(m, s));
  const n = data.length;
  let sum = samples[0].grad;
  for (const b of samples.slice(1)) sum = mapDeep(sum, b.grad, (x, y) => x + y);
  const grad = mapDeep(sum, sum, (x) => x / n);
  const after = mapDeep(m, grad, (w, g) => w - eta * g);
  return {
    epoch,
    before: m,
    samples,
    loss: samples.reduce((acc, b) => acc + b.loss, 0) / n,
    grad,
    gradNorm: Math.sqrt(flatDeep(grad).reduce((acc, g) => acc + g * g, 0)),
    accuracy: deepAccuracy(m, data),
    after,
  };
}

/** h[e] = weights after e updates (h[0] = init), for e = 0..epochs. */
export function deepHistory(init: DeepNet, data: Poke[], eta: number, epochs: number): DeepNet[] {
  const h: DeepNet[] = [init];
  let m = init;
  for (let e = 1; e <= epochs; e++) {
    m = deepEpoch(m, data, eta, e).after;
    h.push(m);
  }
  return h;
}

export function deepCurve(init: DeepNet, data: Poke[], eta: number, epochs: number, every = 5) {
  const curve: { epoch: number; loss: number; accuracy: number }[] = [];
  let m = init;
  // entry e = loss with the weights after e updates, same convention as trainMlp
  for (let e = 0; e <= epochs; e++) {
    const st = deepEpoch(m, data, eta, e);
    if (e % every === 0 || e === epochs) curve.push({ epoch: e, loss: st.loss, accuracy: st.accuracy });
    if (e < epochs) m = st.after;
  }
  return { curve, final: m };
}

/** Initial weights uniform in [−1, 1], rounded to one decimal; sizes = neurons per layer, e.g. [2, 2, 3]. */
export function initDeep(seed: number, sizes: number[], nInputs = 2): DeepNet {
  const r = rng(seed);
  const u = () => clean(Math.round((r() * 2 - 1) * 10) / 10);
  let prev = nInputs;
  const layers = sizes.map((k) => {
    const layer = Array.from({ length: k }, () => ({ w: Array.from({ length: prev }, u), theta: u() }));
    prev = k;
    return layer;
  });
  return { layers };
}
