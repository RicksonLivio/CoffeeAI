/**
 * Rosenblatt's perceptron with a step activation, written to be read in class.
 *
 * Conventions (same as the lesson):
 *   inputs   x1 = Speed, x2 = Defense, x0 = -1 (fixed input that carries the bias)
 *   weights  w1, w2 and θ (the bias weight)
 *   net      = w1·x1 + w2·x2 + θ·x0 = w1·x1 + w2·x2 − θ
 *   output   y = 1 (Fire / Charizard) if net ≥ 0, else y = 0 (Water / Blastoise)
 *
 * Learning rule of the perceptron (error-correction rule):
 *   e  = d − y
 *   Δw1 = η·e·x1,  Δw2 = η·e·x2,  Δθ = η·e·x0 = −η·e
 * Weights only change when the sample is misclassified (e ≠ 0).
 */

export const X0 = -1;

export type Label = 0 | 1;

export interface Sample {
  id: string;
  /** x1: Speed */
  x1: number;
  /** x2: Defense */
  x2: number;
  /** desired output d */
  d: Label;
}

export interface Weights {
  w1: number;
  w2: number;
  theta: number;
}

export interface TrainStep {
  /** 1-based epoch */
  epoch: number;
  /** position of the sample inside the epoch (0-based) */
  index: number;
  sample: Sample;
  before: Weights;
  /** products of each input by its weight, in the order x1, x2, x0 */
  products: [number, number, number];
  net: number;
  y: Label;
  e: number;
  delta: Weights;
  after: Weights;
}

export interface EpochSummary {
  epoch: number;
  errors: number;
  weights: Weights;
}

export interface TrainResult {
  steps: TrainStep[];
  epochs: EpochSummary[];
  converged: boolean;
  final: Weights;
}

/** Base stats from the Pokémon games: Charizard Speed 100 / Defense 78, Blastoise 78 / 100. */
export const CHARIZARD: Sample = { id: 'charizard', x1: 100, x2: 78, d: 1 };
export const BLASTOISE: Sample = { id: 'blastoise', x1: 78, x2: 100, d: 0 };
export const LESSON_SAMPLES: Sample[] = [CHARIZARD, BLASTOISE];
export const LESSON_INIT: Weights = { w1: 0, w2: 0, theta: 0.5 };
export const LESSON_ETA = 1;

/** Cleans floating-point noise (0.1 + 0.2) so the class sees the numbers it would write by hand. */
export function clean(v: number): number {
  const r = Math.round(v * 1e9) / 1e9;
  return Object.is(r, -0) ? 0 : r;
}

export function netInput(w: Weights, x1: number, x2: number): number {
  return clean(w.w1 * x1 + w.w2 * x2 + w.theta * X0);
}

export function activate(net: number): Label {
  return net >= 0 ? 1 : 0;
}

/** One presentation of one sample: forward pass, error and weight update. */
export function trainStep(w: Weights, s: Sample, eta: number, epoch: number, index: number): TrainStep {
  const products: [number, number, number] = [clean(w.w1 * s.x1), clean(w.w2 * s.x2), clean(w.theta * X0)];
  const net = clean(products[0] + products[1] + products[2]);
  const y = activate(net);
  const e = s.d - y;
  const delta: Weights = { w1: clean(eta * e * s.x1), w2: clean(eta * e * s.x2), theta: clean(eta * e * X0) };
  const after: Weights = {
    w1: clean(w.w1 + delta.w1),
    w2: clean(w.w2 + delta.w2),
    theta: clean(w.theta + delta.theta),
  };
  return { epoch, index, sample: s, before: { ...w }, products, net, y, e, delta, after };
}

/**
 * Trains epoch by epoch, in the order of `samples`, until an epoch has no errors
 * (that checking epoch is kept in the trace, as in the lesson's epoch 2) or `maxEpochs` is reached.
 */
export function train(samples: Sample[], init: Weights, eta: number, maxEpochs = 20): TrainResult {
  const steps: TrainStep[] = [];
  const epochs: EpochSummary[] = [];
  let w: Weights = { ...init };
  let converged = false;
  for (let epoch = 1; epoch <= maxEpochs; epoch++) {
    let errors = 0;
    samples.forEach((s, i) => {
      const st = trainStep(w, s, eta, epoch, i);
      if (st.e !== 0) errors++;
      steps.push(st);
      w = st.after;
    });
    epochs.push({ epoch, errors, weights: { ...w } });
    if (errors === 0) {
      converged = true;
      break;
    }
  }
  return { steps, epochs, converged, final: w };
}

export function classify(w: Weights, x1: number, x2: number): { net: number; y: Label } {
  const net = netInput(w, x1, x2);
  return { net, y: activate(net) };
}

export interface Box {
  min: number;
  max: number;
}

export type Point = [number, number];

/**
 * Part of the square [min, max]² where the perceptron answers 1 (net ≥ 0):
 * the square clipped by the half-plane w1·x1 + w2·x2 − θ ≥ 0 (one step of Sutherland–Hodgman).
 */
export function fireRegion(w: Weights, box: Box): Point[] {
  const f = (p: Point) => w.w1 * p[0] + w.w2 * p[1] - w.theta;
  const square: Point[] = [
    [box.min, box.min],
    [box.max, box.min],
    [box.max, box.max],
    [box.min, box.max],
  ];
  const out: Point[] = [];
  for (let i = 0; i < square.length; i++) {
    const a = square[i];
    const b = square[(i + 1) % square.length];
    const fa = f(a);
    const fb = f(b);
    if (fa >= 0) out.push(a);
    if ((fa >= 0) !== (fb >= 0)) {
      const t = fa / (fa - fb);
      out.push([clean(a[0] + t * (b[0] - a[0])), clean(a[1] + t * (b[1] - a[1]))]);
    }
  }
  return out;
}

/** Segment of the decision boundary w1·x1 + w2·x2 = θ inside the square, or null if it misses it. */
export function boundarySegment(w: Weights, box: Box): [Point, Point] | null {
  if (w.w1 === 0 && w.w2 === 0) return null;
  const pts: Point[] = [];
  const add = (p: Point) => {
    if (p[0] < box.min - 1e-9 || p[0] > box.max + 1e-9 || p[1] < box.min - 1e-9 || p[1] > box.max + 1e-9) return;
    if (!pts.some((q) => Math.abs(q[0] - p[0]) < 1e-6 && Math.abs(q[1] - p[1]) < 1e-6)) pts.push(p);
  };
  for (const x1 of [box.min, box.max]) if (w.w2 !== 0) add([x1, clean((w.theta - w.w1 * x1) / w.w2)]);
  for (const x2 of [box.min, box.max]) if (w.w1 !== 0) add([clean((w.theta - w.w2 * x2) / w.w1), x2]);
  return pts.length >= 2 ? [pts[0], pts[1]] : null;
}
