/**
 * Small convolutional network for lesson 3, written to be read in class (no libraries).
 *
 * A tensor is a plain object { c, h, w, d } with d laid out as [channel][row][column].
 * Convolution kernels are laid out [filter][channel][row][column].
 *
 * Card network ("CardNet"), input 32×32×3 (a card squeezed into 32×32 pixels):
 *   conv 5×5 ×6 filters → ReLU → max pool 2×2 → conv 3×3 ×8 filters → ReLU → max pool 2×2 → flatten (288) → dense 3 → softmax
 * Output order: Fire, Water, Grass (same as lesson 2).
 */
import { rng } from './mlp.ts';

export interface Tensor { c: number; h: number; w: number; d: number[] }
export const tensor = (c: number, h: number, w: number, fill = 0): Tensor => ({ c, h, w, d: new Array(c * h * w).fill(fill) });
export const at = (t: Tensor, c: number, y: number, x: number) => t.d[(c * t.h + y) * t.w + x];

/** Output side of a convolution or pooling: ⌊(n + 2p − k) / s⌋ + 1 */
export const outSize = (n: number, k: number, pad: number, stride: number) => Math.floor((n + 2 * pad - k) / stride) + 1;

/** Slides the kernel over a single 2D grid (one channel): the picture in the lesson. */
export function conv2dGrid(grid: number[][], kernel: number[][], pad = 0, stride = 1, bias = 0): number[][] {
  const h = grid.length, w = grid[0].length, k = kernel.length;
  const oh = outSize(h, k, pad, stride), ow = outSize(w, k, pad, stride);
  const out: number[][] = [];
  for (let oy = 0; oy < oh; oy++) {
    const row: number[] = [];
    for (let ox = 0; ox < ow; ox++) {
      let s = bias;
      for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
        const y = oy * stride + ky - pad, x = ox * stride + kx - pad;
        if (y >= 0 && y < h && x >= 0 && x < w) s += grid[y][x] * kernel[ky][kx];
      }
      row.push(s);
    }
    out.push(row);
  }
  return out;
}

export type PoolKind = 'max' | 'avg' | 'min';
export function pool2dGrid(grid: number[][], size = 2, stride = size, kind: PoolKind = 'max'): number[][] {
  const oh = outSize(grid.length, size, 0, stride), ow = outSize(grid[0].length, size, 0, stride);
  return Array.from({ length: oh }, (_, oy) => Array.from({ length: ow }, (_, ox) => {
    const v: number[] = [];
    for (let ky = 0; ky < size; ky++) for (let kx = 0; kx < size; kx++) v.push(grid[oy * stride + ky][ox * stride + kx]);
    return kind === 'max' ? Math.max(...v) : kind === 'min' ? Math.min(...v) : v.reduce((a, b) => a + b, 0) / v.length;
  }));
}

export interface ConvLayer { f: number; c: number; k: number; w: number[]; b: number[] }

export function conv(x: Tensor, L: ConvLayer, pad = 0, stride = 1): Tensor {
  const oh = outSize(x.h, L.k, pad, stride), ow = outSize(x.w, L.k, pad, stride);
  const y = tensor(L.f, oh, ow);
  for (let f = 0; f < L.f; f++) for (let oy = 0; oy < oh; oy++) for (let ox = 0; ox < ow; ox++) {
    let s = L.b[f];
    for (let c = 0; c < L.c; c++) for (let ky = 0; ky < L.k; ky++) {
      const iy = oy * stride + ky - pad;
      if (iy < 0 || iy >= x.h) continue;
      for (let kx = 0; kx < L.k; kx++) {
        const ix = ox * stride + kx - pad;
        if (ix < 0 || ix >= x.w) continue;
        s += x.d[(c * x.h + iy) * x.w + ix] * L.w[((f * L.c + c) * L.k + ky) * L.k + kx];
      }
    }
    y.d[(f * oh + oy) * ow + ox] = s;
  }
  return y;
}

export const relu = (x: Tensor): Tensor => ({ ...x, d: x.d.map((v) => (v > 0 ? v : 0)) });

/** 2×2 max pooling, stride 2. `arg` remembers which input won (needed to send the error back). */
export function maxPool(x: Tensor, size = 2): { y: Tensor; arg: number[] } {
  const oh = Math.floor(x.h / size), ow = Math.floor(x.w / size);
  const y = tensor(x.c, oh, ow), arg: number[] = new Array(y.d.length);
  for (let c = 0; c < x.c; c++) for (let oy = 0; oy < oh; oy++) for (let ox = 0; ox < ow; ox++) {
    let best = -Infinity, bi = 0;
    for (let ky = 0; ky < size; ky++) for (let kx = 0; kx < size; kx++) {
      const i = (c * x.h + oy * size + ky) * x.w + ox * size + kx;
      if (x.d[i] > best) { best = x.d[i]; bi = i; }
    }
    const o = (c * oh + oy) * ow + ox;
    y.d[o] = best; arg[o] = bi;
  }
  return { y, arg };
}

export interface Dense { n: number; m: number; w: number[]; b: number[] } // w[j * m + i]: output j, input i

export function softmax(z: number[]): number[] {
  const mx = Math.max(...z), e = z.map((v) => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export interface CardNet { conv1: ConvLayer; conv2: ConvLayer; fc: Dense }

export function initCardNet(seed: number): CardNet {
  const r = rng(seed);
  const he = (fanIn: number) => (r() * 2 - 1) * Math.sqrt(6 / fanIn);
  const mk = (f: number, c: number, k: number): ConvLayer => ({ f, c, k, w: Array.from({ length: f * c * k * k }, () => he(c * k * k)), b: new Array(f).fill(0) });
  return {
    conv1: mk(6, 3, 5), conv2: mk(8, 6, 3),
    fc: { n: 3, m: 288, w: Array.from({ length: 3 * 288 }, () => he(288)), b: [0, 0, 0] },
  };
}

export interface Forward {
  x: Tensor; z1: Tensor; a1: Tensor; p1: Tensor; arg1: number[];
  z2: Tensor; a2: Tensor; p2: Tensor; arg2: number[]; flat: number[]; logits: number[]; p: number[];
}

export function forward(net: CardNet, x: Tensor): Forward {
  const z1 = conv(x, net.conv1), a1 = relu(z1), { y: p1, arg: arg1 } = maxPool(a1);
  const z2 = conv(p1, net.conv2), a2 = relu(z2), { y: p2, arg: arg2 } = maxPool(a2);
  const flat = p2.d, { n, m, w, b } = net.fc;
  const logits = Array.from({ length: n }, (_, j) => b[j] + flat.reduce((s, v, i) => s + v * w[j * m + i], 0));
  return { x, z1, a1, p1, arg1, z2, a2, p2, arg2, flat, logits, p: softmax(logits) };
}

export const loss = (p: number[], target: number) => -Math.log(Math.max(p[target], 1e-12));

/** Gradient of the cross-entropy loss for one image, same shapes as the network. */
export function backward(net: CardNet, f: Forward, target: number): { grad: CardNet; loss: number } {
  const g: CardNet = {
    conv1: { ...net.conv1, w: new Array(net.conv1.w.length).fill(0), b: new Array(net.conv1.f).fill(0) },
    conv2: { ...net.conv2, w: new Array(net.conv2.w.length).fill(0), b: new Array(net.conv2.f).fill(0) },
    fc: { ...net.fc, w: new Array(net.fc.w.length).fill(0), b: [0, 0, 0] },
  };
  const dz = f.p.map((v, j) => v - (j === target ? 1 : 0)); // softmax + cross-entropy: p − d
  const dflat = new Array(net.fc.m).fill(0);
  for (let j = 0; j < net.fc.n; j++) {
    g.fc.b[j] = dz[j];
    for (let i = 0; i < net.fc.m; i++) { g.fc.w[j * net.fc.m + i] = dz[j] * f.flat[i]; dflat[i] += dz[j] * net.fc.w[j * net.fc.m + i]; }
  }
  // through pool 2 and ReLU 2 (only the winner of each block, only where the ReLU was on)
  const da2 = new Array(f.a2.d.length).fill(0);
  f.arg2.forEach((src, o) => { da2[src] += dflat[o]; });
  const dz2 = da2.map((v, i) => (f.z2.d[i] > 0 ? v : 0));
  const dp1 = new Array(f.p1.d.length).fill(0);
  backConv(net.conv2, f.p1, dz2, f.z2.h, f.z2.w, g.conv2, dp1);
  const da1 = new Array(f.a1.d.length).fill(0);
  f.arg1.forEach((src, o) => { da1[src] += dp1[o]; });
  const dz1 = da1.map((v, i) => (f.z1.d[i] > 0 ? v : 0));
  backConv(net.conv1, f.x, dz1, f.z1.h, f.z1.w, g.conv1, null);
  return { grad: g, loss: loss(f.p, target) };
}

function backConv(L: ConvLayer, x: Tensor, dz: number[], oh: number, ow: number, g: ConvLayer, dx: number[] | null) {
  for (let f = 0; f < L.f; f++) for (let oy = 0; oy < oh; oy++) for (let ox = 0; ox < ow; ox++) {
    const d = dz[(f * oh + oy) * ow + ox];
    if (d === 0) continue;
    g.b[f] += d;
    for (let c = 0; c < L.c; c++) for (let ky = 0; ky < L.k; ky++) for (let kx = 0; kx < L.k; kx++) {
      const xi = (c * x.h + oy + ky) * x.w + ox + kx, wi = ((f * L.c + c) * L.k + ky) * L.k + kx;
      g.w[wi] += d * x.d[xi];
      if (dx) dx[xi] += d * L.w[wi];
    }
  }
}

export const flatNet = (n: CardNet): number[] => [...n.conv1.w, ...n.conv1.b, ...n.conv2.w, ...n.conv2.b, ...n.fc.w, ...n.fc.b];
export const countParams = (n: CardNet) => flatNet(n).length;

/** Pixels (0–255, RGB interleaved, as in canvas ImageData) → centered tensor 3×size×size in [−0.5, 0.5]. */
export function toTensor(rgb: ArrayLike<number>, size = 32): Tensor {
  const t = tensor(3, size, size);
  for (let i = 0; i < size * size; i++) for (let c = 0; c < 3; c++) t.d[c * size * size + i] = rgb[i * 3 + c] / 255 - 0.5;
  return t;
}

export const CLASSES = ['fire', 'water', 'grass'] as const;
export type CardClass = (typeof CLASSES)[number];

export interface CardSample { id: string; name: string; type: CardClass; file: string; split: 'train' | 'test'; set: string; px: string }

/** Shift, flip and light changes: more "different" cards from the few we have. */
export function augment(t: Tensor, r: () => number): Tensor {
  const size = t.h, dx = Math.floor(r() * 5) - 2, dy = Math.floor(r() * 5) - 2, flip = r() < 0.5;
  const gain = 0.8 + r() * 0.4, shift = (r() - 0.5) * 0.16;
  const o = tensor(3, size, size);
  for (let c = 0; c < 3; c++) for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const sy = Math.min(size - 1, Math.max(0, y + dy)), sx0 = Math.min(size - 1, Math.max(0, x + dx)), sx = flip ? size - 1 - sx0 : sx0;
    o.d[(c * size + y) * size + x] = t.d[(c * size + sy) * size + sx] * gain + shift;
  }
  return o;
}

export interface TrainConfig { seed: number; epochs: number; lr: number; momentum: number }

export function evaluate(net: CardNet, set: { x: Tensor; y: number }[]) {
  let hit = 0, ls = 0;
  const confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const s of set) {
    const f = forward(net, s.x), pred = f.p.indexOf(Math.max(...f.p));
    if (pred === s.y) hit++;
    confusion[s.y][pred]++;
    ls += loss(f.p, s.y);
  }
  return { accuracy: hit / set.length, loss: ls / set.length, confusion };
}

/** SGD with momentum, one image at a time, with augmentation. Deterministic for a given seed. */
export function trainCardNet(train: { x: Tensor; y: number }[], test: { x: Tensor; y: number }[], cfg: TrainConfig) {
  const net = initCardNet(cfg.seed), r = rng(cfg.seed + 1);
  const vel = new Array(countParams(net)).fill(0);
  const curve: { epoch: number; loss: number; trainAcc: number; testAcc: number }[] = [];
  const apply = (grad: CardNet) => {
    const g = flatNet(grad);
    let k = 0;
    const upd = (arr: number[]) => { for (let i = 0; i < arr.length; i++, k++) { vel[k] = cfg.momentum * vel[k] - cfg.lr * g[k]; arr[i] += vel[k]; } };
    upd(net.conv1.w); upd(net.conv1.b); upd(net.conv2.w); upd(net.conv2.b); upd(net.fc.w); upd(net.fc.b);
  };
  for (let e = 1; e <= cfg.epochs; e++) {
    const order = train.map((_, i) => i).sort(() => r() - 0.5);
    let total = 0;
    for (const i of order) {
      const x = augment(train[i].x, r), f = forward(net, x), { grad, loss: l } = backward(net, f, train[i].y);
      total += l; apply(grad);
    }
    if (e === 1 || e % 5 === 0 || e === cfg.epochs)
      curve.push({ epoch: e, loss: total / train.length, trainAcc: evaluate(net, train).accuracy, testAcc: evaluate(net, test).accuracy });
  }
  return { net, curve };
}

export const roundNet = (n: CardNet, digits = 5): CardNet => {
  const q = (a: number[]) => a.map((v) => Number(v.toFixed(digits)));
  return { conv1: { ...n.conv1, w: q(n.conv1.w), b: q(n.conv1.b) }, conv2: { ...n.conv2, w: q(n.conv2.w), b: q(n.conv2.b) }, fc: { ...n.fc, w: q(n.fc.w), b: q(n.fc.b) } };
};

// --------------------------------------------------------------------------- //
// Lesson 3: shared constants (the page, the tests and scripts/train-cnn.ts use these)
// --------------------------------------------------------------------------- //
export const CNN_CONFIG: TrainConfig = { seed: 11, epochs: 40, lr: 0.002, momentum: 0.9 };
export const CNN_MODEL_PATH = 'models/cnn_pokemon/model.json';
export const CNN_CARDS_PATH = 'models/cnn_pokemon/cards32.json';

/** Shapes of the card network, from the input to the output: the "big square to small square" picture. */
export const CARDNET_SHAPES = [
  { key: 'input', c: 3, s: 32 },
  { key: 'conv1', c: 6, s: 28, k: 5 },
  { key: 'pool1', c: 6, s: 14 },
  { key: 'conv2', c: 8, s: 12, k: 3 },
  { key: 'pool2', c: 8, s: 6 },
  { key: 'flat', c: 1, s: 288 },
  { key: 'out', c: 1, s: 3 },
] as const;

/** 6×6 patch with a bright left side and a dark right side (a vertical edge). */
export const EDGE_PATCH: number[][] = [
  [9, 9, 9, 1, 1, 1],
  [9, 9, 9, 1, 1, 1],
  [9, 9, 8, 2, 1, 1],
  [9, 8, 8, 2, 1, 0],
  [8, 8, 7, 2, 1, 0],
  [8, 7, 7, 1, 0, 0],
];

export const KERNELS: Record<'vertical' | 'horizontal' | 'blur' | 'sharpen', number[][]> = {
  vertical: [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
  horizontal: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
  blur: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
  sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
};

export const POOL_PATCH: number[][] = [
  [1, 3, 2, 9],
  [5, 6, 1, 0],
  [7, 2, 4, 4],
  [0, 1, 8, 3],
];

/** Backpropagation through one convolution, small enough to do by hand. */
export const BACK_X: number[][] = [[1, 2, 0], [0, 1, 3], [2, 1, 1]];
export const BACK_K: number[][] = [[1, -1], [0, 2]];
export const BACK_DELTA: number[][] = [[0.5, -1], [0, 0.25]];

/** ∂L/∂K[a][b] = Σ_ij δ[i][j] · x[i + a][j + b]  (the input "convolved" with the error) */
export function kernelGrad(x: number[][], delta: number[][], k: number): number[][] {
  return Array.from({ length: k }, (_, a) => Array.from({ length: k }, (_, b) =>
    delta.reduce((s, row, i) => s + row.reduce((t, d, j) => t + d * x[i + a][j + b], 0), 0)));
}

/** ∂L/∂x: every input pixel collects δ · K from each window it was part of. */
export function inputGrad(K: number[][], delta: number[][], h: number, w: number): number[][] {
  const g = Array.from({ length: h }, () => new Array(w).fill(0));
  delta.forEach((row, i) => row.forEach((d, j) => K.forEach((kr, a) => kr.forEach((kv, b) => { g[i + a][j + b] += d * kv; }))));
  return g;
}

/** Max pooling sends the error back only to the cell that won; the other three get 0. */
export function maxPoolBack(grid: number[][], delta: number[][], size = 2): number[][] {
  const g = grid.map((r) => r.map(() => 0));
  delta.forEach((row, oy) => row.forEach((d, ox) => {
    let by = oy * size, bx = ox * size;
    for (let y = oy * size; y < oy * size + size; y++) for (let x = ox * size; x < ox * size + size; x++)
      if (grid[y][x] > grid[by][bx]) { by = y; bx = x; }
    g[by][bx] += d;
  }));
  return g;
}

export interface CachedCnn {
  config: TrainConfig;
  params: number;
  net: CardNet;
  curve: { epoch: number; loss: number; trainAcc: number; testAcc: number }[];
  train: { n: number; accuracy: number; confusion: number[][] };
  test: { n: number; accuracy: number; confusion: number[][] };
  samples: { id: string; name: string; type: CardClass; file: string; split: 'train' | 'test'; set: string; p: number[] }[];
}

/** base64 RGB bytes → numbers (works in the browser and in Node). */
export function decodePx(b64: string): Uint8Array {
  const bin = atob(b64), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Average colour of a tensor (R, G, B in 0–1). */
export const meanColor = (t: Tensor): number[] => [0, 1, 2].map((c) => {
  const n = t.h * t.w;
  let s = 0;
  for (let i = 0; i < n; i++) s += t.d[c * n + i] + 0.5;
  return s / n;
});

/** Baseline with no convolution: each type is its average colour; a card goes to the nearest one. */
export function colorBaseline(train: { x: Tensor; y: number }[]) {
  const cent = [0, 1, 2].map((k) => {
    const m = train.filter((s) => s.y === k).map((s) => meanColor(s.x));
    return [0, 1, 2].map((c) => m.reduce((a, v) => a + v[c], 0) / m.length);
  });
  const predictColor = (x: Tensor) => {
    const m = meanColor(x), d = cent.map((c) => c.reduce((a, v, i) => a + (v - m[i]) ** 2, 0));
    return d.indexOf(Math.min(...d));
  };
  return { cent, predictColor };
}
