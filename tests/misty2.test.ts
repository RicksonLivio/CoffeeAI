// Checks that every number Misty says in the Lesson 2 audio comes from the lesson-2 code.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  backprop, fireWaterHull, HAND_NET, initNet, insideConvex, meanGrad, POKE_NAME, SCALE, sgdStep, softForward,
  STARTERS, stepForward, trainMlp, accuracy,
} from '../src/lib/mlp.ts';
import { deepAccuracy, deepBackprop } from '../src/lib/deep.ts';
import { buildCachedMlp2, MLP_CONFIG } from '../src/lib/model.ts';
import { train, type Sample } from '../src/lib/perceptron.ts';

const script = JSON.parse(readFileSync(new URL('../src/content/misty-lesson2.json', import.meta.url), 'utf8')) as {
  sections: { id: string; title: [string, string]; text: [string[], string[]] }[];
};
const audio = JSON.parse(readFileSync(new URL('../src/content/misty-lesson2-audio.json', import.meta.url), 'utf8')) as Record<
  'pt' | 'en', { duration: number; starts: number[]; checked: { paragraphs: number; number_errors: number; final_mp3_sections_ok: number } }
>;

const said = (id: string, lang: 0 | 1) => {
  const s = script.sections.find((q) => q.id === id);
  assert.ok(s, `section ${id}`);
  return s.text[lang].join(' ');
};
/** How the script writes a number: Portuguese decimal comma, "minus"/"menos" for negatives. */
const spoken = (v: number, lang: 0 | 1) => {
  const s = String(Math.abs(v)).replace('.', lang === 0 ? ',' : '.');
  return v < 0 ? `${lang === 0 ? 'menos' : 'minus'} ${s}` : s;
};
function mentions(id: string, values: number[]) {
  for (const lang of [0, 1] as const) {
    const text = said(id, lang);
    for (const v of values) {
      const w = spoken(v, lang);
      // whole token: "0,35" must not match inside "0,355", "11" not inside "11,8"
      const re = new RegExp(`(^|[^\\d,.])${w.replace(/[.]/g, '\\.')}(?![\\d]|[,.]\\d)`);
      assert.ok(re.test(text), `${id} [${lang ? 'en' : 'pt'}] should say ${w}`);
    }
  }
}
function names(id: string, ids: string[]) {
  for (const lang of [0, 1] as const) for (const n of ids) assert.ok(said(id, lang).includes(POKE_NAME[n]), `${id} should name ${n}`);
}
const round = (v: number, d: number) => Number(v.toFixed(d));
const pct = (v: number) => Math.round(100 * v);

const venusaur = STARTERS.find((s) => s.id === 'venusaur')!;
const charizard = STARTERS.find((s) => s.id === 'charizard')!;
const init = initNet(MLP_CONFIG.seed);

test('the problem: Speed − Defense is 22, −22, −4/−3; Ivysaur and Venusaur inside the Fire/Water area', () => {
  const diffs = (t: string) => [...new Set(STARTERS.filter((s) => s.type === t).map((s) => s.x1 - s.x2))];
  assert.deepEqual(diffs('fire'), [22]);
  assert.deepEqual(diffs('water'), [-22]);
  assert.deepEqual(diffs('grass').sort((a, b) => a - b), [-4, -3]);
  mentions('problema', [22, -22, -4, -3]);
  const hull = fireWaterHull();
  const corner = (p: number[]) => STARTERS.find((s) => s.x1 === p[0] && s.x2 === p[1])!.id;
  const corners = hull.map(corner);
  assert.deepEqual([...corners].sort(), ['blastoise', 'charizard', 'charmander', 'squirtle']);
  names('problema', corners);
  const inside = STARTERS.filter((s) => s.type === 'grass' && insideConvex(hull, [s.x1, s.x2])).map((s) => s.id);
  assert.deepEqual(inside, ['ivysaur', 'venusaur']);
  names('problema', inside);
});

test('a "Grass or not" perceptron, 100 epochs: never zero errors, best is 1', () => {
  const samples: Sample[] = STARTERS.map((s) => ({ id: s.id, x1: s.x1, x2: s.x2, d: s.type === 'grass' ? 1 : 0 }));
  const r = train(samples, { w1: 0, w2: 0, theta: 0.5 }, 1, 100);
  const errs = r.epochs.map((e) => e.errors);
  assert.equal(errs.length, 100);
  assert.equal(Math.min(...errs), 1);
  mentions('problema', [100, 1]);
});

test('the hand-made network: threshold 11 (half of 22), Venusaur −14, −8 → Grass 0.5; Charizard 11, −33 → Fire', () => {
  assert.equal(HAND_NET.hidden[0].theta, 11);
  assert.equal(HAND_NET.hidden[1].theta, 11);
  mentions('oculta', [11, 22]);
  const v = stepForward(HAND_NET, venusaur.x1, venusaur.x2);
  assert.deepEqual([v.hiddenNet, v.h, v.outputNet, v.type], [[-14, -8], [0, 0], [-0.5, -0.5, 0.5], 'grass']);
  const c = stepForward(HAND_NET, charizard.x1, charizard.x2);
  assert.deepEqual([c.hiddenNet, c.h, c.outputNet[0], c.type], [[11, -33], [1, 0], 0.5, 'fire']);
  mentions('mao', [80, 83, 11, -14, -8, 0.5, -0.5, 100, 78, -33]);
  assert.ok(STARTERS.every((s) => stepForward(HAND_NET, s.x1, s.x2).type === s.type), 'all nine right');
});

test('sigmoid: derivative at most 0.25, inputs divided by 100 (80 → 0.8)', () => {
  assert.equal(SCALE, 100);
  assert.equal(venusaur.x1 / SCALE, 0.8);
  mentions('sigmoide', [0.25, 100, 80, 0.8]);
});

test('initial weights: Venusaur 14 % / 51 % / 35 %, loss 1.05; mean loss 1.21; 3 of 9 right', () => {
  const f = softForward(init, venusaur.x1, venusaur.x2);
  assert.deepEqual(f.p.map(pct), [14, 51, 35]);
  const b = backprop(init, venusaur);
  assert.equal(round(b.loss, 2), 1.05);
  assert.equal(round(meanGrad(init, STARTERS).loss, 2), 1.21);
  assert.equal(Math.round(9 * accuracy(init, STARTERS)), 3);
  assert.ok(STARTERS.every((s) => {
    const p = softForward(init, s.x1, s.x2).p;
    return p.indexOf(Math.max(...p)) === 1;
  }), 'guesses Water for everyone');
  mentions('softmax', [14, 51, 35, 0.35, 1.05, 1.21, 3, 9]);
});

test('one backpropagation step on Venusaur: δo, δh, φ 0.2 → −1.1, Grass 96 %, loss 0.04', () => {
  const b = backprop(init, venusaur);
  assert.deepEqual(b.deltaOut.map((v) => round(v, 2)), [0.14, 0.51, -0.65]);
  assert.deepEqual(b.deltaHidden.map((v) => round(v, 3)), [-0.065, -0.046]);
  const after = sgdStep(init, b.grad, MLP_CONFIG.eta);
  assert.equal(init.output[2].theta, 0.2);
  assert.equal(round(b.grad.output[2].theta, 2), 0.65);
  assert.equal(round(after.output[2].theta, 1), -1.1);
  const fa = softForward(after, venusaur.x1, venusaur.x2);
  assert.equal(pct(fa.p[2]), 96);
  assert.equal(round(-Math.log(fa.p[2]), 2), 0.04);
  const weights = init.hidden.length * 3 + init.output.length * 3;
  assert.equal(weights, 15);
  assert.equal(MLP_CONFIG.eta, 2);
  mentions('backprop', [0.14, 0.51, -0.65, 0.35, -0.065, -0.046, 15, 2, 0.2, 0.65, -1.1, 96, 1.05, 0.04]);
});

const run = trainMlp(STARTERS, init, MLP_CONFIG.eta, MLP_CONFIG.epochs, [], 1);

test('full run: all nine at epoch 52, stumbles 109–158, steady from 159, loss 0.0068, Venusaur 99 % Grass', () => {
  const right = run.curve.map((c) => Math.round(9 * c.accuracy));
  assert.equal(MLP_CONFIG.epochs, 1500);
  assert.equal(right.indexOf(9), 52);
  const bad = right.map((r, e) => (r < 9 ? e : -1)).filter((e) => e > 52);
  assert.equal(bad[0], 109);
  assert.equal(bad.at(-1), 158);
  assert.equal(Math.min(...right.slice(109, 159)), 3);
  assert.equal(round(run.curve.at(-1)!.loss, 4), 0.0068);
  assert.equal(pct(softForward(run.final, venusaur.x1, venusaur.x2).p[2]), 99);
  mentions('treino', [1500, 52, 109, 158, 3, 159, 0.0068, 99]);
});

test('what the hidden layer learned: (11.8, −12.6) Fire detector and (−10.2, 9.9) Water detector', () => {
  const [h1, h2] = run.final.hidden;
  assert.deepEqual(h1.w.map((v) => round(v, 1)), [11.8, -12.6]);
  assert.deepEqual(h2.w.map((v) => round(v, 1)), [-10.2, 9.9]);
  mentions('treino', [11.8, -12.6, -10.2, 9.9]);
});

test('step size in the MLP: η = 0.2 slow (epoch 507, loss 0.23), η = 10 stuck (never 9, ends 3 of 9, loss 2.98)', () => {
  const slow = trainMlp(STARTERS, init, 0.2, MLP_CONFIG.epochs, [], 1).curve;
  assert.equal(slow.find((c) => c.accuracy === 1)!.epoch, 507);
  assert.equal(round(slow.at(-1)!.loss, 2), 0.23);
  const big = trainMlp(STARTERS, init, 10, MLP_CONFIG.epochs, [], 1).curve;
  assert.ok(big.every((c) => c.accuracy < 1));
  assert.equal(Math.round(9 * big.at(-1)!.accuracy), 3);
  assert.equal(round(big.at(-1)!.loss, 2), 2.98);
  mentions('passo', [0.1, 0.7, 1.5, 0.2, 507, 0.23, 2, 10, 3, 9, 2.98]);
});

test('two hidden layers: 21 weights, |δ| 0.88 → 0.29 → 0.06 for Venusaur, all nine right', () => {
  const m = buildCachedMlp2();
  const n = m.init.layers.reduce((acc, l) => acc + l.reduce((s, q) => s + q.w.length + 1, 0), 0);
  assert.equal(n, 21);
  const mags = deepBackprop(m.init, venusaur).delta.map((d) => round(Math.max(...d.map(Math.abs)), 2));
  assert.deepEqual(mags, [0.06, 0.29, 0.88]);
  assert.equal(deepAccuracy(m.final, STARTERS), 1);
  mentions('duas', [21, 15, 0.88, 0.29, 0.06, 0.25]);
});

test('recap: η = 2, 1500 epochs', () => {
  mentions('resumo', [2, 1500]);
});

test('audio timings: one start per section, increasing, within the duration, 9 to 17 minutes', () => {
  for (const lang of ['pt', 'en'] as const) {
    const a = audio[lang];
    assert.equal(a.starts.length, script.sections.length);
    assert.equal(a.starts[0], 0);
    for (let k = 1; k < a.starts.length; k++) assert.ok(a.starts[k] > a.starts[k - 1]);
    assert.ok(a.starts.at(-1)! < a.duration);
    assert.ok(a.duration > 9 * 60 && a.duration < 17 * 60, `${lang} duration ${a.duration}s`);
  }
});

test('the generator checked every paragraph and every chapter of the MP3 with no number errors', () => {
  const paragraphs = script.sections.reduce((n, s) => n + s.text[0].length, 0);
  for (const lang of ['pt', 'en'] as const) {
    assert.equal(script.sections.every((s) => s.text[0].length === s.text[1].length), true);
    assert.deepEqual(audio[lang].checked.paragraphs, paragraphs);
    assert.equal(audio[lang].checked.number_errors, 0);
    assert.equal(audio[lang].checked.final_mp3_sections_ok, script.sections.length);
  }
});
