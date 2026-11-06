// Checks that every number Misty says in the audio lesson comes from the lesson-1 code.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  BLASTOISE, CHARIZARD, LESSON_ETA, LESSON_INIT, LESSON_SAMPLES, classify, train,
} from '../src/lib/perceptron.ts';

const script = JSON.parse(readFileSync(new URL('../src/content/misty-lesson1.json', import.meta.url), 'utf8')) as {
  sections: { id: string; title: [string, string]; text: [string[], string[]] }[];
};
const audio = JSON.parse(readFileSync(new URL('../src/content/misty-lesson1-audio.json', import.meta.url), 'utf8')) as Record<
  'pt' | 'en', { duration: number; starts: number[]; checked: { paragraphs: number; number_errors: number; final_mp3_sections_ok: number } }
>;

const said = (id: string, lang: 0 | 1) => script.sections.find((s) => s.id === id)!.text[lang].join(' ');
/** How the script writes a number: Portuguese decimal comma, "minus"/"menos" for negatives. */
const spoken = (v: number, lang: 0 | 1) => {
  const s = String(Math.abs(v)).replace('.', lang === 0 ? ',' : '.');
  return v < 0 ? `${lang === 0 ? 'menos' : 'minus'} ${s}` : s;
};
function mentions(id: string, values: number[]) {
  for (const lang of [0, 1] as const) {
    const text = said(id, lang);
    for (const v of values) assert.ok(text.includes(spoken(v, lang)), `${id} [${lang ? 'en' : 'pt'}] should say ${spoken(v, lang)}`);
  }
}

const run = train(LESSON_SAMPLES, LESSON_INIT, LESSON_ETA);
const [c1, b1, c2, b2] = run.steps;

test('stats of the three Pokémon', () => {
  mentions('problema', [CHARIZARD.x1, CHARIZARD.x2, BLASTOISE.x1, BLASTOISE.x2]);
  mentions('venusaur', [80, 83]);
});

test('epoch 1, Charizard: net −0.5, error 1, weights → (100, 78, −0.5)', () => {
  assert.deepEqual([c1.net, c1.e], [-0.5, 1]);
  mentions('charizard', [LESSON_INIT.theta, c1.net, c1.after.w1, c1.after.w2, c1.after.theta]);
});

test('epoch 1, Blastoise: products 7800, net 15600.5, error −1, weights → (22, −22, 0.5)', () => {
  assert.deepEqual([b1.products[0], b1.products[1], b1.net, b1.e], [7800, 7800, 15600.5, -1]);
  mentions('blastoise', [7800, b1.net, b1.after.w1, b1.after.w2, b1.after.theta]);
});

test('epoch 2: 2200, 1716, nets 483.5 and −484.5, no errors', () => {
  assert.deepEqual([c2.products[0], -c2.products[1], c2.net, b2.net], [2200, 1716, 483.5, -484.5]);
  assert.deepEqual(run.epochs.map((e) => e.errors), [2, 0]);
  mentions('convergencia', [2200, 1716, c2.net, b2.net, 0.02]);
  assert.ok(Math.abs(run.final.theta / run.final.w1 - 0.02) < 0.005);
});

test('step size: η = 0.1 → (2.2, −2.2, 0.5), η = 0.01 → (0.22, −0.22, 0.5), two epochs each', () => {
  for (const [eta, w] of [[0.1, 2.2], [0.01, 0.22]] as const) {
    const r = train(LESSON_SAMPLES, LESSON_INIT, eta);
    assert.deepEqual(r.final, { w1: w, w2: -w, theta: 0.5 });
    assert.equal(r.epochs.length, 2);
    mentions('passo', [eta, w, -w]);
  }
});

test('Blastoise first: lucky guess (net −0.5), three epochs, same final weights', () => {
  const r = train([BLASTOISE, CHARIZARD], LESSON_INIT, LESSON_ETA);
  assert.equal(r.steps[0].net, -0.5);
  assert.equal(r.steps[0].e, 0);
  assert.equal(r.epochs.length, 3);
  assert.deepEqual(r.final, run.final);
});

test('Venusaur: 1760 − 1826 − 0.5 = −66.5 → Water', () => {
  const v = classify(run.final, 80, 83);
  assert.deepEqual(v, { net: -66.5, y: 0 });
  mentions('venusaur', [1760, 1826, v.net]);
});

test('audio timings: one start per section, increasing, within the duration, 10 to 15 minutes', () => {
  for (const lang of ['pt', 'en'] as const) {
    const a = audio[lang];
    assert.equal(a.starts.length, script.sections.length);
    assert.equal(a.starts[0], 0);
    for (let k = 1; k < a.starts.length; k++) assert.ok(a.starts[k] > a.starts[k - 1]);
    assert.ok(a.starts.at(-1)! < a.duration);
    assert.ok(a.duration > 9 * 60 && a.duration < 15 * 60, `${lang} duration ${a.duration}s`);
  }
});

test('the generator checked every paragraph and every chapter of the MP3 with no number errors', () => {
  const paragraphs = script.sections.reduce((n, s) => n + s.text[0].length, 0);
  for (const lang of ['pt', 'en'] as const) {
    assert.deepEqual(audio[lang].checked.paragraphs, paragraphs);
    assert.equal(audio[lang].checked.number_errors, 0);
    assert.equal(audio[lang].checked.final_mp3_sections_ok, script.sections.length);
  }
});
