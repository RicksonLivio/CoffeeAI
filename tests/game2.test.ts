// Checks that the Pokédex mini-game, phase 2, plays exactly the lesson-2 numbers.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildGame2, questionCount2, trainingSummary, type HandEncounter, type LearnEncounter } from '../src/lib/game2.ts';
import { backprop, HAND_NET, initNet, sgdStep, softForward, STARTERS, stepForward } from '../src/lib/mlp.ts';
import { MLP_CONFIG } from '../src/lib/model.ts';

const game = buildGame2();
const hand = game.filter((g): g is HandEncounter => g.kind === 'hand');
const learn = game.filter((g): g is LearnEncounter => g.kind === 'learn');

test('three hand-made encounters, then two Venusaurs with the network that learns', () => {
  assert.deepEqual(game.map((g) => [g.kind, g.pokemon]), [
    ['hand', 'charizard'], ['hand', 'blastoise'], ['hand', 'venusaur'], ['learn', 'venusaur'], ['learn', 'venusaur'],
  ]);
  assert.deepEqual(learn.map((g) => g.attempt), [1, 2]);
});

test('hand-made network: the hidden nets and Balls of lesson 2, every capture works', () => {
  assert.deepEqual(hand.map((g) => g.hiddenNet), [[11, -33], [-33, 11], [-14, -8]]);
  assert.deepEqual(hand.map((g) => g.h), [[1, 0], [0, 1], [0, 0]]);
  assert.deepEqual(hand.map((g) => g.ball), ['fire', 'water', 'grass']);
  assert.ok(hand.every((g) => g.caught));
  for (const g of hand) {
    const s = STARTERS.find((q) => q.id === g.pokemon)!;
    assert.deepEqual(g.hiddenNet, stepForward(HAND_NET, s.x1, s.x2).hiddenNet);
  }
});

test('every hidden-net question has 3 distinct options, the right one and the "forgot the bias" mistake', () => {
  for (const g of hand) {
    for (const [opts, right, j] of [[g.h1Options, g.hiddenNet[0], 0], [g.h2Options, g.hiddenNet[1], 1]] as const) {
      assert.equal(opts.length, 3);
      assert.equal(new Set(opts).size, 3);
      assert.ok(opts.includes(right));
      assert.ok(opts.includes(right + HAND_NET.hidden[j].theta), `${g.pokemon} h${j + 1}: forgot θ`);
    }
  }
  assert.ok(new Set(hand.map((g) => g.h1Options.indexOf(g.hiddenNet[0]))).size > 1, 'right answer moves around');
});

test('learning network, attempt 1: Water Ball (p = 0.14, 0.51, 0.35), Venusaur breaks free', () => {
  const [a] = learn;
  assert.deepEqual(a.p, [0.14, 0.51, 0.35]);
  assert.deepEqual(a.h, [0.61, 0.61]);
  assert.equal(a.ball, 'water');
  assert.equal(a.caught, false);
  assert.deepEqual(a.net, initNet(MLP_CONFIG.seed));
});

test('the backpropagation step: L = 1.05, δo = (0.14, 0.51, −0.65), φ_Grass 0.2 → −1.1, Grass 96 %', () => {
  const t = learn[0].train!;
  const venusaur = STARTERS.find((s) => s.id === 'venusaur')!;
  const b = backprop(initNet(MLP_CONFIG.seed), venusaur);
  assert.equal(t.loss, 1.05);
  assert.ok(Math.abs(b.loss - t.loss) < 0.005);
  assert.deepEqual(t.delta, [0.14, 0.51, -0.65]);
  assert.deepEqual(t.deltaHidden, [-0.065, -0.046]);
  assert.deepEqual([t.phi, t.gradPhi, t.eta, t.phiAfter], [0.2, 0.65, 2, -1.1]);
  // the rounded answer matches the exact update of the network
  const exact = sgdStep(initNet(MLP_CONFIG.seed), b.grad, MLP_CONFIG.eta).output[2].theta;
  assert.ok(Math.abs(exact - t.phiAfter) < 0.005, `exact φ ${exact}`);
  assert.deepEqual([t.grassAfter, t.lossAfter], [96, 0.04]);
});

test('answer options: right one present, usual mistakes offered, 3 distinct each', () => {
  const t = learn[0].train!;
  assert.deepEqual([...t.lossOptions].sort(), [-1.05, 0.35, 1.05].sort());
  assert.equal(new Set(t.deltaOptions.map(String)).size, 3);
  assert.ok(t.deltaOptions.some((v) => String(v) === String([0.14, 0.51, -0.65])));
  assert.ok(t.deltaOptions.some((v) => String(v) === String([-0.14, -0.51, 0.65])), 'd − p offered');
  assert.deepEqual([...t.phiOptions].sort(), [-1.1, -0.45, 1.5].sort());
  assert.ok(t.orderOptions.includes('right'));
  assert.equal(new Set(t.orderOptions).size, 3);
});

test('learning network, attempt 2: after one step Venusaur goes in the Grass Ball', () => {
  const [, b] = learn;
  const venusaur = STARTERS.find((s) => s.id === 'venusaur')!;
  const init = initNet(MLP_CONFIG.seed);
  const after = sgdStep(init, backprop(init, venusaur).grad, MLP_CONFIG.eta);
  assert.deepEqual(b.net, after);
  assert.deepEqual(b.p, softForward(after, 80, 83).p.map((v) => Math.round(v * 100) / 100));
  assert.equal(b.p[2], 0.96);
  assert.equal(b.ball, 'grass');
  assert.equal(b.caught, true);
  assert.equal(b.train, null);
});

test('question count is 15 and the end screen numbers match the lesson run', () => {
  assert.equal(questionCount2(game), 15);
  const s = trainingSummary();
  assert.equal(s.epochs, 1500);
  assert.equal(s.firstAllRight, 52);
  assert.equal(s.finalAccuracy, 1);
  assert.equal(Number(s.finalLoss.toFixed(4)), 0.0068);
  assert.equal(Math.round(100 * s.venusaurGrass), 99);
});
