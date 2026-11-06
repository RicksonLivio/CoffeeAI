// Checks that the Pokédex mini-game plays exactly the lesson-1 numbers.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildGame, ERROR_OPTIONS, questionCount } from '../src/lib/game.ts';

const game = buildGame();

test('five encounters: the four training steps of lesson 1, then Venusaur', () => {
  assert.deepEqual(game.map((g) => [g.pokemon, g.epoch]), [
    ['charizard', 1], ['blastoise', 1], ['charizard', 2], ['blastoise', 2], ['venusaur', null],
  ]);
});

test('net, Ball and capture follow the lesson table', () => {
  assert.deepEqual(game.map((g) => g.net), [-0.5, 15600.5, 483.5, -484.5, -66.5]);
  assert.deepEqual(game.map((g) => g.ball), ['water', 'fire', 'fire', 'water', 'water']);
  assert.deepEqual(game.map((g) => g.caught), [false, false, true, true, false]);
  assert.deepEqual(game.map((g) => g.e), [1, -1, 0, 0, null]);
});

test('weights after each encounter match the lesson', () => {
  assert.deepEqual(game.map((g) => g.after), [
    { w1: 100, w2: 78, theta: -0.5 },
    { w1: 22, w2: -22, theta: 0.5 },
    { w1: 22, w2: -22, theta: 0.5 },
    { w1: 22, w2: -22, theta: 0.5 },
    { w1: 22, w2: -22, theta: 0.5 },
  ]);
});

test('every net question has 3 distinct options and includes the right one', () => {
  for (const g of game) {
    assert.equal(g.netOptions.length, 3);
    assert.equal(new Set(g.netOptions).size, 3);
    assert.ok(g.netOptions.includes(g.net), `${g.pokemon} ${g.n}`);
  }
  // the right answer is not always in the same slot
  assert.ok(new Set(game.map((g) => g.netOptions.indexOf(g.net))).size > 1);
});

test('update questions only when the Pokémon breaks free, with the right weights among 3', () => {
  for (const g of game) {
    if (g.e === 1 || g.e === -1) {
      assert.equal(g.updateOptions.length, 3);
      assert.equal(new Set(g.updateOptions.map((w) => JSON.stringify(w))).size, 3);
      assert.ok(g.updateOptions.some((w) => JSON.stringify(w) === JSON.stringify(g.after)));
    } else {
      assert.equal(g.updateOptions.length, 0);
    }
  }
  assert.ok(game[0].updateOptions.some((w) => w.w1 === -100), 'sign mistake offered');
  assert.ok(game[0].updateOptions.some((w) => w.w1 === 100 && w.theta === 1.5), 'x0 = −1 mistake offered');
});

test('error options cover −1, 0, 1 and the question count is 14', () => {
  assert.deepEqual(ERROR_OPTIONS, [-1, 0, 1]);
  assert.equal(questionCount(game), 14);
});
