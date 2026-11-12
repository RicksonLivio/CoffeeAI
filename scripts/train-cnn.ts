// Trains the lesson-3 card network (src/lib/cnn.ts) on the cards in public/models/cnn_pokemon/cards32.json
// (32×32 versions of the card pictures, made by scripts/prepare_cnn_cards.py) and writes the cached model
// public/models/cnn_pokemon/model.json. Run with: npm run train-cnn
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import {
  CLASSES, CNN_CONFIG, countParams, decodePx, evaluate, forward, roundNet, toTensor, trainCardNet, type CardSample,
} from '../src/lib/cnn.ts';

const cards: CardSample[] = JSON.parse(readFileSync(new URL('../public/models/cnn_pokemon/cards32.json', import.meta.url), 'utf8'));
const prep = (c: CardSample) => ({ x: toTensor(decodePx(c.px)), y: CLASSES.indexOf(c.type) });
const train = cards.filter((c) => c.split === 'train').map(prep);
const test = cards.filter((c) => c.split === 'test').map(prep);

const { net, curve } = trainCardNet(train, test, CNN_CONFIG);
const q = roundNet(net);
const trainEval = evaluate(q, train), testEval = evaluate(q, test);
const samples = cards.map((c) => {
  const p = forward(q, prep(c).x).p.map((v) => Number(v.toFixed(4)));
  return { id: c.id, name: c.name, type: c.type, file: c.file, split: c.split, set: c.set, p };
});

const out = new URL('../public/models/cnn_pokemon/model.json', import.meta.url);
mkdirSync(new URL('.', out), { recursive: true });
writeFileSync(out, JSON.stringify({
  config: CNN_CONFIG, params: countParams(q), net: q, curve,
  train: { n: train.length, accuracy: trainEval.accuracy, confusion: trainEval.confusion },
  test: { n: test.length, accuracy: testEval.accuracy, confusion: testEval.confusion },
  samples,
}) + '\n');
console.log(`params=${countParams(q)} train=${trainEval.accuracy.toFixed(3)} (${train.length}) test=${testEval.accuracy.toFixed(3)} (${test.length})`);
console.log(JSON.stringify(testEval.confusion), curve.map((c) => `${c.epoch}:${c.loss.toFixed(2)}/${c.testAcc.toFixed(2)}`).join(' '));
