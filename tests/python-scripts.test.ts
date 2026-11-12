// The downloadable Python examples (public/code/*.py) must print the same results as the site.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const dir = fileURLToPath(new URL('../public/code/', import.meta.url));
const py = spawnSync('python3', ['--version']).status === 0 ? 'python3' : null;
const run = (file: string) => {
  const r = spawnSync(py!, [file], { cwd: dir, encoding: 'utf8', timeout: 120_000 });
  assert.equal(r.status, 0, r.stderr);
  return r.stdout;
};

test('perceptron_pokemon.py prints the lesson 1 numbers', { skip: !py && 'python3 not found' }, () => {
  const out = run('perceptron_pokemon.py');
  for (const s of ['= -0.5', '= 15600.5', 'w1 = 100 + (1·-1·78) = 22', '= 483.5', '= -484.5', 'net = -66.5 => y = 0 (Água)',
    'Convergiu: True, em 2 épocas', 'w1 = 2.2, w2 = -2.2, θ = 0.5']) assert.ok(out.includes(s), s);
});

test('mlp_pokemon.py prints the lesson 2 numbers', { skip: !py && 'python3 not found' }, () => {
  const out = run('mlp_pokemon.py');
  for (const s of ['Menor número de erros em 100 épocas: 1', 'Venusaur    net_h = (-14, -8) -> h = (0, 0) -> saídas [0, 0, 1] -> Planta ✓',
    '= 22.5 para qualquer Pokémon', 'p = softmax = (0.1441, 0.5062, 0.3496)', 'δ_Planta = -0.6504', 'δ_h1 = -0.0651',
    'w1,1 = 0.5000 − 2·∂L/∂w1,1 = 0.5548', 'erro médio da época 1: 1.2079', 'época 1500: erro médio L = 0.0068 · acertos 9/9',
    'Acerta os nove pela primeira vez na época 52.', 'η = 0.1: termina em', '-> mínimo local', 'L = 0.113 -> mínimo global',
    'η =   10: erro final', 'acertos 3/9', 'δ_g2 = 0.2890', 'Acerta os nove pela primeira vez na época 225.']) assert.ok(out.includes(s), s);
});

// PyTorch versions: run only where torch is installed (set PYTHON_TORCH to a python that has it).
const pyTorch = process.env.PYTHON_TORCH ?? 'python3';
const hasTorch = spawnSync(pyTorch, ['-c', 'import torch'], { encoding: 'utf8' }).status === 0;
const runTorch = (file: string) => {
  const r = spawnSync(pyTorch, [file], { cwd: dir, encoding: 'utf8', timeout: 300_000, env: { ...process.env, CUDA_VISIBLE_DEVICES: '' } });
  assert.equal(r.status, 0, r.stderr);
  return r.stdout;
};

test('perceptron_pokemon_torch.py: same lesson 1 numbers', { skip: !hasTorch && 'torch not installed' }, () => {
  const out = runTorch('perceptron_pokemon_torch.py');
  for (const s of ['net = -0.5', 'net = 15600.5', 'w = [22.0, -22.0, 0.5]', 'net = 483.5', 'net = -484.5', 'net = -66.5']) assert.ok(out.includes(s), s);
});

test('mlp_pokemon_torch.py: manual gradients equal autograd; both versions reach 9/9', { skip: !hasTorch && 'torch not installed' }, () => {
  const out = runTorch('mlp_pokemon_torch.py');
  const diffs = [...out.matchAll(/maior diferença = ([0-9.e+-]+)/g)].map((m) => Number(m[1]));
  assert.equal(diffs.length, 2);
  for (const d of diffs) assert.ok(d < 1e-12);
  assert.ok(out.includes('L = média de −ln p_certo = 1.2079'));
  assert.ok(out.includes('L = média de −ln p_certo = 1.2514'));
  assert.equal((out.match(/acurácia 1\.000\n  probabilidades do Venusaur/g) ?? []).length, 4);
});

test('backprop_notebook.py: backprop equals finite differences; the toy and the saddles', { skip: !py && 'python3 not found' }, () => {
  const out = run('backprop_notebook.py');
  for (const s of ['custo: diferença finita = 20 avaliações de J para 10 pesos', '0    0.5000    0.5000    0.5625   -0.7500   -0.7500',
    '1    0.5750    0.5750    0.4481', 'autovalores (-2.0, 2.0): sela estrita', 'autovalores (0.0, 4.0)', 'parada na sela (∇J = 0)',
    'J < 0,001 no passo 45', 'J < 0,001 no passo 26', '[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]']) assert.ok(out.includes(s), s);
  const diff = Number(/maior diferença: ([0-9.e+-]+)/.exec(out)![1]);
  assert.ok(diff < 1e-7);
});

test('backprop_notebook_torch.py: eq. 8 equals autograd; Hessians at the saddles', { skip: !hasTorch && 'torch not installed' }, () => {
  const out = runTorch('backprop_notebook_torch.py');
  const diffs = [...out.matchAll(/camadas, larguras .*: maior diferença = ([0-9.e+-]+)/g)].map((m) => Number(m[1]));
  assert.equal(diffs.length, 3);
  for (const d of diffs) assert.ok(d < 1e-10);
  assert.ok(out.includes('autovalores tensor([-2.,  2.]) -> sela estrita'));
  assert.ok(out.includes('autovalores tensor([0., 0., 0.]) -> sela ruim'));
});

// Lesson 3: the scripts read the local cards and model (--dir) instead of downloading them from the site.
const CNN_DIR = ['--dir', '../models/cnn_pokemon'];

test('cnn_pokemon.py prints the lesson 3 numbers and the Pokédex gets 23/24', { skip: !py && 'python3 not found' }, () => {
  const r = spawnSync(py!, ['cnn_pokemon.py', ...CNN_DIR], { cwd: dir, encoding: 'utf8', timeout: 120_000 });
  assert.equal(r.status, 0, r.stderr);
  for (const s of ['posição (0, 0): net = 9·1 + 9·0 + 9·-1 + 9·1 + 9·0 + 9·-1 + 9·1 + 9·0 + 8·-1 = 1', '= 23',
    'padding 1, stride 2: saída = ⌊(6 + 2·1 − 3)/2⌋ + 1 = 3  (conferido: 3)', '∂L/∂K[1][1] = 0.5·1 + -1·3 + 0·1 + 0.25·1 = -2.25',
    'K[1][1] = 2 − 0.1·(-2.25) = 2.225', '1763 pesos', 'Orbeetle V             real Planta  Pokédex Fogo', 'acertos no teste: 23/24']) assert.ok(r.stdout.includes(s), s);
  assert.ok(Number(/probabilidades do site: ([0-9.e+-]+)/.exec(r.stdout)![1]) < 1e-4);
});

test('cnn_pokemon_torch.py: site weights give the same probabilities; hand gradient equals autograd', { skip: !hasTorch && 'torch not installed' }, () => {
  const r = spawnSync(pyTorch, ['cnn_pokemon_torch.py', ...CNN_DIR], { cwd: dir, encoding: 'utf8', timeout: 300_000, env: { ...process.env, CUDA_VISIBLE_DEVICES: '' } });
  assert.equal(r.status, 0, r.stderr);
  assert.ok(Number(/site \(93 cartas\): ([0-9.e+-]+)/.exec(r.stdout)![1]) < 1e-4);
  for (const s of ['pesos: 1763', 'acertos: treino 69/69, teste 23/24', '∂L/∂K (autograd) = [[-1.25, 1.75], [-0.75, -2.25]]',
    '∂L/∂x (autograd) = [[0.5, -1.5, 1.0], [0.0, 1.25, -2.25], [0.0, 0.0, 0.5]]']) assert.ok(r.stdout.includes(s), s);
});
