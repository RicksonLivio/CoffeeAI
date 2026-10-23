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
