import {
  LESSON_ETA, LESSON_INIT, LESSON_SAMPLES, X0, train,
  type EpochSummary, type Sample, type TrainStep, type Weights,
} from './perceptron.ts';
import { initNet, SCALE, STARTERS, trainMlp, type MlpTrainResult, type Net, type Poke } from './mlp.ts';

/** Paths of the cached models, relative to the site root (public/ in the repo). */
export const CACHED_MODEL_PATH = 'models/perceptron_pokemon/model.json';
export const CACHED_MLP_PATH = 'models/mlp_pokemon/model.json';

export interface CachedModel {
  name: string;
  version: number;
  task: string;
  features: { x1: string; x2: string; x0: string };
  classes: { '1': string; '0': string };
  config: { eta: number; init: Weights; max_epochs: number; activation: 'step'; x0: number };
  samples: Sample[];
  trace: TrainStep[];
  epochs: EpochSummary[];
  converged: boolean;
  final: Weights;
}

export const MAX_EPOCHS = 20;

export function buildCachedModel(
  samples: Sample[] = LESSON_SAMPLES,
  init: Weights = LESSON_INIT,
  eta: number = LESSON_ETA,
): CachedModel {
  const r = train(samples, init, eta, MAX_EPOCHS);
  return {
    name: 'perceptron_pokemon',
    version: 1,
    task: 'Fire (Charizard) vs Water (Blastoise) from base Speed and Defense',
    features: { x1: 'speed', x2: 'defense', x0: 'bias input (-1)' },
    classes: { '1': 'fire', '0': 'water' },
    config: { eta, init, max_epochs: MAX_EPOCHS, activation: 'step', x0: X0 },
    samples,
    trace: r.steps,
    epochs: r.epochs,
    converged: r.converged,
    final: r.final,
  };
}

export interface CachedMlp {
  name: string;
  version: number;
  task: string;
  config: {
    seed: number;
    eta: number;
    epochs: number;
    hidden: number;
    hidden_activation: 'sigmoid';
    output: 'softmax';
    loss: 'cross-entropy';
    input_scale: number;
    batch: 'full';
  };
  data: Poke[];
  init: Net;
  curve: MlpTrainResult['curve'];
  snapshots: MlpTrainResult['snapshots'];
  final: Net;
}

export const MLP_CONFIG = { seed: 2, eta: 2, epochs: 1500 };
export const MLP_SNAPSHOTS = [0, 10, 25, 50, 100, 150, 200, 300, 500, 1000, 1500];

export function buildCachedMlp(): CachedMlp {
  const init = initNet(MLP_CONFIG.seed);
  const r = trainMlp(STARTERS, init, MLP_CONFIG.eta, MLP_CONFIG.epochs, MLP_SNAPSHOTS, 5);
  return {
    name: 'mlp_pokemon',
    version: 1,
    task: 'Fire, Water or Grass from base Speed and Defense (the nine starter Pokémon)',
    config: {
      ...MLP_CONFIG,
      hidden: 2,
      hidden_activation: 'sigmoid',
      output: 'softmax',
      loss: 'cross-entropy',
      input_scale: SCALE,
      batch: 'full',
    },
    data: STARTERS,
    init,
    curve: r.curve,
    snapshots: r.snapshots,
    final: r.final,
  };
}
