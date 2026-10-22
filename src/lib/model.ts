import {
  LESSON_ETA, LESSON_INIT, LESSON_SAMPLES, X0, train,
  type EpochSummary, type Sample, type TrainStep, type Weights,
} from './perceptron.ts';

/** Path of the cached model, relative to the site root (public/ in the repo). */
export const CACHED_MODEL_PATH = 'models/perceptron_pokemon/model.json';

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
