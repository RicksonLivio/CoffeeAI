// Static "API": the site has no backend, it reads files published with it (GitHub Pages).
// Same idea as web_app/frontend/src/api.ts in the NFH app, with files instead of FastAPI routes.
import {
  buildCachedMlp, buildCachedModel, CACHED_MLP_PATH, CACHED_MODEL_PATH, type CachedMlp, type CachedModel,
} from './lib/model';

const BASE = import.meta.env.BASE_URL;

export const assets = {
  photo: `${BASE}Rick.jpeg`,
  avatar: `${BASE}Rick_avatar.jpg`,
  resume: `${BASE}Resume_updated_2026.pdf`,
  card: (id: string) =>
    id === 'venusaur' ? `${BASE}pokemon/venusaur_card.png` : `${BASE}pokemon/${id}_card.jpg`,
};

export type ModelSource = 'cache' | 'browser';
export interface Loaded<T> {
  model: T;
  source: ModelSource;
  path: string;
}

/** Reads a cached model. If the file can't be read, trains the same model in the browser. */
async function load<T>(path: string, fallback: () => T): Promise<Loaded<T>> {
  try {
    const r = await fetch(`${BASE}${path}`);
    if (!r.ok) throw new Error(String(r.status));
    return { model: (await r.json()) as T, source: 'cache', path };
  } catch {
    return { model: fallback(), source: 'browser', path };
  }
}

export const loadCachedModel = () => load<CachedModel>(CACHED_MODEL_PATH, () => buildCachedModel());
export const loadCachedMlp = () => load<CachedMlp>(CACHED_MLP_PATH, () => buildCachedMlp());
