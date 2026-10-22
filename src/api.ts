// Static "API": the site has no backend, it reads files published with it (GitHub Pages).
// Same idea as web_app/frontend/src/api.ts in the NFH app, with files instead of FastAPI routes.
import { buildCachedModel, CACHED_MODEL_PATH, type CachedModel } from './lib/model';

const BASE = import.meta.env.BASE_URL;

export const assets = {
  photo: `${BASE}Rick.jpeg`,
  avatar: `${BASE}Rick_avatar.jpg`,
  resume: `${BASE}Resume_updated_2026.pdf`,
  card: (id: string) =>
    id === 'venusaur' ? `${BASE}pokemon/venusaur_card.png` : `${BASE}pokemon/${id}_card.jpg`,
};

export type ModelSource = 'cache' | 'browser';

/** Loads the cached perceptron. If the file can't be read, trains the same model in the browser. */
export async function loadCachedModel(): Promise<{ model: CachedModel; source: ModelSource }> {
  try {
    const r = await fetch(`${BASE}${CACHED_MODEL_PATH}`);
    if (!r.ok) throw new Error(String(r.status));
    return { model: (await r.json()) as CachedModel, source: 'cache' };
  } catch {
    return { model: buildCachedModel(), source: 'browser' };
  }
}

export { CACHED_MODEL_PATH };
