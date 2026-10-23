/**
 * Logic of the "adding smooth steps" simulation (lesson 2, part 3), kept apart from the React component
 * so the tests can check every preset.
 */
import { sigmoid, STARTERS, type PokeType } from './mlp.ts';

export interface CurveParams { k: number; c1: number; c2: number; v1: number; v2: number }

export const INITIAL: CurveParams = { k: 0.4, c1: -11, c2: 11, v1: 1, v2: -1 };

/**
 * A mini network with one input x = Speed − Defense, two sigmoid hidden neurons and one output that adds them:
 *   y(x) = v1·σ(k·(x − c1)) + v2·σ(k·(x − c2))
 * Each hidden neuron is σ(w·x − θ) with w = k and θ = k·c, so c = θ/w is where its step happens.
 */
export function curveY(p: CurveParams, x: number): number {
  return p.v1 * sigmoid(p.k * (x - p.c1)) + p.v2 * sigmoid(p.k * (x - p.c2));
}

/** Which type the curve puts clearly above the other two (gap ≥ 0.3), if any. */
export function isolated(p: CurveParams): { type: PokeType | null; gap: number; byType: Record<PokeType, [number, number]> } {
  const byType = { fire: [Infinity, -Infinity], water: [Infinity, -Infinity], grass: [Infinity, -Infinity] } as Record<PokeType, [number, number]>;
  for (const s of STARTERS) {
    const y = curveY(p, s.x1 - s.x2);
    byType[s.type][0] = Math.min(byType[s.type][0], y);
    byType[s.type][1] = Math.max(byType[s.type][1], y);
  }
  let best: { type: PokeType | null; gap: number } = { type: null, gap: -Infinity };
  for (const t of ['fire', 'water', 'grass'] as PokeType[]) {
    const others = (['fire', 'water', 'grass'] as PokeType[]).filter((o) => o !== t).map((o) => byType[o][1]);
    const gap = byType[t][0] - Math.max(...others);
    if (gap > best.gap) best = { type: t, gap };
  }
  return { type: best.gap >= 0.3 ? best.type : null, gap: best.gap, byType };
}

export interface Preset { id: string; p: CurveParams; name: [string, string]; look: [string, string] }

export const PRESETS: Preset[] = [
  {
    id: 'start', p: INITIAL, name: ['Detector de Planta', 'Grass detector'],
    look: ['Um degrau sobe em −11, o outro desce em +11. A soma forma um morro: alto no meio (Planta) e baixo nas pontas (Fogo e Água).',
      'One step rises at −11, the other falls at +11. The sum makes a bump: high in the middle (Grass) and low at the ends (Fire and Water).'],
  },
  {
    id: 'one', p: { ...INITIAL, v2: 0 }, name: ['Um degrau só', 'A single step'],
    look: ['Com v2 = 0 sobra um neurônio. A curva só sobe: ela separa "x pequeno" de "x grande", mas Planta e Fogo ficam juntos no alto. Um degrau sozinho nunca isola o meio, e é por isso que um perceptron não consegue.',
      'With v2 = 0 only one neuron is left. The curve only rises: it separates "small x" from "large x", but Grass and Fire end up together at the top. A single step never isolates the middle, which is why a perceptron cannot.'],
  },
  {
    id: 'hard', p: { ...INITIAL, k: 2 }, name: ['Degrau duro', 'Hard step'],
    look: ['Com k grande, cada sigmoide vira quase um degrau, e o morro vira uma caixa: é a rede feita à mão da parte 2, que usava degraus de verdade.',
      'With a large k, each sigmoid becomes almost a step, and the bump becomes a box: that is the hand-made network of part 2, which used real steps.'],
  },
  {
    id: 'soft', p: { ...INITIAL, k: 0.08 }, name: ['Degrau mole', 'Soft step'],
    look: ['Com k pequeno as rampas ficam tão longas que o morro achata: Planta quase não se destaca. Pesos pequenos demais deixam a rede "indecisa".',
      'With a small k the ramps get so long that the bump flattens: Grass barely stands out. Weights that are too small leave the network "undecided".'],
  },
  {
    id: 'fire', p: { k: 0.4, c1: 11, c2: 33, v1: 1, v2: -1 }, name: ['Morro em outro lugar', 'Bump somewhere else'],
    look: ['Movendo c1 e c2 para 11 e 33, o morro passa a cobrir x = +22: agora a soma é um detector de Fogo. Os c dizem onde fica a faixa; os v dizem se ela sobe ou desce.',
      'Moving c1 and c2 to 11 and 33, the bump now covers x = +22: the sum is now a Fire detector. The c values say where the band is; the v values say whether it goes up or down.'],
  },
];

