import type { ReactNode } from 'react';
import { assets } from '../../api';
import { pokeName } from '../learn/CalcPanel';

/** Pieces shared by the Pokédex phases (PokedexGame, PokedexGame2). */

export type BallType = 'fire' | 'water' | 'grass';

export const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export function typeName(t: BallType, pt: boolean): string {
  if (t === 'fire') return pt ? 'Fogo' : 'Fire';
  if (t === 'water') return pt ? 'Água' : 'Water';
  return pt ? 'Planta' : 'Grass';
}

/** Same name in both languages, like the game's item names. */
export function ballName(b: BallType): string {
  if (b === 'fire') return 'Fire Ball 🔥';
  if (b === 'water') return 'Water Ball 💧';
  return 'Grass Ball 🌿';
}

interface ChoicesProps<T> {
  options: T[];
  keyOf: (o: T) => string;
  label: (o: T) => ReactNode;
  isRight: (o: T) => boolean;
  wrong: string[];
  onPick: (right: boolean, key: string) => void;
  /** one option per line, for long answers */
  wide?: boolean;
}

export function Choices<T>({ options, keyOf, label, isRight, wrong, onPick, wide }: ChoicesProps<T>) {
  return (
    <div className={`pd-choices ${wide ? 'wide' : ''}`}>
      {options.map((o) => {
        const k = keyOf(o);
        const bad = wrong.includes(k);
        return (
          <button key={k} className={`pd-choice ${bad ? 'wrong' : ''}`} disabled={bad} onClick={() => onPick(isRight(o), k)}>
            {label(o)}
          </button>
        );
      })}
    </div>
  );
}

/** Small Pokémon portrait cropped from the card art; dark until the Pokédex has seen it. */
export function Wild({ id, revealed, className = '' }: { id: string; revealed: boolean; className?: string }) {
  return (
    <div className={`pd-wild ${revealed ? id : 'hidden'} ${className}`}>
      <img src={assets.card(id)} alt={revealed ? pokeName(id) : '???'} draggable={false} />
      {!revealed && <span className="pd-qmark" aria-hidden="true">?</span>}
    </div>
  );
}
