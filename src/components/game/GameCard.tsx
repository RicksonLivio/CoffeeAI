import { lazy, Suspense, useState } from 'react';
import { useLang } from '../../i18n';
import './game.css';

const PokedexGame = lazy(() => import('./PokedexGame'));

/** Entry card on the Learn sheet; the Pokédex itself loads only when it is opened. */
export default function GameCard({ onLesson }: { onLesson: (n: number) => void }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const [open, setOpen] = useState(false);
  return (
    <section className="lesson pd-branch" id="game" aria-labelledby="game-title">
      <div className="panel lesson-card pd-card">
        <span className="pd-icon" aria-hidden="true" />
        <div>
          <span className="label">{pt ? 'Minijogo · fase 1' : 'Mini-game · phase 1'}</span>
          <h2 id="game-title">{pt ? 'Pokédex: capture com um perceptron' : 'Pokédex: catch them with a perceptron'}</h2>
          <p>
            {pt
              ? 'Você é um treinador na Rota 1. Para capturar cada Pokémon selvagem, o seu Pokédex precisa prever o tipo dele com o perceptron da Aula 1, e você treina o Pokédex a cada fuga.'
              : 'You are a trainer on Route 1. To catch each wild Pokémon, your Pokédex has to predict its type with the perceptron of Lesson 1, and you train the Pokédex every time one breaks free.'}
          </p>
        </div>
        <button className="btn btn--gold" onClick={() => setOpen(true)} aria-haspopup="dialog">
          {pt ? '▶ Abrir o Pokédex' : '▶ Open the Pokédex'}
        </button>
      </div>
      {open && (
        <Suspense fallback={null}>
          <PokedexGame onClose={() => setOpen(false)} onLesson={onLesson} />
        </Suspense>
      )}
    </section>
  );
}
