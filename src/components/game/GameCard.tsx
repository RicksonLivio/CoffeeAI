import { lazy, Suspense, useState } from 'react';
import { useLang } from '../../i18n';
import './game.css';

const PokedexGame = lazy(() => import('./PokedexGame'));
const PokedexGame2 = lazy(() => import('./PokedexGame2'));

const TEXT = {
  1: {
    label: ['Minijogo · fase 1', 'Mini-game · phase 1'],
    title: ['Pokédex: capture com um perceptron', 'Pokédex: catch them with a perceptron'],
    about: [
      'Você é um treinador na Rota 1. Para capturar cada Pokémon selvagem, o seu Pokédex precisa prever o tipo dele com o perceptron da Aula 1, e você treina o Pokédex a cada fuga.',
      'You are a trainer on Route 1. To catch each wild Pokémon, your Pokédex has to predict its type with the perceptron of Lesson 1, and you train the Pokédex every time one breaks free.',
    ],
  },
  2: {
    label: ['Minijogo · fase 2', 'Mini-game · phase 2'],
    title: ['Pokédex: capture o Venusaur com um MLP', 'Pokédex: catch Venusaur with an MLP'],
    about: [
      'Na Rota 2, o Pokédex ganha uma camada oculta e a Grass Ball. Primeiro você captura com a rede feita à mão da Aula 2; depois, quando a rede que aprende errar, você faz um passo de backpropagation.',
      'On Route 2, the Pokédex gets a hidden layer and the Grass Ball. First you catch with the hand-made network of Lesson 2; then, when the learning network gets it wrong, you do one backpropagation step.',
    ],
  },
} as const;

/** Entry card on the Learn sheet; the Pokédex itself loads only when it is opened. */
export default function GameCard({ phase = 1, onLesson }: { phase?: 1 | 2; onLesson: (n: number) => void }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const i = pt ? 0 : 1;
  const [open, setOpen] = useState(false);
  const t = TEXT[phase];
  const id = phase === 1 ? 'game' : 'game-2';
  const Game = phase === 1 ? PokedexGame : PokedexGame2;
  return (
    <section className="lesson pd-branch" id={id} aria-labelledby={`${id}-title`}>
      <div className="panel lesson-card pd-card">
        <span className="pd-icon" aria-hidden="true" />
        <div>
          <span className="label">{t.label[i]}</span>
          <h2 id={`${id}-title`}>{t.title[i]}</h2>
          <p>{t.about[i]}</p>
        </div>
        <button className="btn btn--gold" onClick={() => setOpen(true)} aria-haspopup="dialog">
          {pt ? '▶ Abrir o Pokédex' : '▶ Open the Pokédex'}
        </button>
      </div>
      {open && (
        <Suspense fallback={null}>
          <Game onClose={() => setOpen(false)} onLesson={onLesson} />
        </Suspense>
      )}
    </section>
  );
}
