import { useEffect, useState } from 'react';
import { useLang } from '../i18n';
import Lesson1 from './learn/Lesson1';
import Lesson2 from './learn/Lesson2';
import PythonButton from './learn/PythonButton';

interface LessonInfo {
  n: number;
  title: [string, string];
  summary: [string, string];
  Body: () => JSX.Element;
  code: { file: string; about: [string, string] };
}

const LESSONS: LessonInfo[] = [
  {
    n: 1,
    title: ['O Perceptron', 'The Perceptron'],
    summary: [
      'Um neurônio aprende a separar Charizard (Fogo) de Blastoise (Água) usando Velocidade e Defesa. Você acompanha cada conta, do primeiro erro até a convergência.',
      'One neuron learns to tell Charizard (Fire) from Blastoise (Water) using Speed and Defense. You follow every calculation, from the first error to convergence.',
    ],
    Body: Lesson1,
    code: {
      file: 'perceptron_pokemon.py',
      about: [
        'O perceptron da aula em Python puro: imprime cada época com net, saída, erro e o ajuste dos pesos, testa o Venusaur e compara η = 1, 0,1 e 0,01.',
        'The lesson’s perceptron in plain Python: prints every epoch with net, output, error and weight update, tests Venusaur and compares η = 1, 0.1 and 0.01.',
      ],
    },
  },
  {
    n: 2,
    title: ['O MLP (perceptron multicamadas)', 'The MLP (multilayer perceptron)'],
    summary: [
      'Por que um perceptron não consegue chamar o Venusaur de Planta, como uma camada oculta resolve, e como o backpropagation treina a rede.',
      'Why a perceptron cannot call Venusaur a Grass type, how a hidden layer solves it, and how backpropagation trains the network.',
    ],
    Body: Lesson2,
    code: {
      file: 'mlp_pokemon.py',
      about: [
        'As cinco partes da aula em Python puro: o perceptron que não isola Planta, a rede feita à mão, o MLP com backpropagation (a primeira época conta por conta), o tamanho do passo e a rede com duas camadas ocultas. Com --plot e o matplotlib instalado, desenha as curvas e a superfície 3D.',
        'The five parts of the lesson in plain Python: the perceptron that cannot isolate Grass, the hand-made network, the MLP with backpropagation (the first epoch step by step), step size and the two-hidden-layer network. With --plot and matplotlib installed, it draws the curves and the 3D surface.',
      ],
    },
  },
];

/** Lessons start closed; `open` comes from the URL (#learn/1, #learn/2). Closed lessons are not rendered. */
export default function LearnSheet({ open, onToggle }: { open: number[]; onToggle: (n: number) => void }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const i = pt ? 0 : 1;
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    if (!mounted) {
      setMounted(true);
      if (open.length) document.getElementById(`lesson-${open[0]}`)?.scrollIntoView();
    }
  }, [mounted, open]);

  return (
    <div className="sheet">
      <section className="section" style={{ marginTop: 0 }}>
        <span className="kicker">&gt; {pt ? 'APRENDA COMIGO' : 'LEARN WITH ME'}</span>
        <h1 className="wordmark" style={{ fontSize: 'clamp(1.8rem, 5vw, 3.4rem)' }}>{pt ? 'Aprenda comigo' : 'Learn with me'}</h1>
        <p className="lead">
          {pt
            ? 'Aulas curtas sobre como as redes neurais aprendem, com cada conta à mostra. Abra uma aula para começar; feche quando quiser.'
            : 'Short lessons on how neural networks learn, with every calculation in view. Open a lesson to start; close it whenever you like.'}
        </p>
      </section>

      {LESSONS.map((l) => {
        const isOpen = open.includes(l.n);
        return (
          <section key={l.n} className="lesson" id={`lesson-${l.n}`} aria-labelledby={`lesson-${l.n}-title`}>
            <div className="panel lesson-card">
              <span className="lesson-num hex" aria-hidden="true">{String(l.n).padStart(2, '0')}</span>
              <div>
                <span className="label">{pt ? 'Aula' : 'Lesson'} {l.n}</span>
                <h2 id={`lesson-${l.n}-title`}>{l.title[i]}</h2>
                <p>{l.summary[i]}</p>
              </div>
              <button className={`btn ${isOpen ? 'btn--ghost' : 'btn--gold'}`} aria-expanded={isOpen} aria-controls={`lesson-${l.n}-body`}
                onClick={() => onToggle(l.n)}>
                {isOpen ? (pt ? '▴ Recolher' : '▴ Collapse') : (pt ? '▾ Abrir aula' : '▾ Open lesson')}
              </button>
            </div>
            {isOpen && (
              <div id={`lesson-${l.n}-body`}>
                <l.Body />
                <PythonButton file={l.code.file} about={l.code.about[i]} />
                <div className="hero-actions" style={{ justifyContent: 'flex-end' }}>
                  <button className="btn btn--ghost btn--sm" onClick={() => { onToggle(l.n); document.getElementById(`lesson-${l.n}`)?.scrollIntoView(); }}>
                    ▴ {pt ? `Recolher a aula ${l.n}` : `Collapse lesson ${l.n}`}
                  </button>
                </div>
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}
