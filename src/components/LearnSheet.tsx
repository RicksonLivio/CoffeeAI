import { lazy, Suspense, useEffect, useState } from 'react';
import { useLang } from '../i18n';
import Lesson1 from './learn/Lesson1';
import Lesson2 from './learn/Lesson2';
import PythonButton, { type CodeFile } from './learn/PythonButton';

// lesson 3 brings KaTeX (formulas): loaded only when the lesson is opened
const Lesson3 = lazy(() => import('./learn/Lesson3'));

interface LessonInfo {
  n: number;
  title: [string, string];
  summary: [string, string];
  Body: React.ComponentType;
  code: CodeFile[];
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
    code: [
      {
        file: 'perceptron_pokemon.py',
        label: ['sem bibliotecas', 'no libraries'],
        about: [
          'Python puro: imprime cada época com net, saída, erro e o ajuste dos pesos, testa o Venusaur e compara η = 1, 0,1 e 0,01.',
          'Plain Python: prints every epoch with net, output, error and weight update, tests Venusaur and compares η = 1, 0.1 and 0.01.',
        ],
        run: 'python3 perceptron_pokemon.py',
      },
      {
        file: 'perceptron_pokemon_torch.py',
        label: ['PyTorch', 'PyTorch'],
        about: [
          'A mesma aula com tensores: versão manual que imprime vetores e produtos passo a passo, e versão rápida, vetorizada.',
          'The same lesson with tensors: a manual version that prints vectors and products step by step, and a quick vectorized one.',
        ],
        run: 'pip install torch && python3 perceptron_pokemon_torch.py',
      },
    ],
  },
  {
    n: 2,
    title: ['O MLP (perceptron multicamadas)', 'The MLP (multilayer perceptron)'],
    summary: [
      'Por que um perceptron não consegue chamar o Venusaur de Planta, como uma camada oculta resolve, e como o backpropagation treina a rede.',
      'Why a perceptron cannot call Venusaur a Grass type, how a hidden layer solves it, and how backpropagation trains the network.',
    ],
    Body: Lesson2,
    code: [
      {
        file: 'mlp_pokemon.py',
        label: ['sem bibliotecas', 'no libraries'],
        about: [
          'Python puro com as cinco partes da aula: o perceptron que não isola Planta, a rede feita à mão, o MLP com backpropagation conta por conta, o tamanho do passo e duas camadas ocultas. Com --plot e o matplotlib, desenha os gráficos.',
          'Plain Python with the five parts of the lesson: the perceptron that cannot isolate Grass, the hand-made network, the MLP with backpropagation step by step, step size and two hidden layers. With --plot and matplotlib, it draws the charts.',
        ],
        run: 'python3 mlp_pokemon.py',
      },
      {
        file: 'mlp_pokemon_torch.py',
        label: ['PyTorch', 'PyTorch'],
        about: [
          'Versão manual com matrizes (imprime X, W, Z, A, δ e gradientes da primeira época e confere contra o autograd) e versão rápida com nn.Sequential, CrossEntropyLoss e SGD, para 1 e 2 camadas ocultas, com acurácia, precisão, recall e F1.',
          'Manual version with matrices (prints X, W, Z, A, δ and gradients of the first epoch and checks them against autograd) and a quick version with nn.Sequential, CrossEntropyLoss and SGD, for 1 and 2 hidden layers, with accuracy, precision, recall and F1.',
        ],
        run: 'pip install torch && python3 mlp_pokemon_torch.py',
      },
    ],
  },
  {
    n: 3,
    title: ['Caderno: como aprendemos os pesos', 'Notebook: how we learn the weights'],
    summary: [
      'Gradiente descendente, backpropagation passo a passo com derivadas parciais e regra da cadeia, e a paisagem de otimização: convexidade, mínimos globais e pontos de sela (Baldi e Hornik, 1989; Kawaguchi, 2016).',
      'Gradient descent, backpropagation step by step with partial derivatives and the chain rule, and the optimization landscape: convexity, global minima and saddle points (Baldi and Hornik, 1989; Kawaguchi, 2016).',
    ],
    Body: Lesson3,
    code: [
      {
        file: 'backprop_notebook.py',
        label: ['sem bibliotecas', 'no libraries'],
        about: [
          'Python puro: rede linear de 3 camadas com matrizes, backpropagation pelas equações (4) a (8), conferido contra diferenças finitas e com a contagem de avaliações; o exemplo de 2 pesos, a sela em (0, 0), a prova de não convexidade e a sela ruim com 3 pesos.',
          'Plain Python: a 3-layer linear network with matrices, backpropagation from equations (4) to (8), checked against finite differences with the evaluation count; the 2-weight example, the saddle at (0, 0), the non-convexity proof and the bad saddle with 3 weights.',
        ],
        run: 'python3 backprop_notebook.py',
      },
      {
        file: 'backprop_notebook_torch.py',
        label: ['PyTorch', 'PyTorch'],
        about: [
          'A fórmula fechada (8) com tensores, comparada ao autograd para L = 2, 3 e 5 camadas, a Hessiana exata nas selas com torch.autograd.functional.hessian e o treino da rede linear profunda.',
          'The closed form (8) with tensors, compared with autograd for L = 2, 3 and 5 layers, the exact Hessian at the saddles with torch.autograd.functional.hessian and training of the deep linear network.',
        ],
        run: 'pip install torch && python3 backprop_notebook_torch.py',
      },
    ],
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
                <Suspense fallback={<div className="panel">{pt ? 'Carregando a aula…' : 'Loading the lesson…'}</div>}>
                  <l.Body />
                </Suspense>
                <PythonButton files={l.code} />
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
