# CoffeeAI

Web portfolio of **Rickson Livio de Souza Gaspar**, Staff AI & MLOps Engineer, with a **Learn with me**
section that teaches how neural networks learn, starting with the perceptron.

The look is a retro "Cyber-Net" HUD (navy, steel gray, cyan and gold), inspired by Mega Man Battle
Network and Pokémon cards. Design tokens are in [`src/styles/tokens.css`](src/styles/tokens.css) and the
reference screen in [`docs/design/megaman_theme.png`](docs/design/megaman_theme.png).

## Sections

| Tab | What it has |
|---|---|
| **01 Portfolio** | hero with photo, story as a quest log, work history timeline, skill gauges, certificates, master's research (Neural Fourier Head), education and contact |
| **02 Learn with me** | collapsible lessons (closed until you open them; `#learn/1`, `#learn/2` open them from a link). Lesson 1: the perceptron. Lesson 2: the MLP |
| **03 Resume** | the PDF in the page, with download |

The resume (`public/Resume_updated_2026.pdf`) can also be downloaded from the top bar on every tab.

### Lesson 1: the perceptron

1. **Start:** the problem, the shape of the neuron (x1 = Speed, x2 = Defense, x0 = −1 for the bias), the
   step function and the learning rule `e = d − y`, `Δw = η·e·x`.
2. **How it works and trains:** an animated graph of the neuron next to the same step written as
   arithmetic. Gold balls carry the data forward and red balls carry the error back to the weights, the
   same convention as the training view of the NFH web app. You can play, pause, change speed or go
   phase by phase: inputs → sum → activation → error → update. A decision-boundary chart and a summary
   table follow each step.
3. **Try it yourself:** move w1, w2 and θ, classify Charizard, Blastoise or Venusaur (not in the training
   data), and retrain with another learning rate or sample order.

With η = 1 and initial weights (0, 0, 0.5) the lesson's numbers are:

| Epoch | Pokémon | net | y | e | weights after (w1, w2, θ) |
|---|---|---|---|---|---|
| 1 | Charizard (100, 78), d = 1 | −0.5 | 0 | 1 | (100, 78, −0.5) |
| 1 | Blastoise (78, 100), d = 0 | 15600.5 | 1 | −1 | (22, −22, 0.5) |
| 2 | Charizard | 483.5 | 1 | 0 | (22, −22, 0.5) |
| 2 | Blastoise | −484.5 | 0 | 0 | (22, −22, 0.5) |

`tests/perceptron.test.ts` checks every one of them.

### Lesson 2: the MLP

Lesson 1 calls Venusaur (Speed 80, Defense 83) a Water type. Lesson 2 uses the nine starters (Charmander,
Charmeleon, Charizard, Squirtle, Wartortle, Blastoise, Bulbasaur, Ivysaur, Venusaur) to show why:

1. **The problem:** Speed − Defense is +22 for every Fire starter, −22 for every Water starter and −4 or −3
   for the Grass ones. Ivysaur and Venusaur lie inside the area enclosed by Charmander, Charizard,
   Blastoise and Squirtle, so no single line separates Grass from the rest. A "Grass or not" perceptron
   trained for 100 epochs never reaches zero errors.
2. **The hidden layer, by hand:** h1 fires when Speed − Defense ≥ 11 and h2 when Defense − Speed ≥ 11; the
   output layer says Fire = h1, Water = h2, Grass = neither. Every calculation is shown for each Pokémon,
   in the original plane and in the (h1, h2) plane.
3. **How it learns:** sigmoid, softmax and cross-entropy; one backpropagation step written out for
   Venusaur; then the full training run (2-2-3 network, η = 2, 1500 epochs, full batch), cached in
   `public/models/mlp_pokemon/model.json`, with the loss curve, the regions per epoch and the weights the
   hidden layer learned.

`tests/mlp.test.ts` checks the claims: the hull argument, the non-convergence of the single perceptron,
the hand-made network on all nine Pokémon, the backpropagation gradient against a numerical gradient, and
the trained network getting all nine right.

## How it is built

Same front-end stack and conventions as the web app of my master's dissertation (Neural Fourier Head):
React 18 + TypeScript + Vite 6, charts with Recharts, pinned versions, `npm ci`, `ignore-scripts=true`.

There is **no backend**. Like the NFH app, which loads trained weights from versioned files instead of
retraining, the lesson reads a **cached model**:

```
src/lib/perceptron.ts                        the perceptron, written to be read in class
src/lib/mlp.ts                               the MLP: hand-made step network, sigmoid/softmax, backpropagation
src/lib/model.ts                             builds the cached model (config, samples, full trace, final weights)
scripts/export-model.ts                      writes public/models/{perceptron_pokemon,mlp_pokemon}/model.json
tests/cached-model.test.ts                   fails if a JSON no longer matches the code
```

If the JSON can't be loaded, the page trains the same model in the browser and says so. The "Try it
yourself" section always trains live, since it takes microseconds.

| Path | Contents |
|---|---|
| `src/App.tsx` | top bar, tabs with hash routing (`#portfolio`, `#learn/1,2`, `#resume`), EN/PT switch |
| `src/content/profile.ts` | all portfolio text, in English and Portuguese, taken from the resume |
| `src/components/PortfolioSheet.tsx`, `ResumeSheet.tsx` | portfolio and resume tabs |
| `src/components/LearnSheet.tsx` | lesson index with collapsible lessons |
| `src/components/learn/*` | Lesson1, Lesson2, animated perceptron and MLP graphs, calculations, charts, lab, notes |
| `src/styles/tokens.css`, `app.css` | Cyber-Net tokens (unchanged) and the site layout |
| `public/` | photo, resume PDF, Pokémon cards, cached model, favicon |

## Run locally

Requires Node ≥ 22.6 (the tests use Node's built-in TypeScript type stripping).

```bash
npm ci
npm run dev            # http://127.0.0.1:5174 (hot reload)
npm test               # perceptron, MLP and cached-model tests
npm run build          # typecheck + build to dist/
npm run preview        # serves dist/ on http://127.0.0.1:4174
npm run export-model   # regenerate the cached model after changing src/lib/perceptron.ts
```

On WSL, use a Linux Node (for example `export PATH="$HOME/.local/node/bin:$PATH"`); the Windows Node
can't build on the Linux filesystem.

## Publish on GitHub Pages

`.github/workflows/pages.yml` tests, builds and deploys on every push to `main`.

1. In the repository: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push to `main`. The site goes to https://ricksonlivio.github.io/CoffeeAI/; the build uses relative
   paths (`base: './'`), so it also works under another repository name.

GitHub Pages is free for public repositories.

## Credits

Pokémon card images and names belong to Nintendo, Creatures and GAME FREAK; Mega Man Battle Network
belongs to Capcom. They are used here for teaching, with no commercial purpose.
