"""Builds the lesson-3 card dataset (Fire / Water / Grass) from Pokémon TCG card pictures.

1. Picks single-type Basic Pokémon cards from the public card list of the Pokémon TCG API
   (github.com/PokemonTCG/pokemon-tcg-data), spread over many sets, and downloads the pictures
   from images.pokemontcg.io into new_files/cards/ (skipped if already there).
2. Adds the three lesson cards (Charizard, Blastoise, Venusaur) from public/pokemon/.
3. Writes public/pokemon/cnn/<id>.jpg (245×342, shown in the page) and
   public/models/cnn_pokemon/cards32.json (each card squeezed to 32×32 RGB, base64), the network input.

Needs Pillow and curl.   Run:  python3 scripts/prepare_cnn_cards.py
Then:  npm run train-cnn
"""
import base64, json, os, random, subprocess, urllib.request
from collections import Counter
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'new_files', 'cards')
OUT = os.path.join(ROOT, 'public', 'pokemon', 'cnn')
DATA = os.path.join(ROOT, 'public', 'models', 'cnn_pokemon', 'cards32.json')
SETS = ['base1', 'base2', 'base3', 'neo1', 'neo2', 'ex1', 'ex5', 'dp1', 'dp3', 'bw1', 'bw4', 'xy1', 'xy5',
        'sm1', 'sm4', 'swsh1', 'swsh4', 'swsh7', 'sv1', 'sv3']
PER_TYPE, TEST_PER_TYPE, SIZE = 30, 8, 32
TYPES = {'Fire': 'fire', 'Water': 'water', 'Grass': 'grass'}
LESSON = [('charizard', 'Charizard', 'fire', 'charizard_card.jpg'), ('blastoise', 'Blastoise', 'water', 'blastoise_card.jpg'),
          ('venusaur', 'Venusaur', 'grass', 'venusaur_card.png')]


def pick():
    random.seed(7)
    pool = {t: [] for t in TYPES}
    for s in SETS:
        url = f'https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master/cards/en/{s}.json'
        for c in json.load(urllib.request.urlopen(url, timeout=30)):
            if c.get('supertype') == 'Pokémon' and len(c.get('types', [])) == 1 and c['types'][0] in pool \
                    and 'Basic' in c.get('subtypes', []):
                pool[c['types'][0]].append((s, c['id'], c['name']))
    out = []
    for t, cards in pool.items():
        by_set = {}
        for x in cards:
            by_set.setdefault(x[0], []).append(x)
        names, sel = set(), []
        for _ in range(6):  # round robin over the sets, one new name per set per round
            for s, l in by_set.items():
                random.shuffle(l)
                for x in l:
                    if x[2] not in names:
                        sel.append(x); names.add(x[2]); break
        out += [{'type': TYPES[t], 'id': x[1], 'name': x[2], 'set': x[0]} for x in sel[:PER_TYPE]]
    return out


def main():
    os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.dirname(DATA), exist_ok=True)
    cards = pick()
    for c in cards:
        fn = os.path.join(RAW, c['id'] + '.png')
        if not os.path.exists(fn) or os.path.getsize(fn) < 5000:
            s, n = c['id'].split('-', 1)
            subprocess.run(['curl', '-sf', '-m', '40', '-o', fn, f'https://images.pokemontcg.io/{s}/{n}.png'], check=True)
    random.seed(3)
    items = [{'id': i, 'name': n, 'type': t, 'src': os.path.join(ROOT, 'public', 'pokemon', f), 'file': f'pokemon/{f}',
              'split': 'train', 'set': 'lesson'} for i, n, t, f in LESSON]
    for t in TYPES.values():
        group = [c for c in cards if c['type'] == t]
        random.shuffle(group)
        for k, c in enumerate(group):
            dst = os.path.join(OUT, c['id'] + '.jpg')
            Image.open(os.path.join(RAW, c['id'] + '.png')).convert('RGB').resize((245, 342), Image.LANCZOS).save(dst, quality=82)
            items.append({**c, 'src': dst, 'file': f"pokemon/cnn/{c['id']}.jpg", 'split': 'test' if k < TEST_PER_TYPE else 'train'})
    data = []
    for it in items:
        im = Image.open(it['src']).convert('RGB').resize((SIZE, SIZE), Image.BILINEAR)
        data.append({k: it[k] for k in ('id', 'name', 'type', 'file', 'split', 'set')} | {'px': base64.b64encode(im.tobytes()).decode()})
    json.dump(data, open(DATA, 'w'))
    print(len(data), Counter((d['type'], d['split']) for d in data))


if __name__ == '__main__':
    main()
