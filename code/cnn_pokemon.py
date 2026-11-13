"""Aula 3 · CNN: convolução, pooling e uma Pokédex que lê cartas. Python puro, sem bibliotecas.

Partes:
  1. a convolução em blocos (borda vertical no recorte 6×6), com padding e stride
  2. max, average e min pooling no mapa 4×4, e o erro voltando pelo max pooling
  3. backpropagation em uma convolução, à mão (entrada 3×3, filtro 2×2)
  4. a Pokédex: a rede treinada do site (1763 pesos) classifica as 24 cartas de teste

A parte 4 lê cards32.json e model.json. Por padrão baixa os dois do site;
com --dir PASTA usa arquivos locais (por exemplo public/models/cnn_pokemon do repositório).

    python3 cnn_pokemon.py
    python3 cnn_pokemon.py --dir ../models/cnn_pokemon
"""
import base64
import json
import math
import os
import sys
import urllib.request

SITE = 'https://ricksonlivio.github.io/CoffeeAI/models/cnn_pokemon/'
TIPOS = ['Fogo', 'Água', 'Planta']
CLASSES = ['fire', 'water', 'grass']


def fmt(v):
    v = round(v, 4)
    return str(int(v)) if v == int(v) else str(v)


def mostra(nome, grade):
    print(f'  {nome}')
    for linha in grade:
        print('    ' + ' '.join(f'{fmt(v):>6}' for v in linha))


# --------------------------------------------------------------------------- #
# 1. Convolução em uma grade
# --------------------------------------------------------------------------- #
def tamanho_saida(n, k, p, s):
    return (n + 2 * p - k) // s + 1


def conv2d(grade, filtro, pad=0, stride=1):
    h, w, k = len(grade), len(grade[0]), len(filtro)
    oh, ow = tamanho_saida(h, k, pad, stride), tamanho_saida(w, k, pad, stride)
    saida = []
    for oy in range(oh):
        linha = []
        for ox in range(ow):
            soma = 0
            for ky in range(k):
                for kx in range(k):
                    y, x = oy * stride + ky - pad, ox * stride + kx - pad
                    if 0 <= y < h and 0 <= x < w:
                        soma += grade[y][x] * filtro[ky][kx]
            linha.append(soma)
        saida.append(linha)
    return saida


RECORTE = [
    [9, 9, 9, 1, 1, 1],
    [9, 9, 9, 1, 1, 1],
    [9, 9, 8, 2, 1, 1],
    [9, 8, 8, 2, 1, 0],
    [8, 8, 7, 2, 1, 0],
    [8, 7, 7, 1, 0, 0],
]
BORDA_VERTICAL = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]

print('=' * 70)
print('1 · Convolução: o filtro que desliza')
print('=' * 70)
mostra('recorte 6×6 (claro à esquerda, escuro à direita)', RECORTE)
mostra('filtro 3×3 de borda vertical', BORDA_VERTICAL)
y = conv2d(RECORTE, BORDA_VERTICAL)
termos = [f'{RECORTE[ky][kx]}·{BORDA_VERTICAL[ky][kx]}' for ky in range(3) for kx in range(3)]
print(f'  posição (0, 0): net = {" + ".join(termos)} = {y[0][0]}')
termos = [f'{RECORTE[ky][kx + 1]}·{BORDA_VERTICAL[ky][kx]}' for ky in range(3) for kx in range(3)]
print(f'  posição (0, 1): net = {" + ".join(termos)} = {y[0][1]}')
mostra('saída 4×4 (as colunas do meio acendem: é a borda)', y)
for p, s in [(0, 1), (1, 1), (0, 2), (1, 2)]:
    print(f'  padding {p}, stride {s}: saída = ⌊(6 + 2·{p} − 3)/{s}⌋ + 1 = {tamanho_saida(6, 3, p, s)}'
          f'  (conferido: {len(conv2d(RECORTE, BORDA_VERTICAL, p, s))})')
mostra('depois da ReLU, max(0, net), com padding 1', [[max(0, v) for v in l] for l in conv2d(RECORTE, BORDA_VERTICAL, 1, 1)])

# --------------------------------------------------------------------------- #
# 2. Pooling
# --------------------------------------------------------------------------- #
MAPA = [[1, 3, 2, 9], [5, 6, 1, 0], [7, 2, 4, 4], [0, 1, 8, 3]]


def pool(grade, tipo, t=2):
    f = {'max': max, 'min': min, 'média': lambda v: sum(v) / len(v)}[tipo]
    return [[f([grade[oy * t + a][ox * t + b] for a in range(t) for b in range(t)])
             for ox in range(len(grade[0]) // t)] for oy in range(len(grade) // t)]


def max_pool_volta(grade, delta, t=2):
    g = [[0] * len(grade[0]) for _ in grade]
    for oy, linha in enumerate(delta):
        for ox, d in enumerate(linha):
            blocos = [(oy * t + a, ox * t + b) for a in range(t) for b in range(t)]
            y, x = max(blocos, key=lambda c: grade[c[0]][c[1]])
            g[y][x] += d
    return g


print()
print('=' * 70)
print('2 · Pooling 2×2, stride 2')
print('=' * 70)
mostra('mapa 4×4', MAPA)
for tipo in ['max', 'média', 'min']:
    mostra(f'{tipo} pooling', pool(MAPA, tipo))
mostra('erro voltando pelo max pooling, δ = (0.4, -0.2; 0.1, 0.3): só o vencedor recebe',
       max_pool_volta(MAPA, [[0.4, -0.2], [0.1, 0.3]]))

# --------------------------------------------------------------------------- #
# 3. Backpropagation em uma convolução
# --------------------------------------------------------------------------- #
X = [[1, 2, 0], [0, 1, 3], [2, 1, 1]]
K = [[1, -1], [0, 2]]
D = [[0.5, -1], [0, 0.25]]
print()
print('=' * 70)
print('3 · Backpropagation em uma convolução, à mão')
print('=' * 70)
mostra('x', X); mostra('K', K); mostra('y = x ∗ K', conv2d(X, K)); mostra('δ (erro que chegou na saída)', D)
gK = [[0, 0], [0, 0]]
for a in range(2):
    for b in range(2):
        termos = [f'{D[i][j]}·{X[i + a][j + b]}' for i in range(2) for j in range(2)]
        gK[a][b] = sum(D[i][j] * X[i + a][j + b] for i in range(2) for j in range(2))
        print(f'  ∂L/∂K[{a}][{b}] = {" + ".join(termos)} = {fmt(gK[a][b])}')
gX = [[0] * 3 for _ in range(3)]
for i in range(2):
    for j in range(2):
        for a in range(2):
            for b in range(2):
                gX[i + a][j + b] += D[i][j] * K[a][b]
mostra('∂L/∂x (o erro que segue para baixo)', gX)
# conferência com diferença finita: L = Σ δ·y
L = lambda KK: sum(D[i][j] * v for i, l in enumerate(conv2d(X, KK)) for j, v in enumerate(l))
pior = 0
for a in range(2):
    for b in range(2):
        kp = [r[:] for r in K]; km = [r[:] for r in K]
        kp[a][b] += 1e-6; km[a][b] -= 1e-6
        pior = max(pior, abs((L(kp) - L(km)) / 2e-6 - gK[a][b]))
print(f'  diferença finita vs fórmula: maior diferença = {pior:.1e}')
print(f'  com η = 0.1: K[1][1] = 2 − 0.1·({fmt(gK[1][1])}) = {fmt(2 - 0.1 * gK[1][1])}')


# --------------------------------------------------------------------------- #
# 4. Pokédex: a rede treinada do site
# --------------------------------------------------------------------------- #
def ler(nome, pasta):
    if pasta:
        with open(os.path.join(pasta, nome), encoding='utf8') as f:
            return json.load(f)
    with urllib.request.urlopen(SITE + nome, timeout=60) as r:
        return json.load(r)


def para_tensor(px):
    b = base64.b64decode(px)  # RGB intercalado, 32×32
    return [[[b[(y * 32 + x) * 3 + c] / 255 - 0.5 for x in range(32)] for y in range(32)] for c in range(3)]


def conv_camada(x, cam):
    f, c, k = cam['f'], cam['c'], cam['k']
    w, bias = cam['w'], cam['b']
    h = len(x[0]); o = h - k + 1
    saida = []
    for fi in range(f):
        mapa = [[bias[fi]] * o for _ in range(o)]
        for ci in range(c):
            xc = x[ci]
            for ky in range(k):
                for kx in range(k):
                    peso = w[((fi * c + ci) * k + ky) * k + kx]
                    for oy in range(o):
                        lx, ly = xc[oy + ky], mapa[oy]
                        for ox in range(o):
                            ly[ox] += peso * lx[ox + kx]
        saida.append(mapa)
    return saida


def relu_pool(x):
    return [[[max(0, m[2 * y][2 * xx], m[2 * y][2 * xx + 1], m[2 * y + 1][2 * xx], m[2 * y + 1][2 * xx + 1])
              for xx in range(len(m) // 2)] for y in range(len(m) // 2)] for m in x]


def softmax(z):
    mx = max(z); e = [math.exp(v - mx) for v in z]; s = sum(e)
    return [v / s for v in e]


def pokedex(rede, x):
    p1 = relu_pool(conv_camada(x, rede['conv1']))  # 32 → 28 → 14   (ReLU antes do max dá o mesmo que max e depois ReLU)
    p2 = relu_pool(conv_camada(p1, rede['conv2']))  # 14 → 12 → 6
    plano = [v for m in p2 for l in m for v in l]  # 6·6·8 = 288
    fc = rede['fc']
    z = [fc['b'][j] + sum(plano[i] * fc['w'][j * fc['m'] + i] for i in range(fc['m'])) for j in range(fc['n'])]
    return softmax(z)


pasta = sys.argv[sys.argv.index('--dir') + 1] if '--dir' in sys.argv else None
print()
print('=' * 70)
print('4 · Pokédex: a rede treinada classifica as cartas de teste')
print('=' * 70)
cartas = ler('cards32.json', pasta)
modelo = ler('model.json', pasta)
rede = modelo['net']
n_pesos = sum(len(rede[c]['w']) + len(rede[c]['b']) for c in ('conv1', 'conv2', 'fc'))
print(f'  rede: conv 5×5 ×6 → ReLU → max pool → conv 3×3 ×8 → ReLU → max pool → densa 288 → 3 · {n_pesos} pesos')
acertos, teste, pior = 0, [c for c in cartas if c['split'] == 'test'], 0
esperado = {s['id']: s['p'] for s in modelo['samples']}
for c in teste:
    p = pokedex(rede, para_tensor(c['px']))
    pior = max(pior, max(abs(a - b) for a, b in zip(p, esperado[c['id']])))
    k = p.index(max(p)); ok = CLASSES[k] == c['type']
    acertos += ok
    print(f'  {c["name"]:<22} real {TIPOS[CLASSES.index(c["type"])]:<7} Pokédex {TIPOS[k]:<7} '
          f'p = ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) {"✓" if ok else "✗"}')
print(f'  acertos no teste: {acertos}/{len(teste)}')
print(f'  diferença para as probabilidades do site: {pior:.1e}')
