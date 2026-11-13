"""Aula 3 · CNN com PyTorch: a mesma rede de cartas do site, com nn.Conv2d, nn.MaxPool2d e nn.Linear.

Partes:
  1. carrega os pesos treinados do site na rede do PyTorch e confere as probabilidades das 93 cartas
  2. confere o backpropagation da convolução feito à mão (aula) contra o autograd
  3. treina a mesma arquitetura do zero, com aumento de dados, e mede treino e teste

Lê cards32.json e model.json: por padrão baixa do site; com --dir PASTA usa arquivos locais.

    pip install torch
    python3 cnn_pokemon_torch.py
    python3 cnn_pokemon_torch.py --dir ../models/cnn_pokemon
"""
import base64
import json
import os
import sys
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

SITE = 'https://ricksonlivio.github.io/CoffeeAI/models/cnn_pokemon/'
CLASSES = ['fire', 'water', 'grass']
TIPOS = ['Fogo', 'Água', 'Planta']
torch.set_default_dtype(torch.float64)


def ler(nome, pasta):
    if pasta:
        with open(os.path.join(pasta, nome), encoding='utf8') as f:
            return json.load(f)
    with urllib.request.urlopen(SITE + nome, timeout=60) as r:
        return json.load(r)


pasta = sys.argv[sys.argv.index('--dir') + 1] if '--dir' in sys.argv else None
cartas = ler('cards32.json', pasta)
modelo = ler('model.json', pasta)


def para_tensor(px):
    b = torch.tensor(list(base64.b64decode(px)), dtype=torch.float64)
    return (b.view(32, 32, 3).permute(2, 0, 1) / 255 - 0.5)  # (3, 32, 32), canais primeiro


X = torch.stack([para_tensor(c['px']) for c in cartas])
Y = torch.tensor([CLASSES.index(c['type']) for c in cartas])
treino = torch.tensor([c['split'] == 'train' for c in cartas])
teste = ~treino


class RedeDeCartas(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # 32 → 28
        self.conv2 = nn.Conv2d(6, 8, 3)   # 14 → 12
        self.fc = nn.Linear(288, 3)       # 6·6·8 → 3

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28 → 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 12 → 6
        return self.fc(x.flatten(1))                 # logits; a softmax fica na perda


# --------------------------------------------------------------------------- #
# 1. Os pesos do site no PyTorch
# --------------------------------------------------------------------------- #
print('=' * 70)
print('1 · Os pesos do site dentro de nn.Conv2d / nn.Linear')
print('=' * 70)
rede = RedeDeCartas()
n = modelo['net']
with torch.no_grad():
    # o site guarda os filtros como [filtro][canal][linha][coluna], a mesma ordem do PyTorch
    rede.conv1.weight.copy_(torch.tensor(n['conv1']['w']).view(6, 3, 5, 5)); rede.conv1.bias.copy_(torch.tensor(n['conv1']['b']))
    rede.conv2.weight.copy_(torch.tensor(n['conv2']['w']).view(8, 6, 3, 3)); rede.conv2.bias.copy_(torch.tensor(n['conv2']['b']))
    rede.fc.weight.copy_(torch.tensor(n['fc']['w']).view(3, 288)); rede.fc.bias.copy_(torch.tensor(n['fc']['b']))
print(f'  pesos: {sum(p.numel() for p in rede.parameters())}')
for nome, p in rede.named_parameters():
    print(f'    {nome:<14} {tuple(p.shape)}')
with torch.no_grad():
    P = F.softmax(rede(X), dim=1)
site = torch.tensor([s['p'] for s in modelo['samples']])
print(f'  maior diferença para as probabilidades do site (93 cartas): {(P - site).abs().max().item():.1e}')
acerto = (P.argmax(1) == Y)
print(f'  acertos: treino {acerto[treino].sum().item()}/{treino.sum().item()}, teste {acerto[teste].sum().item()}/{teste.sum().item()}')

# --------------------------------------------------------------------------- #
# 2. O backpropagation da convolução, à mão vs autograd
# --------------------------------------------------------------------------- #
print()
print('=' * 70)
print('2 · ∂L/∂K e ∂L/∂x da aula contra o autograd')
print('=' * 70)
x = torch.tensor([[1., 2, 0], [0, 1, 3], [2, 1, 1]], requires_grad=True)
K = torch.tensor([[1., -1], [0, 2]], requires_grad=True)
d = torch.tensor([[0.5, -1], [0, 0.25]])
y = F.conv2d(x.view(1, 1, 3, 3), K.view(1, 1, 2, 2)).view(2, 2)  # o "conv2d" do PyTorch é a correlação da aula
(y * d).sum().backward()                                           # L = Σ δ·y, então ∂L/∂y = δ
print(f'  y = {y.tolist()}')
print(f'  ∂L/∂K (autograd) = {K.grad.tolist()}   aula: [[-1.25, 1.75], [-0.75, -2.25]]')
print(f'  ∂L/∂x (autograd) = {x.grad.tolist()}')
print(f'  ∂L/∂K = correlação de x com δ: {F.conv2d(x.detach().view(1, 1, 3, 3), d.view(1, 1, 2, 2)).view(2, 2).tolist()}')

# --------------------------------------------------------------------------- #
# 3. Treino do zero
# --------------------------------------------------------------------------- #
print()
print('=' * 70)
print('3 · Treino do zero (SGD com momento, aumento de dados)')
print('=' * 70)
torch.manual_seed(0)
nova = RedeDeCartas()
opt = torch.optim.SGD(nova.parameters(), lr=0.002, momentum=0.9)
perda = nn.CrossEntropyLoss()
Xtr, Ytr = X[treino], Y[treino]


def aumenta(lote):
    # desloca até 2 pixels (aqui girando a borda; o site repete a borda), espelha metade, muda brilho e contraste
    dx, dy = torch.randint(-2, 3, (2,)).tolist()
    lote = torch.roll(lote, shifts=(dy, dx), dims=(2, 3))
    if torch.rand(1).item() < 0.5:
        lote = lote.flip(3)
    return lote * (0.8 + 0.4 * torch.rand(1)) + (torch.rand(1) - 0.5) * 0.16


for epoca in range(1, 41):
    total = 0.0
    for i in torch.randperm(len(Xtr)).tolist():  # uma carta por vez, como no site
        opt.zero_grad()
        l = perda(nova(aumenta(Xtr[i:i + 1])), Ytr[i:i + 1])
        l.backward()
        opt.step()
        total += l.item()
    if epoca in (1, 5, 10, 20, 30, 40):
        with torch.no_grad():
            pred = nova(X).argmax(1)
        print(f'  época {epoca:>2}: erro médio {total / len(Xtr):.4f} · treino {(pred[treino] == Y[treino]).sum().item()}/{treino.sum().item()}'
              f' · teste {(pred[teste] == Y[teste]).sum().item()}/{teste.sum().item()}')

with torch.no_grad():
    pred = nova(X).argmax(1)
print('  cartas de teste que a rede nova errou:')
for c, p, ok in zip(cartas, pred.tolist(), (pred == Y).tolist()):
    if c['split'] == 'test' and not ok:
        print(f'    {c["name"]}: real {TIPOS[CLASSES.index(c["type"])]}, previsto {TIPOS[p]}')
print('  (o resultado exato muda com a semente e a versão do PyTorch; o do site é fixo, semente 11)')
