#!/usr/bin/env python3
"""
CoffeeAI · Aula 2 · MLP em PyTorch
==================================

Mesmo exemplo da Aula 2 (https://ricksonlivio.github.io/CoffeeAI/#learn/2), com tensores.
Precisa do PyTorch:  pip install torch

    python3 mlp_pokemon_torch.py            # as duas versões
    python3 mlp_pokemon_torch.py --manual   # só a versão com matrizes, passo a passo
    python3 mlp_pokemon_torch.py --rapida   # só a versão curta, com nn.Module e otimizador

Versão 1 (manual): para a frente, entropia cruzada, backpropagation e atualização escritos com
    matrizes, imprimindo cada uma na primeira época; no fim confere os gradientes contra o autograd.
Versão 2 (rápida): nn.Sequential + nn.CrossEntropyLoss + torch.optim.SGD, como se faz no dia a dia.
Nas duas: 2 -> 2 -> 3 e também 2 -> 2 -> 2 -> 3, pesos iniciais iguais aos da página, η = 2,
lote completo (os nove Pokémon por época), e no fim as métricas (acurácia, precisão, recall, F1).

Convenção da página: net = W·x + θ·(-1). No nn.Linear do PyTorch é W·x + b, então b = -θ.

A primeira época e o resultado final batem com a página. No meio de treinos longos com η = 2 os
valores podem diferir nas últimas casas entre a página, o Python puro e o PyTorch: cada um arredonda
exp e somas de um jeito, e passos grandes amplificam essas diferenças (veja a parte 4 da aula).

Autor: Rickson Livio de Souza Gaspar · CoffeeAI
"""
import sys

import torch
from torch import nn

torch.set_default_dtype(torch.float64)  # as mesmas contas da página (float64)
torch.set_printoptions(precision=4, sci_mode=False)

TYPES = ["Fogo", "Água", "Planta"]
NAMES = ["Charmander", "Charmeleon", "Charizard", "Squirtle", "Wartortle", "Blastoise",
         "Bulbasaur", "Ivysaur", "Venusaur"]
# [Velocidade, Defesa] divididos por 100
X = torch.tensor([[65, 43], [80, 58], [100, 78], [43, 65], [58, 80], [78, 100],
                  [45, 49], [60, 63], [80, 83]], dtype=torch.float64) / 100
Y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])           # índice do tipo certo
D = torch.nn.functional.one_hot(Y, 3).to(torch.float64)  # "um ligado": Planta = [0, 0, 1]

# Pesos iniciais da página: W[j] = pesos do neurônio j, θ[j] = seu bias (entrada -1)
INIT_1H = {
    "W": [torch.tensor([[0.5, -0.4], [0.1, 0.8]]), torch.tensor([[0.0, -0.3], [-0.8, 0.6], [-0.2, 0.7]])],
    "theta": [torch.tensor([-0.4, 0.3]), torch.tensor([0.6, -0.6, 0.2])],
}
INIT_2H = {
    "W": [torch.tensor([[0.5, -0.4], [0.1, 0.8]]), torch.tensor([[0.0, -0.3], [-0.8, 0.6]]),
          torch.tensor([[-0.2, 0.7], [-0.7, 0.3], [0.9, -0.9]])],
    "theta": [torch.tensor([-0.4, 0.3]), torch.tensor([0.6, -0.6]), torch.tensor([0.2, -0.3, 0.8])],
}


def show(name, t):
    print(f"{name} {tuple(t.shape)} =\n{t}")


# --------------------------------------------------------------------------- #
# Versão 1 · manual
# --------------------------------------------------------------------------- #
def forward(W, theta):
    """A[0] = X; A[l+1] = σ(A[l] W[l]ᵀ − θ[l]) nas ocultas; na última, softmax."""
    A, Z = [X], []
    for l in range(len(W)):
        z = A[-1] @ W[l].T - theta[l]          # (9 x neurônios): todas as somas de uma vez
        Z.append(z)
        if l == len(W) - 1:
            A.append(torch.softmax(z, dim=1))  # cada linha soma 1
        else:
            A.append(torch.sigmoid(z))
    return A, Z


def backward(W, A):
    """Backpropagation com a média dos nove: devolve os gradientes de W e θ, camada por camada."""
    n = X.shape[0]
    P = A[-1]
    delta = (P - D) / n                         # erro da saída: (p − d), já dividido por n (média)
    deltas = [None] * len(W)
    deltas[-1] = delta
    for l in range(len(W) - 2, -1, -1):        # de trás para a frente, com os pesos ANTIGOS
        H = A[l + 1]
        deltas[l] = (deltas[l + 1] @ W[l + 1]) * H * (1 - H)
    gW = [deltas[l].T @ A[l] for l in range(len(W))]        # ∂L/∂W = δᵀ · entrada
    gtheta = [-deltas[l].sum(dim=0) for l in range(len(W))]  # ∂L/∂θ = δ · (−1)
    return gW, gtheta, deltas


def loss_fn(P):
    return -(D * torch.log(P)).sum(dim=1).mean()  # entropia cruzada média: −Σ d·ln p


def manual(init, eta, epochs, label):
    print("\n" + "=" * 70 + f"\nVERSÃO 1 · manual · {label}\n" + "=" * 70)
    W = [w.clone() for w in init["W"]]
    theta = [t.clone() for t in init["theta"]]
    for e in range(epochs + 1):
        A, Z = forward(W, theta)
        L = loss_fn(A[-1])
        acc = (A[-1].argmax(dim=1) == Y).double().mean().item()
        if e == 0:
            print("Época 1, para a frente (as 9 linhas são os 9 Pokémon):")
            show("X", X)
            for l in range(len(W)):
                show(f"W[{l}]", W[l]); show(f"θ[{l}]", theta[l])
                show(f"Z[{l}] = A[{l}]·W[{l}]ᵀ − θ[{l}]", Z[l])
                show(f"A[{l + 1}] = {'softmax' if l == len(W) - 1 else 'σ'}(Z[{l}])", A[l + 1])
            print("soma de cada linha de P:", A[-1].sum(dim=1))
            print(f"L = média de −ln p_certo = {L.item():.4f}")
        gW, gtheta, deltas = backward(W, A)
        if e == 0:
            print("\nPara trás (δ da saída para a entrada; já divididos por 9):")
            for l in range(len(W) - 1, -1, -1):
                show(f"δ[{l}]", deltas[l])
            for l in range(len(W)):
                show(f"∂L/∂W[{l}]", gW[l]); show(f"∂L/∂θ[{l}]", gtheta[l])
            check_autograd(init, gW, gtheta)
        if e % 250 == 0 or e == epochs:
            print(f"  época {e:>4}: L = {L.item():.4f} · acurácia {acc:.3f}")
        if e < epochs:
            # todos os pesos juntos: peso ← peso − η · gradiente
            W = [W[l] - eta * gW[l] for l in range(len(W))]
            theta = [theta[l] - eta * gtheta[l] for l in range(len(W))]
    A, _ = forward(W, theta)
    metrics(A[-1], label)
    return W, theta


def check_autograd(init, gW, gtheta):
    """Os gradientes feitos à mão batem com os do autograd do PyTorch?"""
    W = [w.clone().requires_grad_(True) for w in init["W"]]
    theta = [t.clone().requires_grad_(True) for t in init["theta"]]
    A, _ = forward(W, theta)
    loss_fn(A[-1]).backward()
    diff = max(max((a.grad - b).abs().max().item() for a, b in zip(W, gW)),
               max((a.grad - b).abs().max().item() for a, b in zip(theta, gtheta)))
    print(f"\nAutograd x conta à mão: maior diferença = {diff:.2e} (zero, a menos de arredondamento)")


# --------------------------------------------------------------------------- #
# Métricas: acurácia, precisão, recall (sensibilidade), F1, por tipo
# --------------------------------------------------------------------------- #
def metrics(P, label):
    pred = P.argmax(dim=1)
    print(f"\nMétricas · {label} (nos mesmos 9 Pokémon do treino):")
    conf = torch.zeros(3, 3, dtype=torch.int64)
    for t, p in zip(Y, pred):
        conf[t, p] += 1
    print("Matriz de confusão (linha = real, coluna = previsto):\n", conf)
    for c in range(3):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"  {TYPES[c]:<6} precisão {prec:.3f} · recall/sensibilidade {rec:.3f} · F1 {f1:.3f}")
    print(f"  acurácia {(pred == Y).double().mean().item():.3f}")
    print("  probabilidades do Venusaur:", [f"{TYPES[k]} {100 * P[8, k].item():.1f}%" for k in range(3)])


# --------------------------------------------------------------------------- #
# Versão 2 · rápida (nn.Module + CrossEntropyLoss + SGD)
# --------------------------------------------------------------------------- #
def build(init):
    layers = []
    for l, (w, t) in enumerate(zip(init["W"], init["theta"])):
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(-t)  # b = −θ (a página usa a entrada fixa −1)
        layers.append(lin)
        if l < len(init["W"]) - 1:
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)  # a última camada devolve as somas z; o softmax fica na perda


def rapida(init, eta, epochs, label):
    print("\n" + "=" * 70 + f"\nVERSÃO 2 · rápida · {label}\n" + "=" * 70)
    model = build(init)
    print(model)
    loss = nn.CrossEntropyLoss()           # softmax + entropia cruzada numa conta só (mais estável)
    opt = torch.optim.SGD(model.parameters(), lr=eta)
    for e in range(epochs + 1):
        z = model(X)
        L = loss(z, Y)
        if e % 250 == 0 or e == epochs:
            acc = (z.argmax(dim=1) == Y).double().mean().item()
            print(f"  época {e:>4}: L = {L.item():.4f} · acurácia {acc:.3f}")
        if e < epochs:
            opt.zero_grad()
            L.backward()                   # backpropagation automático
            opt.step()                     # peso ← peso − η · gradiente, todos juntos
    with torch.no_grad():
        metrics(torch.softmax(model(X), dim=1), label)
    return model


if __name__ == "__main__":
    args = sys.argv[1:]
    runs = [(INIT_1H, 2.0, 1500, "1 camada oculta (2 -> 2 -> 3)"),
            (INIT_2H, 2.0, 2000, "2 camadas ocultas (2 -> 2 -> 2 -> 3)")]
    for init, eta, epochs, label in runs:
        if "--rapida" not in args:
            manual(init, eta, epochs, label)
        if "--manual" not in args:
            rapida(init, eta, epochs, label)
