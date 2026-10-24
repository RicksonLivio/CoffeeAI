#!/usr/bin/env python3
"""
CoffeeAI · Aula 1 · Perceptron em PyTorch
=========================================

Mesmo exemplo da Aula 1 (https://ricksonlivio.github.io/CoffeeAI/#learn/1), agora com tensores.
Precisa do PyTorch:  pip install torch

    python3 perceptron_pokemon_torch.py            # as duas versões
    python3 perceptron_pokemon_torch.py --manual   # só a versão com matrizes, passo a passo
    python3 perceptron_pokemon_torch.py --rapida   # só a versão curta

Versão 1 (manual): cada conta da aula escrita com vetores e matrizes, imprimindo tudo.
Versão 2 (rápida): a mesma regra em poucas linhas, para todas as amostras de uma vez.

Observação: a regra do perceptron não usa gradiente (o degrau não tem derivada útil), então aqui
não entram autograd nem otimizadores do PyTorch. Isso muda na Aula 2.

Autor: Rickson Livio de Souza Gaspar · CoffeeAI
"""
import sys

import torch

torch.set_default_dtype(torch.float64)  # mesmas contas exatas da página

NAMES = ["Charizard", "Blastoise"]
# Cada linha: [Velocidade, Defesa, -1]. O -1 é a entrada fixa do bias, então θ vira o 3º peso.
X = torch.tensor([[100.0, 78.0, -1.0],
                  [78.0, 100.0, -1.0]])
D = torch.tensor([1.0, 0.0])  # 1 = Fogo (Charizard), 0 = Água (Blastoise)


def degrau(net):
    return (net >= 0).to(net.dtype)


def manual(eta=1.0):
    print("=" * 66 + "\nVERSÃO 1 · matrizes, passo a passo\n" + "=" * 66)
    w = torch.tensor([0.0, 0.0, 0.5])  # [w1, w2, θ]
    print("X (amostras x [x1, x2, x0]) =\n", X)
    print("d =", D, "\nw inicial = [w1, w2, θ] =", w)
    for epoch in range(1, 21):
        print(f"\n--- Época {epoch} ---")
        errors = 0
        for i in range(len(X)):
            x = X[i]
            produtos = w * x                 # [w1·x1, w2·x2, θ·(-1)]
            net = produtos.sum()             # net = w · x
            y = degrau(net)
            e = D[i] - y
            print(f"{NAMES[i]}: x = {x.tolist()}")
            print(f"  w * x = {produtos.tolist()}  ->  net = {net.item():g}  ->  y = {int(y)}  ->  e = d - y = {int(e)}")
            if e != 0:
                errors += 1
                dw = eta * e * x             # Δw = η · e · x (o mesmo para θ, porque x0 = -1)
                w = w + dw
                print(f"  Δw = η·e·x = {dw.tolist()}  ->  w = {w.tolist()}")
        if errors == 0:
            print(f"\nÉpoca sem erros: convergiu. w final = {w.tolist()}")
            break
    net_v = torch.dot(w, torch.tensor([80.0, 83.0, -1.0]))
    print(f"Venusaur (80, 83): net = {net_v.item():g} -> y = {int(degrau(net_v))} (Água: o perceptron só conhece dois lados)")
    return w


def rapida(eta=1.0):
    print("\n" + "=" * 66 + "\nVERSÃO 2 · rápida\n" + "=" * 66)
    w = torch.tensor([0.0, 0.0, 0.5])
    for epoch in range(1, 21):
        errors = 0
        for x, d in zip(X, D):
            e = d - degrau(x @ w)
            w += eta * e * x
            errors += int(e != 0)
        if errors == 0:
            break
    y = degrau(X @ w)  # todas as previsões de uma vez: produto matriz-vetor
    print(f"convergiu na época {epoch}: w = {w.tolist()} · previsões = {y.tolist()} · desejado = {D.tolist()}")
    return w


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--rapida" not in args:
        manual()
    if "--manual" not in args:
        rapida()
