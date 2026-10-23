#!/usr/bin/env python3
"""
CoffeeAI · Aula 3 · Caderno em PyTorch
======================================

Precisa do PyTorch:  pip install torch
    python3 backprop_notebook_torch.py

  1. fórmula fechada do gradiente (eq. 8) para a rede linear profunda, comparada ao autograd,
     para L = 2, 3 e 5 camadas;
  2. Hessiana exata nas selas com torch.autograd.functional.hessian;
  3. treino da rede linear profunda com torch.optim.SGD (a versão rápida).

Autor: Rickson Livio de Souza Gaspar · CoffeeAI
"""
import torch
from torch.autograd.functional import hessian

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=4, sci_mode=False)
gen = torch.Generator().manual_seed(0)

N, d = 8, 3
X = torch.randn(N, d, generator=gen)            # cada linha é um x_i
Y = X @ torch.randn(d, 2, generator=gen)        # alvo linear com 2 saídas


def J(Ws):
    """Eq. 1: J = Σ ‖W_Lᵀ ⋯ W_1ᵀ x_i − y_i‖² (com linhas: X W_1 ⋯ W_L)."""
    out = X
    for W in Ws:
        out = out @ W
    return ((out - Y) ** 2).sum()


def closed_form(Ws):
    """Eq. 8: ∇_{W_ℓ} J = 2 Σ_i (W_{ℓ−1}ᵀ⋯W_1ᵀ x_i)(W_{ℓ+1}⋯W_L r_i)ᵀ."""
    L = len(Ws)
    H = [X]                                     # ida: H[ℓ] tem h_ℓ de cada exemplo nas linhas
    for W in Ws:
        H.append(H[-1] @ W)
    R = H[-1] - Y                               # resíduos r_i nas linhas
    grads = []
    for l in range(L):
        back = R
        for W in reversed(Ws[l + 1:]):          # W_{ℓ+1} ⋯ W_L r  (nas linhas: R W_Lᵀ ⋯ W_{ℓ+1}ᵀ)
            back = back @ W.T
        grads.append(2 * H[l].T @ back)
    return grads


print("=" * 70 + "\n1 · Eq. 8 x autograd\n" + "=" * 70)
for widths in ([3, 4, 2], [3, 4, 4, 2], [3, 5, 4, 4, 3, 2]):
    Ws = [(0.5 * torch.randn(a, b, generator=gen)).requires_grad_(True) for a, b in zip(widths[:-1], widths[1:])]
    J(Ws).backward()
    manual = closed_form([W.detach() for W in Ws])
    diff = max((W.grad - G).abs().max().item() for W, G in zip(Ws, manual))
    print(f"  L = {len(Ws)} camadas, larguras {widths}: maior diferença = {diff:.1e}")

print("\n" + "=" * 70 + "\n2 · Hessianas exatas nas selas\n" + "=" * 70)
J2 = lambda w: (w[1] * w[0] - 1) ** 2
J3 = lambda w: (w[2] * w[1] * w[0] - 1) ** 2
H2 = hessian(J2, torch.zeros(2))
H3 = hessian(J3, torch.zeros(3))
print("  J(w1, w2) = (w2 w1 − 1)² em 0: H =\n", H2, "\n  autovalores", torch.linalg.eigvalsh(H2), "-> sela estrita")
print("  J(a, b, c) = (abc − 1)² em 0: H =\n", H3, "\n  autovalores", torch.linalg.eigvalsh(H3), "-> sela ruim (nenhum negativo)")

print("\n" + "=" * 70 + "\n3 · Versão rápida: treino da rede linear profunda com SGD\n" + "=" * 70)
model = torch.nn.Sequential(torch.nn.Linear(3, 4, bias=False), torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 2, bias=False))
torch.manual_seed(0)
for m in model:
    torch.nn.init.normal_(m.weight, std=0.5)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
for step in range(1001):
    loss = ((model(X) - Y) ** 2).sum()           # mesma J da eq. 1
    if step % 200 == 0:
        print(f"  passo {step:>4}: J = {loss.item():.6f}")
    opt.zero_grad()
    loss.backward()                               # backpropagation automático
    opt.step()                                    # W ← W − η ∇J, todas as camadas juntas
print("  J chega perto de 0: o mínimo global de uma rede linear sem mínimos locais ruins.")
