#!/usr/bin/env python3
"""
CoffeeAI · Aula 2 · parte 6 · Caderno: como aprendemos os pesos
===============================================================

Simulação da parte 6 da Aula 2 de https://ricksonlivio.github.io/CoffeeAI/#learn/2
Python puro, sem bibliotecas extras:  python3 backprop_notebook.py

  1. rede linear profunda f(x) = W_Lᵀ ⋯ W_1ᵀ x e o erro J = Σ ‖f(x_i) − y_i‖²   (eq. 1)
  2. backpropagation: ida guardando h, volta com δ_L = 2r e δ_ℓ = W_{ℓ+1} δ_{ℓ+1},
     gradiente ∇_{W_ℓ} J = Σ h_{ℓ−1} δ_ℓᵀ                                     (eqs. 4 a 8)
  3. conferência contra diferenças finitas, contando as avaliações de J
  4. a menor rede profunda, J(w1, w2) = (w2·w1 − 1)²: passos, sela em (0, 0), não convexidade
  5. três pesos, J(a, b, c) = (abc − 1)²: uma sela "ruim" (gradiente e Hessiana nulos)

Autor: Rickson Livio de Souza Gaspar · CoffeeAI
"""

# --------------------------------------------------------------------------- #
# Álgebra mínima com listas
# --------------------------------------------------------------------------- #
def T(A):
    return [list(col) for col in zip(*A)]


def matvec(A, v):
    return [sum(a * b for a, b in zip(row, v)) for row in A]


def outer(u, v):
    return [[a * b for b in v] for a in u]


def add(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def zeros_like(A):
    return [[0.0] * len(A[0]) for _ in A]


def fmt(M):
    return "[" + ", ".join("[" + ", ".join(f"{v: .4f}" for v in row) + "]" for row in M) + "]"


# Mesmo exemplo da página: L = 3, larguras 2 -> 2 -> 2 -> 1, quatro exemplos
W = [
    [[0.5, -0.3], [0.2, 0.8]],   # W_1: 2 x 2
    [[0.7, 0.1], [-0.4, 0.6]],   # W_2: 2 x 2
    [[0.9], [-0.5]],             # W_3: 2 x 1
]
DATA = [([1, 0], [1]), ([0, 1], [-1]), ([1, 1], [0]), ([0.5, -1], [1.5])]


def forward(W, x):
    """Ida: h_0 = x, h_ℓ = W_ℓᵀ h_{ℓ−1}. Guarda todos os h (a contabilidade)."""
    h = [x]
    for Wl in W:
        h.append(matvec(T(Wl), h[-1]))
    return h


def loss(W, data):
    total = 0.0
    for x, y in data:
        out = forward(W, x)[-1]
        total += sum((o - t) ** 2 for o, t in zip(out, y))
    return total


def backprop(W, data, verbose=False):
    """Volta: δ_L = 2r, δ_ℓ = W_{ℓ+1} δ_{ℓ+1}, gradiente = Σ h_{ℓ−1} δ_ℓᵀ."""
    grads = [zeros_like(Wl) for Wl in W]
    for n, (x, y) in enumerate(data):
        h = forward(W, x)
        r = [o - t for o, t in zip(h[-1], y)]
        delta = [None] * len(W)
        delta[-1] = [2 * v for v in r]                    # eq. 6
        for l in range(len(W) - 2, -1, -1):
            delta[l] = matvec(W[l + 1], delta[l + 1])     # eq. 5 (linear: σ' = 1)
        if verbose and n == 0:
            print(f"  exemplo 1: x = {x}, y = {y}")
            for l, hl in enumerate(h):
                print(f"    h_{l} = {[round(v, 4) for v in hl]}")
            print(f"    r = h_L − y = {[round(v, 4) for v in r]}")
            for l in range(len(W) - 1, -1, -1):
                print(f"    δ_{l + 1} = {[round(v, 4) for v in delta[l]]}")
        for l in range(len(W)):
            grads[l] = add(grads[l], outer(h[l], delta[l]))  # eq. 7
    return grads


def numeric(W, data, eps=1e-6):
    """Diferenças finitas centrais: 2 avaliações de J por peso."""
    evaluations = 0
    G = []
    for l, Wl in enumerate(W):
        Gl = zeros_like(Wl)
        for a in range(len(Wl)):
            for b in range(len(Wl[0])):
                up = [[row[:] for row in M] for M in W]
                dn = [[row[:] for row in M] for M in W]
                up[l][a][b] += eps
                dn[l][a][b] -= eps
                Gl[a][b] = (loss(up, data) - loss(dn, data)) / (2 * eps)
                evaluations += 2
        G.append(Gl)
    return G, evaluations


def part_backprop():
    print("=" * 70 + "\n1-3 · Backpropagation numa rede linear de 3 camadas\n" + "=" * 70)
    print(f"  J(W) = {loss(W, DATA):.4f}")
    G = backprop(W, DATA, verbose=True)
    Gn, evals = numeric(W, DATA)
    P = sum(len(M) * len(M[0]) for M in W)
    for l in range(len(W)):
        print(f"  ∇W_{l + 1} J (backprop)  = {fmt(G[l])}")
        print(f"  ∇W_{l + 1} J (dif. finita) = {fmt(Gn[l])}")
    diff = max(abs(a - b) for l in range(len(W)) for ra, rb in zip(G[l], Gn[l]) for a, b in zip(ra, rb))
    print(f"  maior diferença: {diff:.1e}")
    print(f"  custo: diferença finita = {evals} avaliações de J para {P} pesos; backprop = 1 ida + 1 volta")


# --------------------------------------------------------------------------- #
# 4 · J(w1, w2) = (w2·w1·x − y)², x = y = 1
# --------------------------------------------------------------------------- #
def J2(w1, w2):
    return (w2 * w1 - 1) ** 2


def grad2(w1, w2):
    h = w1                  # ida: h = w1·x
    d2 = 2 * (w2 * h - 1)   # δ2 = 2(ŷ − y)
    d1 = w2 * d2            # δ1 = w2·δ2
    return d1, d2 * h       # (∂J/∂w1, ∂J/∂w2)


def hessian2(w1, w2):
    return [[2 * w2 * w2, 2 * (2 * w1 * w2 - 1)], [2 * (2 * w1 * w2 - 1), 2 * w1 * w1]]


def eig2(H):
    tr = H[0][0] + H[1][1]
    det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
    d = max(0.0, tr * tr / 4 - det) ** 0.5
    return tr / 2 - d, tr / 2 + d


def descend(w1, w2, eta, steps):
    path = [(w1, w2)]
    for _ in range(steps):
        g1, g2 = grad2(w1, w2)
        w1, w2 = w1 - eta * g1, w2 - eta * g2
        path.append((w1, w2))
    return path


def part_toy():
    print("\n" + "=" * 70 + "\n4 · A menor rede profunda: J(w1, w2) = (w2·w1 − 1)²\n" + "=" * 70)
    print("  passo        w1        w2         J     ∂J/∂w1    ∂J/∂w2")
    for k, (a, b) in enumerate(descend(0.5, 0.5, 0.1, 8)):
        g = grad2(a, b)
        print(f"  {k:>5} {a:9.4f} {b:9.4f} {J2(a, b):9.4f} {g[0]:9.4f} {g[1]:9.4f}")
    print(f"\n  Não convexa: J(1,1) = {J2(1, 1)}, J(−1,−1) = {J2(-1, -1)}, mas J(0,0) = {J2(0, 0)} no ponto médio")
    print(f"  Hessiana em (0,0): {hessian2(0, 0)} -> autovalores {eig2(hessian2(0, 0))}: sela estrita")
    print(f"  Hessiana em (1,1): {hessian2(1, 1)} -> autovalores {eig2(hessian2(1, 1))}: mínimo global (vale plano)")
    for s in [(0.0, 0.0), (0.001, 0.001), (0.05, 0.02), (-1.4, 1.6), (-1.5, -0.2)]:
        p = descend(s[0], s[1], 0.1, 150)
        first = next((k for k, (a, b) in enumerate(p) if J2(a, b) < 1e-3), None)
        a, b = p[-1]
        where = "parada na sela (∇J = 0)" if abs(a) + abs(b) < 1e-9 else f"mínimo global, w1·w2 = {a * b:.3f}"
        print(f"  começo {str(s):<15} -> ({a:+.3f}, {b:+.3f}), J = {J2(a, b):.1e}, {where}; J < 0,001 no passo {first}")


# --------------------------------------------------------------------------- #
# 5 · J(a, b, c) = (abc − 1)²: sela ruim em 0
# --------------------------------------------------------------------------- #
def part_bad_saddle():
    print("\n" + "=" * 70 + "\n5 · Três pesos: J(a, b, c) = (abc − 1)²\n" + "=" * 70)
    J3 = lambda a, b, c: (a * b * c - 1) ** 2
    h = 1e-4
    H = []
    for i in range(3):
        row = []
        for j in range(3):
            e = lambda k, d: [d if q == k else 0.0 for q in range(3)]
            f = lambda p: J3(*p)
            pp = [u + v for u, v in zip(e(i, h), e(j, h))]
            pm = [u - v for u, v in zip(e(i, h), e(j, h))]
            mp = [u + v for u, v in zip(e(i, -h), e(j, h))]
            mm = [u - v for u, v in zip(e(i, -h), e(j, h))]
            row.append(round((f(pp) - f(pm) - f(mp) + f(mm)) / (4 * h * h), 6) + 0.0)
        H.append(row)
    print(f"  Hessiana numérica em (0,0,0): {H}  (toda zero: não denuncia a sela)")
    print(f"  J(0.1, 0.1, 0.1) = {J3(0.1, 0.1, 0.1):.6f} < 1 < J(−0.1, −0.1, −0.1) = {J3(-0.1, -0.1, -0.1):.6f} -> é uma sela")


if __name__ == "__main__":
    part_backprop()
    part_toy()
    part_bad_saddle()
