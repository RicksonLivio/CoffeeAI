#!/usr/bin/env python3
"""
CoffeeAI · Aula 2 · MLP: Fogo, Água ou Planta
=============================================

Simulação da Aula 2 de https://ricksonlivio.github.io/CoffeeAI/#learn/2
Python puro, sem bibliotecas extras. Rode com:

    python3 mlp_pokemon.py           # tudo
    python3 mlp_pokemon.py --plot    # também desenha gráficos, se você tiver o matplotlib

Partes, na ordem da aula:
  1. o problema: um único perceptron "Planta ou não" nunca para de errar;
  2. a camada oculta feita à mão (função degrau, pesos escolhidos);
  3. o MLP que aprende: sigmoide, softmax, entropia cruzada e backpropagation,
     com a primeira época impressa conta por conta;
  4. o tamanho do passo η: mínimo local x mínimo global numa superfície de brinquedo,
     e o mesmo MLP com η = 0,2, 2 e 10;
  5. o mesmo treino com duas camadas ocultas (2 -> 2 -> 2 -> 3).

Convenção de todas as aulas: cada neurônio tem um bias θ ligado a uma entrada fixa -1,
então net = Σ w·x + θ·(-1). Os pesos iniciais abaixo são os mesmos da página
(sorteados com semente fixa e arredondados a uma casa).

Sobre bater com o site: a primeira época, as contas à mão e o resultado final (acertos,
época do primeiro 9/9) saem iguais. No meio de um treino longo podem aparecer diferenças
pequenas nas últimas casas, porque a função exp do Python e a do navegador arredondam o
último dígito de formas diferentes, e com η = 2 o treino amplifica essas diferenças.

Autor: Rickson Livio de Souza Gaspar · CoffeeAI
"""
import math
import sys

# --------------------------------------------------------------------------- #
# Dados: os três iniciais de cada tipo, com Velocidade e Defesa base dos jogos
# --------------------------------------------------------------------------- #
TYPES = ["Fogo", "Água", "Planta"]
STARTERS = [
    ("Charmander", 65, 43, "Fogo"), ("Charmeleon", 80, 58, "Fogo"), ("Charizard", 100, 78, "Fogo"),
    ("Squirtle", 43, 65, "Água"), ("Wartortle", 58, 80, "Água"), ("Blastoise", 78, 100, "Água"),
    ("Bulbasaur", 45, 49, "Planta"), ("Ivysaur", 60, 63, "Planta"), ("Venusaur", 80, 83, "Planta"),
]
SCALE = 100  # Velocidade 80 -> 0.8: a sigmoide começa na parte inclinada, onde aprende


def f4(v):
    return f"{v:.4f}"


# --------------------------------------------------------------------------- #
# Parte 1 · um perceptron sozinho não isola o tipo Planta
# --------------------------------------------------------------------------- #
def part1():
    print("\n" + "=" * 70 + "\nPARTE 1 · Um perceptron 'Planta ou não' (regra da Aula 1)\n" + "=" * 70)
    for name, x1, x2, t in STARTERS:
        print(f"  {name:<11} {t:<7} Velocidade − Defesa = {x1 - x2:+d}")
    w1, w2, theta = 0.0, 0.0, 0.5
    errors_per_epoch = []
    for _ in range(100):
        errors = 0
        for _, x1, x2, t in STARTERS:
            d = 1 if t == "Planta" else 0
            y = 1 if w1 * x1 + w2 * x2 - theta >= 0 else 0
            e = d - y
            if e:
                errors += 1
                w1, w2, theta = w1 + e * x1, w2 + e * x2, theta - e
        errors_per_epoch.append(errors)
    print(f"\n  Erros nas primeiras 8 épocas: {errors_per_epoch[:8]}")
    print(f"  Menor número de erros em 100 épocas: {min(errors_per_epoch)} (nunca zero)")
    print("  Motivo: Ivysaur e Venusaur estão dentro da área cercada por Charmander, Charizard,")
    print("  Blastoise e Squirtle; nenhuma reta deixa os Planta de um lado e o resto do outro.")


# --------------------------------------------------------------------------- #
# Parte 2 · camada oculta feita à mão, com a função degrau
# --------------------------------------------------------------------------- #
def degrau(net):
    return 1 if net >= 0 else 0


def part2():
    print("\n" + "=" * 70 + "\nPARTE 2 · Camada oculta à mão (degrau)\n" + "=" * 70)
    print("  h1 = degrau(x1 − x2 − 11): detector de Fogo")
    print("  h2 = degrau(x2 − x1 − 11): detector de Água")
    print("  Fogo = degrau(h1 − 0.5) · Água = degrau(h2 − 0.5) · Planta = degrau(0.5 − h1 − h2)")
    for name, x1, x2, t in STARTERS:
        n1, n2 = x1 - x2 - 11, x2 - x1 - 11  # cada neurônio oculto é um perceptron
        h1, h2 = degrau(n1), degrau(n2)
        o = [degrau(h1 - 0.5), degrau(h2 - 0.5), degrau(0.5 - h1 - h2)]
        pred = TYPES[o.index(1)]
        print(f"  {name:<11} net_h = ({n1:+d}, {n2:+d}) -> h = ({h1}, {h2}) -> saídas {o} -> {pred:<6} "
              f"{'✓' if pred == t else '✗'}")
    # Sem ativação (f(net) = net), a saída Planta fica constante: as duas retas se cancelam.
    print("  Sem ativação: Planta = 0.5 − (x1 − x2 − 11) − (x2 − x1 − 11) =",
          0.5 - (80 - 83 - 11) - (83 - 80 - 11), "para qualquer Pokémon (a faixa some).")


# --------------------------------------------------------------------------- #
# Rede genérica: sigmoide nas ocultas, softmax na saída, entropia cruzada
# Cada camada é uma lista de neurônios {"w": [...], "theta": ...}
# --------------------------------------------------------------------------- #
def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def softmax(z):
    m = max(z)  # subtrair o maior z não muda o resultado e evita overflow
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def forward(layers, x1, x2):
    """Devolve as ativações a[0..L] (a[0] = entrada escalada, a[L] = probabilidades p) e as somas z."""
    a = [[x1 / SCALE, x2 / SCALE]]
    zs = []
    for li, layer in enumerate(layers):
        z = [sum((w * x for w, x in zip(n["w"], a[-1])), 0.0) + n["theta"] * -1 for n in layer]
        zs.append(z)
        a.append(softmax(z) if li == len(layers) - 1 else [sigmoid(v) for v in z])
    return a, zs


def backprop(layers, x1, x2, t):
    """Para a frente, erro na saída e erro levado para trás, tudo com os pesos atuais."""
    a, zs = forward(layers, x1, x2)
    p = a[-1]
    d = [1.0 if k == t else 0.0 for k in TYPES]
    loss = -math.log(p[TYPES.index(t)])
    deltas = [None] * len(layers)
    deltas[-1] = [pk - dk for pk, dk in zip(p, d)]  # saída: δ = p − d
    for li in range(len(layers) - 2, -1, -1):        # ocultas: de trás para a frente
        act = a[li + 1]
        deltas[li] = [sum(n["w"][j] * deltas[li + 1][k] for k, n in enumerate(layers[li + 1])) * h * (1 - h)
                      for j, h in enumerate(act)]
    # gradiente de cada peso: δ do neurônio de chegada × a entrada que o peso carrega (bias: × −1)
    grads = [[{"w": [deltas[li][j] * x for x in a[li]], "theta": deltas[li][j] * -1} for j in range(len(layer))]
             for li, layer in enumerate(layers)]
    return a, zs, deltas, grads, loss


def epoch(layers, eta):
    """Uma época em lote completo: gradiente médio dos nove, depois todos os pesos juntos."""
    total, acc, sums = 0.0, 0, None
    for _, x1, x2, t in STARTERS:
        a, _, _, g, loss = backprop(layers, x1, x2, t)
        total += loss
        acc += TYPES[a[-1].index(max(a[-1]))] == t
        if sums is None:
            sums = g
        else:
            for li in range(len(g)):
                for j in range(len(g[li])):
                    sums[li][j]["w"] = [s + v for s, v in zip(sums[li][j]["w"], g[li][j]["w"])]
                    sums[li][j]["theta"] += g[li][j]["theta"]
    n = len(STARTERS)
    # média primeiro, depois o passo: peso <- peso − η · (soma / n)
    new = [[{"w": [w - eta * (gw / n) for w, gw in zip(nr["w"], sums[li][j]["w"])],
             "theta": nr["theta"] - eta * (sums[li][j]["theta"] / n)}
            for j, nr in enumerate(layer)] for li, layer in enumerate(layers)]
    return new, total / n, acc


def train(layers, eta, epochs, report_every=0):
    curve = []
    for e in range(epochs + 1):
        new, loss, acc = epoch(layers, eta)
        curve.append((e, loss, acc))
        if report_every and (e % report_every == 0 or e == epochs):
            print(f"  época {e:>4}: erro médio L = {f4(loss)} · acertos {acc}/9")
        if e < epochs:
            layers = new
    return layers, curve


def copy(layers):
    return [[{"w": list(n["w"]), "theta": n["theta"]} for n in layer] for layer in layers]


# Pesos iniciais da página (semente fixa, uma casa decimal)
INIT_1H = [
    [{"w": [0.5, -0.4], "theta": -0.4}, {"w": [0.1, 0.8], "theta": 0.3}],                       # h1, h2
    [{"w": [0.0, -0.3], "theta": 0.6}, {"w": [-0.8, 0.6], "theta": -0.6}, {"w": [-0.2, 0.7], "theta": 0.2}],  # saídas
]
INIT_2H = [
    [{"w": [0.5, -0.4], "theta": -0.4}, {"w": [0.1, 0.8], "theta": 0.3}],                       # h1, h2
    [{"w": [0.0, -0.3], "theta": 0.6}, {"w": [-0.8, 0.6], "theta": -0.6}],                      # g1, g2
    [{"w": [-0.2, 0.7], "theta": 0.2}, {"w": [-0.7, 0.3], "theta": -0.3}, {"w": [0.9, -0.9], "theta": 0.8}],  # saídas
]


def explain_first_epoch(layers, eta, names):
    """Imprime a primeira época para o Venusaur, conta por conta, como o passo a passo da página."""
    x1, x2, t = 80, 83, "Planta"
    a, zs, deltas, grads, loss = backprop(layers, x1, x2, t)
    L = len(layers)
    print(f"\n  1 · Para a frente (Venusaur, x = ({x1 / SCALE}, {x2 / SCALE}))")
    for li in range(L):
        for j, z in enumerate(zs[li]):
            nm = names[li][j] if li < L - 1 else TYPES[j]
            extra = f" -> {nm} = σ = {f4(a[li + 1][j])}" if li < L - 1 else ""
            print(f"     z_{nm} = {f4(z)}{extra}")
    print(f"     p = softmax = ({', '.join(f4(v) for v in a[-1])}) · soma = {sum(a[-1]):.4f}")
    print(f"     L = −ln p_Planta = {f4(loss)}")
    print("  2 · Erro na saída: δ = p − d")
    for k, dv in enumerate(deltas[-1]):
        print(f"     δ_{TYPES[k]} = {f4(dv)}")
    print("  3 · Erro para trás (com os pesos ANTIGOS)")
    for li in range(L - 2, -1, -1):
        for j, dv in enumerate(deltas[li]):
            print(f"     δ_{names[li][j]} = {f4(dv)}")
    print("  4 · Gradiente médio dos 9 Pokémon  5 · Atualização de TODOS os pesos juntos")
    new, mean_loss, acc = epoch(layers, eta)
    w_old, w_new = layers[0][0]["w"][0], new[0][0]["w"][0]
    print(f"     ex.: w1,1 = {f4(w_old)} − {eta}·∂L/∂w1,1 = {f4(w_new)}")
    print(f"     erro médio da época 1: {f4(mean_loss)} · acertos {acc}/9")


def show_softmax(layers):
    a, zs = forward(layers, 80, 83)
    z = zs[-1]
    e = [math.exp(v) for v in z]
    s = sum(e)
    print("\n  Softmax para o Venusaur (rede treinada):")
    for k in range(3):
        print(f"     {TYPES[k]:<6} z = {f4(z[k]):>8}  e^z = {f4(e[k]):>10}  p = e^z/S = {f4(e[k] / s)}")
    print(f"     S = {f4(s)} · soma das probabilidades = {sum(v / s for v in e):.4f}"
          f" · resposta = {TYPES[e.index(max(e))]}")


def part3(plot):
    print("\n" + "=" * 70 + "\nPARTE 3 · MLP 2 -> 2 -> 3 aprendendo com backpropagation (η = 2)\n" + "=" * 70)
    explain_first_epoch(copy(INIT_1H), 2, [["h1", "h2"]])
    print("\n  Treino completo (1500 épocas, lote completo):")
    final, curve = train(copy(INIT_1H), 2, 1500, report_every=250)
    first = next(e for e, _, acc in curve if acc == 9)
    print(f"  Acerta os nove pela primeira vez na época {first}.")
    for name, x1, x2, t in STARTERS:
        a, _ = forward(final, x1, x2)
        p = a[-1]
        print(f"  {name:<11} Fogo {100 * p[0]:5.1f}%  Água {100 * p[1]:5.1f}%  Planta {100 * p[2]:5.1f}%  "
              f"{'✓' if TYPES[p.index(max(p))] == t else '✗'}")
    h1, h2 = final[0]
    print(f"  Pesos aprendidos: h1 w = ({h1['w'][0]:.1f}, {h1['w'][1]:.1f}), h2 w = ({h2['w'][0]:.1f}, {h2['w'][1]:.1f})"
          " -> medem Velocidade − Defesa com sinais opostos.")
    show_softmax(final)
    if plot:
        plot_curve(curve, "MLP com 1 camada oculta (η = 2)")
    return curve


# --------------------------------------------------------------------------- #
# Parte 4 · tamanho do passo: mínimo local x global
# --------------------------------------------------------------------------- #
# Superfície de brinquedo com dois vales: o global (largo e fundo) e o local (estreito e raso)
GLOBAL = (1.2, 1.0, 1.0, 1.6)   # centro a, centro b, profundidade, largura
LOCAL = (-1.5, -1.1, 0.55, 0.3)
BOWL = 0.05


def surface(a, b):
    g = lambda c: c[2] * math.exp(-((a - c[0]) ** 2 + (b - c[1]) ** 2) / c[3])
    return 1 - g(GLOBAL) - g(LOCAL) + BOWL * (a * a + b * b)


def gradient(a, b):
    da, db = 2 * BOWL * a, 2 * BOWL * b
    for c in (GLOBAL, LOCAL):
        k = 2 * c[2] * math.exp(-((a - c[0]) ** 2 + (b - c[1]) ** 2) / c[3]) / c[3]
        da += k * (a - c[0])
        db += k * (b - c[1])
    return da, db


def part4(plot):
    print("\n" + "=" * 70 + "\nPARTE 4 · Tamanho do passo: mínimo local x mínimo global\n" + "=" * 70)
    paths = {}
    for eta in (0.1, 0.7, 1.5):
        a, b = -2.3, -2.0
        path = [(a, b)]
        for _ in range(120):
            da, db = gradient(a, b)
            a, b = a - eta * da, b - eta * db  # (a, b) <- (a, b) − η·∇L
            path.append((a, b))
        paths[eta] = path
        dist = lambda c: math.hypot(a - c[0], b - c[1])
        where = "mínimo global" if dist(GLOBAL) < 0.35 else "mínimo local" if dist(LOCAL) < 0.35 else "outro ponto"
        moving = math.hypot(path[-1][0] - path[-2][0], path[-1][1] - path[-2][1]) > 0.05
        print(f"  η = {eta}: termina em ({a:+.2f}, {b:+.2f}), L = {surface(a, b):.3f} -> {where}"
              f"{' (ainda pulando de um lado para o outro)' if moving else ''}")
    print("\n  O mesmo MLP da parte 3 com outros passos:")
    for eta in (0.2, 2, 10):
        final, curve = train(copy(INIT_1H), eta, 1500)
        first = next((e for e, _, acc in curve if acc == 9), None)
        print(f"  η = {eta:>4}: erro final {f4(curve[-1][1])}, acertos {curve[-1][2]}/9, "
              f"primeira vez 9/9 na época {first}")
    print("  Com η = 10 as sigmoides saturam (h ≈ 0), o erro não chega à camada oculta e a rede trava.")
    if plot:
        plot_landscape(paths)


# --------------------------------------------------------------------------- #
# Parte 5 · duas camadas ocultas
# --------------------------------------------------------------------------- #
def part5(plot):
    print("\n" + "=" * 70 + "\nPARTE 5 · Duas camadas ocultas: 2 -> 2 -> 2 -> 3 (21 pesos, η = 2)\n" + "=" * 70)
    explain_first_epoch(copy(INIT_2H), 2, [["h1", "h2"], ["g1", "g2"]])
    _, _, deltas, _, _ = backprop(INIT_2H, 80, 83, "Planta")
    mags = [max(abs(v) for v in d) for d in deltas]
    print(f"\n  Maior |δ| por camada na época 1: oculta 1 = {f4(mags[0])}, oculta 2 = {f4(mags[1])}, saída = {f4(mags[2])}")
    print("  (o erro enfraquece a cada camada: a derivada da sigmoide nunca passa de 0,25)")
    print("\n  Treino completo (2000 épocas):")
    final, curve = train(copy(INIT_2H), 2, 2000, report_every=400)
    first = next(e for e, _, acc in curve if acc == 9)
    print(f"  Acerta os nove pela primeira vez na época {first}.")
    show_softmax(final)
    if plot:
        plot_curve(curve, "MLP com 2 camadas ocultas (η = 2)")


# --------------------------------------------------------------------------- #
# Gráficos opcionais (precisam do matplotlib: pip install matplotlib)
# --------------------------------------------------------------------------- #
def plot_curve(curve, title):
    import matplotlib.pyplot as plt
    ep = [c[0] for c in curve]
    plt.figure(figsize=(7, 3.5))
    plt.semilogy(ep, [c[1] for c in curve], color="#f5b52a", label="erro médio L")
    plt.xlabel("época")
    plt.ylabel("L (log)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_landscape(paths):
    import matplotlib.pyplot as plt
    n = 120
    grid = [-3 + 6 * i / n for i in range(n + 1)]
    Z = [[surface(a, b) for a in grid] for b in grid]
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.contourf(grid, grid, Z, levels=20, cmap="viridis_r")
    ax1.contour(grid, grid, Z, levels=20, colors="k", linewidths=0.4)
    for eta, color in zip(paths, ("#7df0ff", "#e8312f", "#ffd25e")):
        ax1.plot([p[0] for p in paths[eta]], [p[1] for p in paths[eta]], color=color, marker=".", label=f"η = {eta}")
    ax1.set_title("curvas de nível + descida do gradiente")
    ax1.set_xlabel("peso a")
    ax1.set_ylabel("peso b")
    ax1.legend()
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    A = [[a for a in grid] for _ in grid]
    B = [[b for _ in grid] for b in grid]
    import numpy as np  # vem junto com o matplotlib
    ax2.plot_surface(np.array(A), np.array(B), np.array(Z), cmap="viridis_r", alpha=0.85)
    ax2.set_title("superfície 3D do erro (arraste para girar)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    want_plot = "--plot" in sys.argv
    if want_plot:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("matplotlib não encontrado; rodando sem gráficos (pip install matplotlib)")
            want_plot = False
    part1()
    part2()
    part3(want_plot)
    part4(want_plot)
    part5(want_plot)
