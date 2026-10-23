#!/usr/bin/env python3
"""
CoffeeAI · Aula 1 · Perceptron: Charizard (Fogo) x Blastoise (Água)
=====================================================================

Simulação da Aula 1 de https://ricksonlivio.github.io/CoffeeAI/#learn/1
Python puro, sem bibliotecas extras. Rode com:

    python3 perceptron_pokemon.py

O que o script faz, na mesma ordem da aula:
  1. monta o problema (entradas, pesos iniciais, taxa de aprendizado);
  2. treina época por época, imprimindo cada conta: net, saída, erro e ajuste dos pesos;
  3. testa o Venusaur, que não estava no treino;
  4. treina de novo com η = 0,1 e η = 0,01 para comparar.

Convenções (iguais às da página):
  x1 = Velocidade, x2 = Defesa, x0 = -1 (entrada fixa do bias, com peso θ)
  net = w1·x1 + w2·x2 + θ·(-1)
  y = 1 (Fogo) se net >= 0, senão y = 0 (Água)
  e = d - y;  Δw1 = η·e·x1;  Δw2 = η·e·x2;  Δθ = η·e·(-1)

Autor: Rickson Livio de Souza Gaspar · CoffeeAI
"""

X0 = -1  # entrada fixa do bias

# Atributos base dos jogos: (nome, Velocidade, Defesa, saída desejada d)
CHARIZARD = ("Charizard", 100, 78, 1)  # Fogo
BLASTOISE = ("Blastoise", 78, 100, 0)  # Água


def fmt(v):
    """Número como a aula escreve: sem zeros sobrando (100.0 -> 100, -0.5 -> -0.5)."""
    v = round(v, 9) + 0.0  # limpa ruído de ponto flutuante e o -0.0
    return f"{v:g}"


def step(net):
    """Função degrau: 1 se net >= 0, senão 0."""
    return 1 if net >= 0 else 0


def train(samples, w1, w2, theta, eta, max_epochs=20, verbose=True):
    """Regra de aprendizagem do perceptron, uma amostra por vez, até uma época sem erros."""
    for epoch in range(1, max_epochs + 1):
        if verbose:
            print(f"\n=== Época {epoch} ===")
        errors = 0
        for name, x1, x2, d in samples:
            # 1. Somatória (net)
            p1, p2, p0 = w1 * x1, w2 * x2, theta * X0
            net = p1 + p2 + p0
            # 2. Função degrau
            y = step(net)
            # 3. Erro
            e = d - y
            if verbose:
                tipo = "Fogo" if y == 1 else "Água"
                print(f"\n{name}: x1 = {x1} (Velocidade), x2 = {x2} (Defesa), d = {d}")
                print(f"  pesos atuais: w1 = {fmt(w1)}, w2 = {fmt(w2)}, θ = {fmt(theta)}")
                print(f"  net = ({fmt(w1)}·{x1}) + ({fmt(w2)}·{x2}) + ({fmt(theta)}·-1) = "
                      f"{fmt(p1)} + {fmt(p2)} + {fmt(p0)} = {fmt(net)}")
                print(f"  net {'>= 0' if net >= 0 else '< 0'}  =>  y = {y} (o perceptron acha que é {tipo})")
                print(f"  e = d - y = {d} - {y} = {e}" + ("  -> acertou, nenhum ajuste" if e == 0 else "  -> errou, ajustar pesos"))
            # 4. Atualização (só muda alguma coisa se e != 0)
            if e != 0:
                errors += 1
                new_w1 = w1 + eta * e * x1
                new_w2 = w2 + eta * e * x2
                new_theta = theta + eta * e * X0
                if verbose:
                    print(f"  w1 = {fmt(w1)} + ({fmt(eta)}·{e}·{x1}) = {fmt(new_w1)}")
                    print(f"  w2 = {fmt(w2)} + ({fmt(eta)}·{e}·{x2}) = {fmt(new_w2)}")
                    print(f"  θ  = {fmt(theta)} + ({fmt(eta)}·{e}·-1) = {fmt(new_theta)}")
                w1, w2, theta = new_w1, new_w2, new_theta
        if verbose:
            print(f"\nFim da época {epoch}: {errors} erro(s); pesos w1 = {fmt(w1)}, w2 = {fmt(w2)}, θ = {fmt(theta)}")
        if errors == 0:
            return w1, w2, theta, epoch, True
    return w1, w2, theta, max_epochs, False


def classify(w1, w2, theta, x1, x2):
    net = w1 * x1 + w2 * x2 + theta * X0
    return net, step(net)


if __name__ == "__main__":
    print("Aula 1 · Perceptron: Charizard (Fogo) x Blastoise (Água)")
    print("Pesos iniciais: w1 = 0, w2 = 0, θ = 0.5 · taxa de aprendizado η = 1")

    w1, w2, theta, epochs, ok = train([CHARIZARD, BLASTOISE], 0, 0, 0.5, eta=1)
    print("\n" + "-" * 60)
    print(f"Convergiu: {ok}, em {epochs} épocas. Pesos finais: w1 = {fmt(w1)}, w2 = {fmt(w2)}, θ = {fmt(theta)}")
    print(f"Regra aprendida: é Fogo quando {fmt(w1)}·Velocidade + ({fmt(w2)})·Defesa >= {fmt(theta)},")
    print("isto é, quando a Velocidade passa a Defesa.")

    # Venusaur (Planta): fora do treino. Com uma saída 0/1, só existem dois lados.
    net, y = classify(w1, w2, theta, 80, 83)
    print(f"\nVenusaur (Velocidade 80, Defesa 83): net = {fmt(net)} => y = {y} ({'Fogo' if y else 'Água'})")
    print("O perceptron não conhece o tipo Planta: veja a Aula 2 (MLP).")

    # Mesma tarefa com passos menores: os pesos encolhem, a reta continua separando os dois.
    print("\nComparando taxas de aprendizado:")
    for eta in (1, 0.1, 0.01):
        a, b, c, n, ok = train([CHARIZARD, BLASTOISE], 0, 0, 0.5, eta=eta, verbose=False)
        print(f"  η = {fmt(eta):>5}: convergiu = {ok} em {n} épocas -> w1 = {fmt(a)}, w2 = {fmt(b)}, θ = {fmt(c)}")
