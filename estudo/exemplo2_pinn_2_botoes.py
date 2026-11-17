# Exemplo 2: a "PINN de brinquedo" com 2 botões, treinada por gradiente descendente.
# Problema: y' = -y, y(0) = 1. Solução exata: y(t) = exp(-t)
#
# "Rede": y(t) = 1 + a*t + b*t^2   (o "1 +" já garante y(0) = 1)
# Resíduo: r(t) = y'(t) + y(t) = (a + 2bt) + (1 + at + bt^2)
# Loss: média de r(t)^2 nos pontos de colocação
import math

pontos = [0.0, 0.5, 1.0]   # pontos de colocação


def y(t, a, b):
    return 1 + a * t + b * t**2


def residuo(t, a, b):
    dy = a + 2 * b * t                  # derivada de y(t)
    return dy + y(t, a, b)               # se y obedece y' = -y, isso dá 0


def loss(a, b):
    return sum(residuo(t, a, b) ** 2 for t in pontos) / len(pontos)


# ---- Parte A: treino passo a passo ----
a, b = 0.0, 0.0          # chute inicial: y(t) = 1, uma reta horizontal
taxa = 0.1               # tamanho do "giro" em cada botão (learning rate)

print("PARTE A: treinando os 2 botões\n")
print(f"{'iteração':>8} {'a':>9} {'b':>9} {'loss':>11} {'y(1)':>9}")
for it in range(2001):
    if it in (0, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000):
        print(f"{it:8d} {a:9.4f} {b:9.4f} {loss(a, b):11.2e} {y(1, a, b):9.4f}")
    # gradiente: para que lado cada botão aumenta a loss
    # dr/da = 1 + t     dr/db = 2t + t^2
    ga = sum(2 * residuo(t, a, b) * (1 + t) for t in pontos) / len(pontos)
    gb = sum(2 * residuo(t, a, b) * (2 * t + t**2) for t in pontos) / len(pontos)
    a -= taxa * ga       # gira o botão no sentido que DIMINUI a loss
    b -= taxa * gb

print(f"\nexato: y(1) = {math.exp(-1):.4f}   erro final = {abs(y(1, a, b) - math.exp(-1)):.4f}")


# ---- Parte B: e se a gente der mais botões? ----
# y(t) = 1 + a1*t + a2*t^2 + ... + an*t^n, com 20 pontos de colocação.
# Aqui o problema é "linear nos botões", então dá para achar os melhores botões
# por uma fórmula (mínimos quadrados), sem treino iterativo.
def resolver_sistema(M, v):
    # Eliminação de Gauss simples
    n = len(v)
    M = [linha[:] + [v[i]] for i, linha in enumerate(M)]
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(M[k][i]))
        M[i], M[p] = M[p], M[i]
        for k in range(i + 1, n):
            fator = M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] -= fator * M[i][j]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def melhor_polinomio(n, pontos):
    # r(t) = 1 + sum_k a_k * (k t^(k-1) + t^k); queremos r ~ 0 em todos os pontos
    A = [[k * t ** (k - 1) + t**k for k in range(1, n + 1)] for t in pontos]
    AtA = [[sum(A[p][i] * A[p][j] for p in range(len(pontos))) for j in range(n)] for i in range(n)]
    Atb = [sum(A[p][i] * -1 for p in range(len(pontos))) for i in range(n)]
    return resolver_sistema(AtA, Atb)


print("\nPARTE B: mais botões (20 pontos de colocação)\n")
print(f"{'botões':>6} {'erro máximo em [0,1]':>22}")
pontos20 = [i / 19 for i in range(20)]
grade = [i / 999 for i in range(1000)]
for n in range(1, 9):
    coef = melhor_polinomio(n, pontos20)
    def yn(t):
        return 1 + sum(c * t ** (k + 1) for k, c in enumerate(coef))
    erro = max(abs(yn(t) - math.exp(-t)) for t in grade)
    print(f"{n:6d} {erro:22.2e}")
