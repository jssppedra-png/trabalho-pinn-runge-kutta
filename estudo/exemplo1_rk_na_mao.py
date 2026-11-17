# Exemplo 1: Euler e Runge-Kutta (RK4) escritos à mão, em Python puro.
# Problema: y' = -y, y(0) = 1, no intervalo [0, 1]. Solução exata: y(t) = exp(-t)
import math

contador = 0  # quantas vezes a regra f foi calculada


def f(t, y):
    global contador
    contador += 1
    return -y


def euler(f, y0, T, h):
    t, y = 0.0, y0
    for _ in range(round(T / h)):
        y = y + h * f(t, y)           # olha a inclinação 1 vez e anda
        t = t + h
    return y


def rk4(f, y0, T, h):
    t, y = 0.0, y0
    for _ in range(round(T / h)):
        k1 = f(t, y)                          # inclinação no começo
        k2 = f(t + h / 2, y + h / 2 * k1)     # no meio, usando k1
        k3 = f(t + h / 2, y + h / 2 * k2)     # no meio de novo, usando k2
        k4 = f(t + h, y + h * k3)             # no fim, usando k3
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)   # média ponderada
        t = t + h
    return y


exato = math.exp(-1)
print(f"Valor exato y(1) = {exato:.10f}\n")
print(f"{'passo h':>8} | {'erro Euler':>11} {'aval. f':>7} | {'erro RK4':>11} {'aval. f':>7} | erro RK4 caiu")

erro_anterior = None
for h in [0.5, 0.25, 0.125, 0.0625, 0.03125]:
    contador = 0
    erro_e = abs(euler(f, 1.0, 1.0, h) - exato)
    aval_e = contador

    contador = 0
    erro_r = abs(rk4(f, 1.0, 1.0, h) - exato)
    aval_r = contador

    queda = f"{erro_anterior / erro_r:5.1f}x" if erro_anterior else "   -"
    erro_anterior = erro_r
    print(f"{h:8.5f} | {erro_e:11.2e} {aval_e:7d} | {erro_r:11.2e} {aval_r:7d} | {queda}")
