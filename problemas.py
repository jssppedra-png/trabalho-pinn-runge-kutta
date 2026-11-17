# Os problemas do exercício 1, num lugar só, para o rk.py e o pinn.py usarem exatamente os mesmos.
#
# Cada problema tem:
#   f(t, y, xp)   : a "regra de mudança" y' = f(t, y). y pode ter mais de um número (um sistema).
#   y0            : o ponto de partida y(0), uma lista com um número para cada variável
#   intervalo     : de onde até onde no tempo
#   exata(t, xp)  : a resposta certa (quando existe fórmula), usada só para medir o erro
#
# "xp" é a caixa de ferramentas de contas: numpy (no rk.py) ou torch (no pinn.py).
# As duas têm xp.exp, xp.cos, xp.sin, xp.stack, então a mesma fórmula serve para os dois métodos.
# Convenção: a última posição de y guarda as variáveis. y[..., 0] é a primeira, y[..., 1] a segunda.
import numpy as np
from scipy.integrate import solve_ivp


def lotka_volterra(t, y, xp):
    coelhos, raposas = y[..., 0], y[..., 1]
    d_coelhos = coelhos - coelhos * raposas    # coelhos nascem; são comidos quando encontram raposas
    d_raposas = -raposas + coelhos * raposas   # raposas morrem de fome; crescem quando comem coelhos
    return xp.stack([d_coelhos, d_raposas], -1)


PROBLEMAS = {
    # Questão (i): a "colher que esfria rápido". Resposta: y = e^(-50t)
    "rigido": {
        "descricao": "y' = -50y, y(0) = 1",
        "f": lambda t, y, xp: -50 * y,
        "y0": [1.0],
        "intervalo": (0.0, 1.0),
        "exata": lambda t, xp: xp.exp(-50 * t),
    },
    # Questão (ii): algo que oscila rápido. Resposta: y = sen(20t) / 20
    "oscilante": {
        "descricao": "y' = cos(20t), y(0) = 0",
        "f": lambda t, y, xp: xp.cos(20 * t),
        "y0": [0.0],
        "intervalo": (0.0, 1.0),
        "exata": lambda t, xp: xp.sin(20 * t) / 20,
    },
    # Questão (iii): coelhos e raposas (Lotka-Volterra), um sistema não rígido.
    # Não existe fórmula exata: a "resposta certa" vem de um Runge-Kutta de altíssima precisão.
    "coelhos_raposas": {
        "descricao": "Lotka-Volterra: c' = c - c r, r' = -r + c r, c(0) = 2, r(0) = 1",
        "f": lotka_volterra,
        "y0": [2.0, 1.0],
        "intervalo": (0.0, 10.0),
        "exata": None,
    },
}

# Mesma régua para os dois métodos: erro máximo (em todas as variáveis) numa grade de 1000 pontos.
PONTOS_DA_REGUA = 1000


def grade_da_regua(nome):
    a, b = PROBLEMAS[nome]["intervalo"]
    return np.linspace(a, b, PONTOS_DA_REGUA).reshape(-1, 1)   # uma coluna de tempos


def resposta_certa(nome, t):
    """Resposta certa nos tempos t (coluna). Devolve uma tabela: uma linha por tempo, uma coluna por variável."""
    p = PROBLEMAS[nome]
    if p["exata"] is not None:
        return p["exata"](t, np)
    # Sem fórmula: referência com DOP853 (Runge-Kutta de ordem 8) e tolerância 1e-13,
    # mil vezes mais exigente que a tolerância dos métodos comparados.
    ref = solve_ivp(lambda s, y: p["f"](s, y, np), p["intervalo"], p["y0"],
                    method="DOP853", rtol=1e-13, atol=1e-13, dense_output=True)
    return ref.sol(t.ravel()).T
