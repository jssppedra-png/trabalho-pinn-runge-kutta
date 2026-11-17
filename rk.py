# Resolve um problema de problemas.py com Runge-Kutta adaptativo (RK45 do SciPy, tolerância 1e-10).
# Uso:  python rk.py rigido   |   python rk.py oscilante   |   python rk.py coelhos_raposas
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

from problemas import PROBLEMAS, grade_da_regua, resposta_certa


def resolver(nome, tol=1e-10, repeticoes=1):
    """RK45 com tolerância `tol`. Roda `repeticoes` vezes e guarda o tempo mediano (o do meio)."""
    p = PROBLEMAS[nome]

    def f(t, y):
        # Mesma regra do problemas.py, usando a caixa de ferramentas numpy
        return p["f"](t, y, np)

    tempos = []
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        sol = solve_ivp(
            f,
            t_span=p["intervalo"],   # intervalo de tempo
            y0=p["y0"],              # ponto de partida
            method="RK45",
            rtol=tol,                # tolerância relativa
            atol=tol,                # tolerância absoluta
            dense_output=True,       # permite avaliar a solução em qualquer t
        )
        tempos.append(time.perf_counter() - inicio)
    tempo = float(np.median(tempos))

    # A régua comum: erro máximo numa grade de 1000 pontos, em todas as variáveis
    t = grade_da_regua(nome)
    y_rk = sol.sol(t.ravel()).T            # uma linha por tempo, uma coluna por variável
    erro = np.max(np.abs(y_rk - resposta_certa(nome, t)))

    return {"erro": erro, "tempo_s": tempo, "aval_f": sol.nfev, "passos": len(sol.t) - 1,
            "t_passos": sol.t, "y_passos": sol.y.T}


if __name__ == "__main__":
    nome = sys.argv[1] if len(sys.argv) > 1 else "rigido"
    r = resolver(nome)
    print(f"Runge-Kutta (RK45) em {PROBLEMAS[nome]['descricao']}")
    print(f"  erro máximo      : {r['erro']:.2e}")
    print(f"  tempo            : {r['tempo_s'] * 1000:.1f} ms")
    print(f"  avaliações de f  : {r['aval_f']}")
    print(f"  passos dados     : {r['passos']}")
