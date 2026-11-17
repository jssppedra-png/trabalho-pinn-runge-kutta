# Exemplo 3: uma PINN de verdade (rede neural com 1 camada escondida), em Python puro.
# Resolve os 3 problemas: y' = -y,  y' = -50y,  y' = cos(20t), todos com t em [0, 1].
#
# Rede: y(t) = c + soma_i v_i * tanh(w_i * t + b_i)   (N neurônios -> 3N + 1 botões)
# Loss: média de r(t)^2 nos pontos de colocação + (y(0) - y0)^2
#       onde r(t) = y'(t) - f(t, y(t))
# Os gradientes são calculados à mão (o PyTorch faria isso sozinho).
import math
import random
import sys
import time


def treinar_pinn(f, df_dy, y0, exata, N=20, n_pontos=50, iteracoes=4000, taxa=0.01, semente=0,
                 botoes_iniciais=None, mostrar=True, fotos=None):
    # fotos: dicionário {iteração: None}; é preenchido com (curva, loss, segundos) nessas iterações
    random.seed(semente)
    w = [random.gauss(0, 2) for _ in range(N)]
    b = [random.gauss(0, 1) for _ in range(N)]
    v = [random.gauss(0, 0.1) for _ in range(N)]
    c = 0.0
    if botoes_iniciais:   # começa de uma rede já treinada (usado no currículo)
        w, b, v, c = [x[:] for x in botoes_iniciais[:3]] + [botoes_iniciais[3]]
    pontos = [i / (n_pontos - 1) for i in range(n_pontos)]

    # Adam: o otimizador mais usado em PINNs (gradiente descendente com "inércia")
    params = [w, b, v]
    m = [[0.0] * N for _ in range(3)]; s = [[0.0] * N for _ in range(3)]
    mc = sc = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8

    def rede(t):
        y = c; dy = 0.0
        for i in range(N):
            th = math.tanh(w[i] * t + b[i])
            y += v[i] * th
            dy += v[i] * w[i] * (1 - th * th)
        return y, dy

    grade_fotos = [i / 40 for i in range(41)]
    if fotos is not None and 0 in fotos:
        fotos[0] = ([rede(t)[0] for t in grade_fotos], None, 0.0)

    inicio = time.perf_counter()
    for it in range(1, iteracoes + 1):
        g = [[0.0] * N for _ in range(3)]; gc = 0.0; L = 0.0
        # termo da física: resíduo em cada ponto de colocação
        for t in pontos:
            ths = [math.tanh(w[i] * t + b[i]) for i in range(N)]
            y = c + sum(v[i] * ths[i] for i in range(N))
            dy = sum(v[i] * w[i] * (1 - ths[i] ** 2) for i in range(N))
            r = dy - f(t, y)
            L += r * r / n_pontos
            k = 2 * r / n_pontos
            fy = df_dy(t, y)
            for i in range(N):
                th = ths[i]; sp = 1 - th * th
                # derivadas de y e de y' em relação a cada botão
                dy_dw, dyp_dw = v[i] * t * sp, v[i] * (sp - 2 * w[i] * th * sp * t)
                dy_db, dyp_db = v[i] * sp, v[i] * w[i] * (-2 * th * sp)
                dy_dv, dyp_dv = th, w[i] * sp
                g[0][i] += k * (dyp_dw - fy * dy_dw)
                g[1][i] += k * (dyp_db - fy * dy_db)
                g[2][i] += k * (dyp_dv - fy * dy_dv)
            gc += k * (0 - fy * 1)
        # termo da condição inicial: y(0) deve ser y0
        ths = [math.tanh(b[i]) for i in range(N)]
        e0 = c + sum(v[i] * ths[i] for i in range(N)) - y0
        L += e0 * e0
        for i in range(N):
            sp = 1 - ths[i] ** 2
            g[1][i] += 2 * e0 * v[i] * sp
            g[2][i] += 2 * e0 * ths[i]
        gc += 2 * e0

        # passo do Adam: gira cada botão
        for p in range(3):
            for i in range(N):
                m[p][i] = b1 * m[p][i] + (1 - b1) * g[p][i]
                s[p][i] = b2 * s[p][i] + (1 - b2) * g[p][i] ** 2
                mh = m[p][i] / (1 - b1**it); sh = s[p][i] / (1 - b2**it)
                params[p][i] -= taxa * mh / (math.sqrt(sh) + eps)
        mc = b1 * mc + (1 - b1) * gc; sc = b2 * sc + (1 - b2) * gc**2
        c -= taxa * (mc / (1 - b1**it)) / (math.sqrt(sc / (1 - b2**it)) + eps)

        if fotos is not None and it in fotos:
            fotos[it] = ([rede(t)[0] for t in grade_fotos], L, time.perf_counter() - inicio)
        if mostrar and it in (1, 10, 100, 500, 1000, 2000, iteracoes):
            print(f"    iteração {it:5d}   loss = {L:.3e}")
    tempo = time.perf_counter() - inicio

    grade = [i / 999 for i in range(1000)]
    erro = max(abs(rede(t)[0] - exata(t)) for t in grade)
    if mostrar:
        print(f"    -> erro máximo = {erro:.2e}   tempo = {tempo:.1f} s   "
              f"avaliações de f = {iteracoes * n_pontos}")
        print(f"    {'t':>5} {'PINN':>9} {'exato':>9}")
        for t in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
            print(f"    {t:5.2f} {rede(t)[0]:9.4f} {exata(t):9.4f}")
    return erro, (w, b, v, c), rede


problemas = {
    "y' = -y": (lambda t, y: -y, lambda t, y: -1.0, 1.0, lambda t: math.exp(-t)),
    "y' = -50y": (lambda t, y: -50 * y, lambda t, y: -50.0, 1.0, lambda t: math.exp(-50 * t)),
    "y' = cos(20t)": (lambda t, y: math.cos(20 * t), lambda t, y: 0.0, 0.0, lambda t: math.sin(20 * t) / 20),
}

if __name__ == "__main__":
    escolha = sys.argv[1:] or list(problemas)
    for nome in escolha:
        print(f"\n=== {nome} ===")
        treinar_pinn(*problemas[nome])
