# Resolve um problema de problemas.py com uma PINN mínima (PyTorch).
# Uso:  python pinn.py rigido   |   python pinn.py oscilante   |   python pinn.py coelhos_raposas
import sys
import time

import numpy as np
import torch

from problemas import PROBLEMAS, grade_da_regua, resposta_certa


def treinar(nome, n_iteracoes=5000, momentos_foto=(), peso_inicio=1.0, amarrar=False):
    """Treina uma PINN para o problema `nome`.

    A semente NÃO é fixada aqui: quem chama fixa uma única vez, no começo do programa
    (regra de reprodutibilidade do trabalho).
    peso_inicio : quanto vale a multa do ponto de partida (remédio 1: dar mais voz a esse fiscal).
    amarrar     : se True, a resposta é montada como y0 + s * rede(s), que vale y0 em s = 0 de qualquer jeito
                  (remédio 2: amarrar a curva no ponto de partida).
    """
    p = PROBLEMAS[nome]
    a, b = p["intervalo"]
    n_var = len(p["y0"])          # quantas variáveis (1 ou 2)
    y0 = torch.tensor([p["y0"]])

    # A rede neural: uma função y(t) cheia de botões (pesos) ajustáveis.
    # Entra 1 número (o tempo), passa por 2 camadas de 32 neurônios, sai 1 número por variável.
    corpo = torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, n_var),
    )

    def rede(t):
        s = (t - a) / (b - a)     # a rede enxerga o tempo "arrumado" entre 0 e 1 (em [0, 1] não muda nada)
        if amarrar:
            return y0 + s * corpo(s)
        return corpo(s)

    # Pontos de colocação: onde os "fiscais" conferem se a rede obedece a regra.
    # requires_grad=True permite calcular a derivada dy/dt automaticamente.
    t_col = torch.linspace(a, b, 200).reshape(-1, 1).requires_grad_(True)
    t0 = torch.full((1, 1), a)

    otimizador = torch.optim.Adam(corpo.parameters(), lr=1e-3)
    grade_foto = torch.linspace(a, b, 201).reshape(-1, 1)
    fotos = []

    inicio = time.perf_counter()
    for it in range(n_iteracoes):
        otimizador.zero_grad()

        y = rede(t_col)
        # Derivada de cada variável em relação a t (diferenciação automática)
        dy_dt = torch.cat([
            torch.autograd.grad(y[:, k:k + 1], t_col, torch.ones_like(t_col), create_graph=True)[0]
            for k in range(n_var)
        ], dim=1)

        multa_regra = ((dy_dt - p["f"](t_col, y, torch)) ** 2).mean()   # quanto desobedece a regra
        multa_inicio = ((rede(t0) - y0) ** 2).mean()                     # quanto erra o ponto de partida
        loss = multa_regra + peso_inicio * multa_inicio

        if it in momentos_foto:
            with torch.no_grad():
                fotos.append({"it": it, "regra": multa_regra.item(), "inicio": multa_inicio.item(),
                              "y": rede(grade_foto).tolist()})

        loss.backward()     # calcula para que lado girar cada botão
        otimizador.step()   # gira os botões um pouquinho
    tempo = time.perf_counter() - inicio

    # A régua comum: erro máximo numa grade de 1000 pontos, em todas as variáveis
    t = grade_da_regua(nome)
    with torch.no_grad():
        y_pinn = rede(torch.tensor(t, dtype=torch.float32)).numpy()
    erro = float(np.max(np.abs(y_pinn - resposta_certa(nome, t))))

    return {"erro": erro, "tempo_s": tempo, "aval_f": n_iteracoes * len(t_col),
            "loss_final": loss.item(), "fotos": fotos,
            "t_regua": t.ravel(), "y_regua": y_pinn}


if __name__ == "__main__":
    nome = sys.argv[1] if len(sys.argv) > 1 else "rigido"
    torch.manual_seed(0)   # semente fixada uma única vez, antes de tudo
    r = treinar(nome)
    print(f"PINN em {PROBLEMAS[nome]['descricao']}")
    print(f"  erro máximo      : {r['erro']:.2e}")
    print(f"  tempo            : {r['tempo_s']:.1f} s")
    print(f"  avaliações de f  : {r['aval_f']}  (5000 iterações x 200 pontos)")
    print(f"  multa final      : {r['loss_final']:.2e}")
