# Exercício 1: PINN mínima versus Runge-Kutta adaptativo (RK45, tolerância 1e-10).
#
# Este script gera, sozinho, TODAS as tabelas e figuras do exercício 1, na pasta "resultados/".
# Regras do trabalho que ele segue:
#   - nenhum número é digitado à mão: tabelas e legendas saem do código;
#   - a semente é fixada uma única vez, no começo, fora de qualquer laço;
#   - roda do início ao fim sem intervenção:   python exercicio1.py   (uns 3 minutos)
#
# Comparação justa (protocolo de McGreivy e Hakim, 2024):
#   - mesmo computador e mesmo processo para os dois métodos;
#   - mesma régua de erro: erro máximo numa grade de 1000 pontos, contra a mesma resposta certa;
#   - mesma ficha de problema (problemas.py) para os dois;
#   - além do ponto RK45 com tolerância 1e-10, a curva inteira custo x precisão do RK45
#     (várias tolerâncias), para ver se a PINN fica acima ou abaixo dela.
import csv
import json
import os
import platform

import matplotlib

matplotlib.use("Agg")   # desenha direto em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy.integrate import solve_ivp

import pinn
import rk
from problemas import PROBLEMAS, resposta_certa

SEMENTE = 0
PASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")   # sempre ao lado deste script
TOLERANCIAS_CURVA = [10.0 ** -k for k in range(3, 13)]   # 1e-3, 1e-4, ..., 1e-12
REPETICOES_TEMPO_RK = 20                                # o RK é rápido: mede 20 vezes e usa a mediana

# Rótulos curtos de cada problema (usados nas tabelas e figuras)
ROTULO = {
    "rigido": "(i) y' = -50y",
    "oscilante": "(ii) y' = cos(20t)",
    "coelhos_raposas": "(iii) coelhos e raposas",
}

# Os treinos de PINN, NA ORDEM em que rodam (a ordem importa: todos usam a mesma sequência de sorteios).
# (identificador, problema, descrição do método, opções de treino)
TREINOS_PINN = [
    ("pinn_i", "rigido", "PINN, 5 mil iterações", {}),
    ("pinn_ii", "oscilante", "PINN, 5 mil iterações", {}),
    ("pinn_iii", "coelhos_raposas", "PINN, 5 mil iterações", {}),
    ("pinn_iii_20k", "coelhos_raposas", "PINN, 20 mil iterações", {"n_iteracoes": 20000}),
    ("pinn_i_peso", "rigido", "PINN, fiscal do início x 100", {"peso_inicio": 100.0}),
    ("pinn_i_amarrada", "rigido", "PINN, curva amarrada no início", {"amarrar": True}),
]


def numero(x, casas=2):
    """Escreve um número em notação científica com vírgula: 9,39e-11."""
    return f"{x:.{casas}e}".replace(".", ",")


def milhar(n):
    """Escreve um inteiro com ponto de milhar: 1.000.000."""
    return f"{n:,}".replace(",", ".")


def rodar_tudo():
    resultados = {"rk": {}, "rk_curva": {}, "pinn": {}}

    # 1) Runge-Kutta com a tolerância do enunciado, nos três problemas
    for nome in PROBLEMAS:
        print(f"RK45 em {ROTULO[nome]} ...")
        resultados["rk"][nome] = rk.resolver(nome, tol=1e-10, repeticoes=REPETICOES_TEMPO_RK)

    # 2) Curva custo x precisão do RK45 (várias tolerâncias)
    for nome in PROBLEMAS:
        resultados["rk_curva"][nome] = [rk.resolver(nome, tol=tol, repeticoes=5) for tol in TOLERANCIAS_CURVA]

    # 3) PINNs
    for ident, nome, desc, opcoes in TREINOS_PINN:
        print(f"{desc} em {ROTULO[nome]} ...")
        r = pinn.treinar(nome, **opcoes)
        r["problema"], r["descricao"] = nome, desc
        resultados["pinn"][ident] = r
        print(f"   erro {numero(r['erro'])}, {r['tempo_s']:.1f} s")
    return resultados


def salvar_tabelas(res):
    linhas = []
    for nome in PROBLEMAS:
        r = res["rk"][nome]
        linhas.append([ROTULO[nome], "RK45, tolerância 1e-10", r["erro"], r["tempo_s"], r["aval_f"]])
        for ident, prob, desc, _ in TREINOS_PINN:
            if prob == nome:
                p = res["pinn"][ident]
                linhas.append([ROTULO[nome], desc, p["erro"], p["tempo_s"], p["aval_f"]])

    cabecalho = ["Problema", "Método", "Erro máximo", "Tempo (s)", "Avaliações de f"]

    with open(os.path.join(PASTA, "tabela.csv"), "w", newline="", encoding="utf-8") as arq:
        w = csv.writer(arq)
        w.writerow(cabecalho)
        for prob, met, e, t, n in linhas:
            w.writerow([prob, met, f"{e:.3e}", f"{t:.4f}", n])

    with open(os.path.join(PASTA, "tabela.md"), "w", encoding="utf-8") as arq:
        arq.write("| " + " | ".join(cabecalho) + " |\n|" + "---|" * len(cabecalho) + "\n")
        for prob, met, e, t, n in linhas:
            arq.write(f"| {prob} | {met} | {numero(e)} | {numero(t)} | {milhar(n)} |\n")

    with open(os.path.join(PASTA, "tabela.tex"), "w", encoding="utf-8") as arq:
        arq.write("\\begin{tabular}{llrrr}\n\\hline\n")
        arq.write(" & ".join(cabecalho) + " \\\\\n\\hline\n")
        for prob, met, e, t, n in linhas:
            prob_tex = prob.replace("y' = -50y", "$y' = -50y$").replace("y' = cos(20t)", "$y' = \\cos(20t)$")
            arq.write(f"{prob_tex} & {met} & {numero(e)} & {numero(t)} & {milhar(n)} \\\\\n")
        arq.write("\\hline\n\\end{tabular}\n")
    return linhas


def figura_solucoes(res):
    """Figura 1: resposta certa, passos do RK45 e PINN (5 mil iterações) em cada problema."""
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, nome in zip(eixos, PROBLEMAS):
        p = PROBLEMAS[nome]
        t = np.linspace(*p["intervalo"], 2000).reshape(-1, 1)
        certa = resposta_certa(nome, t)
        r = res["rk"][nome]
        pn = next(res["pinn"][i] for i, prob, _, o in TREINOS_PINN if prob == nome and not o)
        nomes_var = ["coelhos", "raposas"] if certa.shape[1] == 2 else ["y"]
        cores = ["tab:green", "tab:orange"] if certa.shape[1] == 2 else ["tab:red"]
        for k, (nv, cor) in enumerate(zip(nomes_var, cores)):
            ax.plot(t, certa[:, k], "k--", lw=1.2, label="resposta certa" if k == 0 else None)
            ax.plot(r["t_passos"], r["y_passos"][:, k], "o", ms=2.5, color="tab:blue",
                    label=f"passos do RK45 ({r['passos']})" if k == 0 else None)
            ax.plot(pn["t_regua"], pn["y_regua"][:, k], color=cor, lw=2,
                    label=f"PINN ({nv})" if certa.shape[1] == 2 else "PINN")
        ax.set_title(f"{ROTULO[nome]}\nerro RK45 {numero(r['erro'])}  |  erro PINN {numero(pn['erro'])}", fontsize=10)
        ax.set_xlabel("t")
        if certa.shape[1] == 2:
            ax.set_ylim(0.3, 2.75)   # abre espaço em cima para a legenda não cobrir as curvas
            ax.legend(fontsize=7, ncol=2, loc="upper center")
        else:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "fig1_solucoes.png"), dpi=150)
    plt.close(fig)


def figura_custo_precisao(res):
    """Figura 2: custo x precisão. Curva do RK45 (várias tolerâncias) e os pontos das PINNs."""
    fig, eixos = plt.subplots(2, 3, figsize=(15, 8))
    marcadores = ["s", "^", "D", "v"]
    for col, nome in enumerate(PROBLEMAS):
        curva = res["rk_curva"][nome]
        pinns = [(i, res["pinn"][i]) for i, prob, _, _ in TREINOS_PINN if prob == nome]
        for lin, (chave, rotulo_x) in enumerate([("aval_f", "avaliações de f"), ("tempo_s", "tempo (s)")]):
            ax = eixos[lin, col]
            ax.loglog([c[chave] for c in curva], [c["erro"] for c in curva], "o-", color="tab:blue",
                      label="RK45 (tolerâncias 1e-3 a 1e-12)")
            r10 = res["rk"][nome]
            ax.loglog(r10[chave], r10["erro"], "*", ms=14, color="tab:blue", label="RK45, tolerância 1e-10")
            for m, (i, pn) in zip(marcadores, pinns):
                ax.loglog(pn[chave], pn["erro"], m, ms=9, color="tab:red", alpha=0.85, label=pn["descricao"])
            ax.set_xlabel(rotulo_x)
            ax.set_ylabel("erro máximo")
            ax.grid(True, which="major", alpha=0.3)
            if lin == 0:
                ax.set_title(ROTULO[nome], fontsize=11)
            ax.legend(fontsize=7)
    fig.suptitle("Custo x precisão: embaixo e à esquerda é melhor", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "fig2_custo_precisao.png"), dpi=150)
    plt.close(fig)


def figura_custo_compacta(res):
    """Figura 5: versão compacta da figura 2 (só avaliações de f), para o relatório curto."""
    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.3))
    marcadores = ["s", "^", "D", "v"]
    for ax, nome in zip(eixos, PROBLEMAS):
        curva = res["rk_curva"][nome]
        ax.loglog([c["aval_f"] for c in curva], [c["erro"] for c in curva], "o-", ms=4, color="tab:blue",
                  label="RK45, tolerâncias 1e-3 a 1e-12")
        r10 = res["rk"][nome]
        ax.loglog(r10["aval_f"], r10["erro"], "*", ms=12, color="tab:blue", label="RK45, tolerância 1e-10")
        pinns = [res["pinn"][i] for i, prob, _, _ in TREINOS_PINN if prob == nome]
        for m, pn in zip(marcadores, pinns):
            ax.loglog(pn["aval_f"], pn["erro"], m, ms=8, color="tab:red", alpha=0.85, label=pn["descricao"])
        ax.set_title(ROTULO[nome], fontsize=10)
        ax.set_xlabel("avaliações de f")
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=6, loc="lower left")
    eixos[0].set_ylabel("erro máximo")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "fig5_custo_compacta.png"), dpi=200)
    plt.close(fig)


def figura_remedios(res):
    """Figura 3: questão (i) com a PINN original e os dois remédios para o ponto de partida."""
    nome = "rigido"
    t = np.linspace(0, 0.3, 1000).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t, resposta_certa(nome, t), "k--", lw=1.5, label="resposta certa, e^(-50t)")
    estilos = {"pinn_i": "-", "pinn_i_peso": "-.", "pinn_i_amarrada": ":"}
    for ident, est in estilos.items():
        pn = res["pinn"][ident]
        sel = pn["t_regua"] <= 0.3
        ax.plot(pn["t_regua"][sel], pn["y_regua"][sel, 0], est, lw=2.2,
                label=f"{pn['descricao']} (erro {numero(pn['erro'])})")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title("(i) y' = -50y: a falha da PINN e dois remédios", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "fig3_remedios_rigido.png"), dpi=150)
    plt.close(fig)


def figura_fase(res):
    """Figura 4: coelhos x raposas. A resposta certa é um laço; a PINN curta espirala para o equilíbrio."""
    nome = "coelhos_raposas"
    t = np.linspace(*PROBLEMAS[nome]["intervalo"], 2000).reshape(-1, 1)
    certa = resposta_certa(nome, t)
    fig, ax = plt.subplots(figsize=(5.5, 6.4))
    ax.plot(certa[:, 0], certa[:, 1], "k--", lw=1.5, label="resposta certa")
    for ident, est in [("pinn_iii", "-"), ("pinn_iii_20k", "-.")]:
        pn = res["pinn"][ident]
        ax.plot(pn["y_regua"][:, 0], pn["y_regua"][:, 1], est, lw=2,
                label=f"{pn['descricao']} (erro {numero(pn['erro'])})")
    ax.plot(1, 1, "o", color="gray", ms=8, label="equilíbrio (1, 1)")
    ax.plot(*PROBLEMAS[nome]["y0"], "s", color="black", ms=7, label="ponto de partida")
    ax.set_xlabel("coelhos (mil)")
    ax.set_ylabel("raposas (mil)")
    ax.set_title("(iii) coelhos x raposas", fontsize=11)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)   # legenda embaixo do gráfico
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "fig4_fase_coelhos_raposas.png"), dpi=150)
    plt.close(fig)


def tex_cientifico(x, casas=1):
    """Número em notação científica para LaTeX: 9{,}4 \\times 10^{-11} (vírgula decimal)."""
    mantissa, expoente = f"{x:.{casas}e}".split("e")
    mantissa = mantissa.replace(".", "{,}")
    expoente = int(expoente)
    if expoente == 0:
        return f"\\ensuremath{{{mantissa}}}"
    return f"\\ensuremath{{{mantissa} \\times 10^{{{expoente}}}}}"


def tex_simples(x, casas=1):
    """Número comum para LaTeX, com vírgula decimal e ponto de milhar: 1.234{,}5."""
    texto = f"{x:,.{casas}f}".replace(",", "X").replace(".", "{,}").replace("X", ".")
    return f"\\ensuremath{{{texto}}}"


def salvar_numeros_tex(res):
    """Gera resultados/numeros.tex: um 'apelido' LaTeX para cada número citado no texto do relatório.
    Assim o texto escreve \\ErroPINNOscilante e nunca um número digitado à mão."""
    nome_tex = {"rigido": "Rigido", "oscilante": "Oscilante", "coelhos_raposas": "Coelhos"}
    nome_pinn = {"pinn_i": "Rigido", "pinn_ii": "Oscilante", "pinn_iii": "Coelhos",
                 "pinn_iii_20k": "CoelhosLongo", "pinn_i_peso": "RigidoPeso", "pinn_i_amarrada": "RigidoAmarrada"}
    m = {}
    for nome, r in res["rk"].items():
        n = nome_tex[nome]
        m[f"ErroRK{n}"] = tex_cientifico(r["erro"])
        m[f"TempoRK{n}"] = tex_simples(r["tempo_s"] * 1000, 1) + "\\,ms"
        m[f"AvalRK{n}"] = tex_simples(r["aval_f"], 0)
        m[f"PassosRK{n}"] = tex_simples(r["passos"], 0)
    for ident, r in res["pinn"].items():
        n = nome_pinn[ident]
        m[f"ErroPINN{n}"] = tex_cientifico(r["erro"])
        m[f"TempoPINN{n}"] = tex_simples(r["tempo_s"], 1) + "\\,s"
        m[f"AvalPINN{n}"] = tex_simples(r["aval_f"], 0)
        m[f"MultaPINN{n}"] = tex_cientifico(r["loss_final"])
    # Melhor PINN de cada problema (menor erro) comparada com o RK45 de tolerância 1e-10
    razoes_erro, razoes_aval = [], []
    for nome, n in nome_tex.items():
        rk_r = res["rk"][nome]
        pn = min((res["pinn"][i] for i, prob, _, _ in TREINOS_PINN if prob == nome), key=lambda r: r["erro"])
        razoes_erro.append(pn["erro"] / rk_r["erro"])
        razoes_aval.append(pn["aval_f"] / rk_r["aval_f"])
        # Quantas avaliações o RK45 precisa (na curva de tolerâncias) para ter erro igual ou menor que essa PINN
        suficientes = [c["aval_f"] for c in res["rk_curva"][nome] if c["erro"] <= pn["erro"]]
        m[f"AvalRKMesmoErro{n}"] = tex_simples(min(suficientes), 0) if suficientes else "(fora da curva)"
        m[f"ErroMelhorPINN{n}"] = tex_cientifico(pn["erro"])
    m["RazaoErroMin"] = tex_cientifico(min(razoes_erro), 0)
    m["RazaoErroMax"] = tex_cientifico(max(razoes_erro), 0)
    m["RazaoAvalMin"] = tex_simples(min(razoes_aval), 0)
    m["RazaoAvalMax"] = tex_simples(max(razoes_aval), 0)
    m["ToleranciaMinima"] = tex_cientifico(TOLERANCIAS_CURVA[-1], 0)
    m["ToleranciaMaxima"] = tex_cientifico(TOLERANCIAS_CURVA[0], 0)
    # Conferência do gabarito de (iii): DOP853 contra Radau, e a quantidade conservada V = c - ln c + r - ln r
    nome = "coelhos_raposas"
    p = PROBLEMAS[nome]
    t = np.linspace(*p["intervalo"], 1000)
    ref = resposta_certa(nome, t.reshape(-1, 1))
    radau = solve_ivp(lambda s, y: p["f"](s, y, np), p["intervalo"], p["y0"], method="Radau",
                      rtol=1e-13, atol=1e-13, dense_output=True)
    V = ref[:, 0] - np.log(ref[:, 0]) + ref[:, 1] - np.log(ref[:, 1])
    m["DifRefRadau"] = tex_cientifico(np.max(np.abs(radau.sol(t).T - ref)), 0)
    m["VariacaoInvariante"] = tex_cientifico(V.max() - V.min(), 0)
    m["RepeticoesTempoRK"] = tex_simples(REPETICOES_TEMPO_RK, 0)
    m["Semente"] = tex_simples(SEMENTE, 0)
    m["ThreadsTorch"] = tex_simples(torch.get_num_threads(), 0)
    m["VersaoTorch"] = torch.__version__.replace("+", "\\texttt{+}")
    m["VersaoScipy"] = scipy.__version__

    with open(os.path.join(PASTA, "numeros.tex"), "w", encoding="utf-8") as arq:
        arq.write("% Gerado automaticamente por exercicio1.py. NÃO EDITE À MÃO: rode o script de novo.\n")
        for chave, valor in m.items():
            arq.write(f"\\newcommand{{\\{chave}}}{{{valor}}}\n")


def salvar_json(res):
    """Todos os números num arquivo só, mais as informações do computador (para a comparação justa)."""
    def limpar(r):
        return {k: v for k, v in r.items() if k in ("erro", "tempo_s", "aval_f", "passos", "loss_final", "descricao")}

    saida = {
        "semente": SEMENTE,
        "computador": {"sistema": platform.platform(), "processador": platform.processor(),
                       "threads_torch": torch.get_num_threads()},
        "versoes": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "torch": torch.__version__},
        "rk45_tol_1e-10": {n: limpar(r) for n, r in res["rk"].items()},
        "rk45_curva": {n: [dict(tol=tol, **limpar(c)) for tol, c in zip(TOLERANCIAS_CURVA, cs)]
                       for n, cs in res["rk_curva"].items()},
        "pinn": {i: dict(problema=r["problema"], **limpar(r)) for i, r in res["pinn"].items()},
    }
    with open(os.path.join(PASTA, "resultados.json"), "w", encoding="utf-8") as arq:
        json.dump(saida, arq, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    torch.manual_seed(SEMENTE)          # semente fixada UMA única vez, antes de tudo
    os.makedirs(PASTA, exist_ok=True)
    res = rodar_tudo()
    linhas = salvar_tabelas(res)
    figura_solucoes(res)
    figura_custo_precisao(res)
    figura_custo_compacta(res)
    figura_remedios(res)
    figura_fase(res)
    salvar_numeros_tex(res)
    salvar_json(res)
    print(f"\nPronto. Tabelas e figuras em '{PASTA}/':")
    for nome_arq in sorted(os.listdir(PASTA)):
        print("  ", nome_arq)
