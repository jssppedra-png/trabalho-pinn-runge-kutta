# Exemplo 4: o remédio "treinamento por currículo" para y' = -50y.
# Em vez de atacar k = 50 direto, treina com y' = -k*y para k = 1, 5, 10, 20, 30, 40, 50,
# sempre começando da rede treinada no k anterior (como tirar as rodinhas aos poucos).
# O total de iterações é o mesmo do exemplo 3 (4000), para a comparação ser justa.
import math

from exemplo3_pinn_rede import treinar_pinn

etapas = [1, 5, 10, 20, 30, 40, 50]
iter_por_etapa = 4000 // len(etapas)
botoes = None

print(f"{'k':>4} {'erro máximo':>12}")
for k in etapas:
    erro, botoes, rede = treinar_pinn(
        f=lambda t, y, k=k: -k * y,
        df_dy=lambda t, y, k=k: -float(k),
        y0=1.0,
        exata=lambda t, k=k: math.exp(-k * t),
        iteracoes=iter_por_etapa,
        botoes_iniciais=botoes,
        mostrar=False,
    )
    print(f"{k:4d} {erro:12.2e}")

print(f"\nResultado final para y' = -50y:")
print(f"    {'t':>5} {'PINN':>9} {'exato':>9}")
for t in (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
    print(f"    {t:5.2f} {rede(t)[0]:9.4f} {math.exp(-50 * t):9.4f}")
