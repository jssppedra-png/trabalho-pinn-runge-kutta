# Exemplo 6: o caso em que a PINN falha, com "fotos" da curva durante o treino.
#
# Mesma ideia do café, mas com algo que esfria MUITO rápido (uma colher de metal quente
# mergulhada em água a 20 graus). Começa a 90 graus e em ~6 segundos já está quase em 20.
# Em unidades "arrumadas": u' = -50u, u(0) = 1, com o tempo s indo de 0 a 1 minuto.
# Resposta exata: T(s) = 20 + 70 * exp(-50 s)
import json
import math

from exemplo3_pinn_rede import treinar_pinn

iteracoes_foto = [0, 5, 20, 50, 100, 200, 400, 1000, 2000, 4000]
fotos = {it: None for it in iteracoes_foto}

erro, _, _ = treinar_pinn(
    f=lambda s, u: -50 * u,
    df_dy=lambda s, u: -50.0,
    y0=1.0,
    exata=lambda s: math.exp(-50 * s),
    iteracoes=4000,
    fotos=fotos,
)

resultado = []
for it in iteracoes_foto:
    curva, loss, segundos = fotos[it]
    temperaturas = [round(20 + 70 * u, 2) for u in curva]
    resultado.append({"iteracao": it, "loss": loss, "segundos": round(segundos, 2), "T": temperaturas})
    print(f"iteração {it:5d}  tempo {segundos:5.2f} s  loss {loss if loss is None else round(loss, 4)}  "
          f"T(0) = {temperaturas[0]:7.2f}  T(3 s) = {temperaturas[2]:7.2f}  T(30 s) = {temperaturas[20]:7.2f}")

print(f"\nexato: T(0) = 90.00  T(3 s) = {20 + 70 * math.exp(-50 * 0.05):.2f}  T(30 s) = 20.00")

with open("fotos_falha.json", "w", encoding="utf-8") as arq:
    json.dump(resultado, arq)
