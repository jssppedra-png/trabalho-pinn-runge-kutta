# Exemplo 5: a PINN aprendendo o problema do café, com "fotos" da curva durante o treino.
#
# Regra: a cada minuto o café perde 10% da diferença para a sala (20 graus). Começa a 90 graus.
# Resposta exata: T(t) = 20 + 70 * 0.9^t, para t de 0 a 10 minutos.
#
# Truque comum em PINNs: a rede trabalha com números "arrumados" entre 0 e 1:
#   s = t / 10           (tempo de 0 a 1 em vez de 0 a 10)
#   u = (T - 20) / 70    (u = 1 no começo, u = 0 na temperatura da sala)
# Nessas unidades, a regra vira u' = -10*ln(1/0.9) * u, com u(0) = 1.
import json
import math

from exemplo3_pinn_rede import treinar_pinn

k = 10 * math.log(1 / 0.9)
iteracoes_foto = [0, 5, 20, 50, 100, 200, 400, 1000, 2000, 4000]
fotos = {it: None for it in iteracoes_foto}

erro, _, _ = treinar_pinn(
    f=lambda s, u: -k * u,
    df_dy=lambda s, u: -k,
    y0=1.0,
    exata=lambda s: math.exp(-k * s),
    iteracoes=4000,
    semente=3,
    fotos=fotos,
)

resultado = []
for it in iteracoes_foto:
    curva, loss, segundos = fotos[it]
    temperaturas = [round(20 + 70 * u, 2) for u in curva]   # volta para graus
    resultado.append({"iteracao": it, "loss": loss, "segundos": round(segundos, 2), "T": temperaturas})
    print(f"iteração {it:5d}  tempo {segundos:5.2f} s  "
          f"T(0) = {temperaturas[0]:7.2f}  T(5 min) = {temperaturas[20]:7.2f}  T(10 min) = {temperaturas[40]:7.2f}")

print(f"\nexato:                      T(0) =   90.00  T(5 min) = {20 + 70 * 0.9**5:7.2f}  T(10 min) = {20 + 70 * 0.9**10:7.2f}")
print(f"erro máximo final: {erro * 70:.3f} graus")

with open("fotos_cafe.json", "w", encoding="utf-8") as arq:
    json.dump(resultado, arq)
