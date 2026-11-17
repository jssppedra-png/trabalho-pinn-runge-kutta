# Refaz os treinos na MESMA ordem e com a MESMA semente de exercicio1.py, guardando fotos para as animações dos slides.
# Uso, na pasta principal do trabalho (uns 2 minutos):
#     python apresentacao/dados_animacoes.py . apresentacao/dados_animacoes.json
# Os números dentro das animações dos slides foram copiados desse arquivo .json.
import json
import sys

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
import pinn  # noqa: E402
import rk  # noqa: E402
from problemas import resposta_certa  # noqa: E402

torch.manual_seed(0)
M5 = {0, 20, 50, 100, 200, 300, 500, 750, 1000, 2000, 3000, 4999}
M20 = {0, 100, 1000, 2000, 3000, 4999, 7500, 9999, 14999, 19999}


def compacta(fotos, passo, casas):
    return [[f["it"], round(f["regra"] + f["inicio"], 6),
             np.round(np.array(f["y"])[::passo], casas).T.tolist()] for f in fotos]


out = {}
r = pinn.treinar("rigido", momentos_foto=M5)
out["i"] = {"fotos": compacta(r["fotos"], 4, 3), "erro": r["erro"]}
r = pinn.treinar("oscilante", momentos_foto=M5)
out["ii"] = {"fotos": compacta(r["fotos"], 4, 4), "erro": r["erro"]}
r = pinn.treinar("coelhos_raposas", momentos_foto=M5)
out["iii"] = {"fotos": compacta(r["fotos"], 4, 3), "erro": r["erro"]}
r = pinn.treinar("coelhos_raposas", n_iteracoes=20000, momentos_foto=M20)
out["iii20"] = {"fotos": compacta(r["fotos"], 4, 3), "erro": r["erro"]}
r = pinn.treinar("rigido", peso_inicio=100.0)
out["peso"] = {"y": np.round(r["y_regua"][:301:3, 0], 3).tolist(), "erro": r["erro"]}   # t de 0 a 0,3
r = pinn.treinar("rigido", amarrar=True)
out["amarrada"] = {"y": np.round(r["y_regua"][:301:3, 0], 3).tolist(), "erro": r["erro"]}

for nome in ["rigido", "oscilante", "coelhos_raposas"]:
    k = rk.resolver(nome)
    out["rk_" + nome] = {"t": np.round(k["t_passos"], 4).tolist(), "y": np.round(k["y_passos"], 4).T.tolist()}
t = np.linspace(0, 10, 51).reshape(-1, 1)
out["ref_coelhos"] = np.round(resposta_certa("coelhos_raposas", t), 3).T.tolist()
json.dump(out, open(sys.argv[2], "w"), separators=(",", ":"))
print({k: v.get("erro") for k, v in out.items() if isinstance(v, dict)})
