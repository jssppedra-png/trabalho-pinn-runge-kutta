# PINN versus Runge–Kutta: exercício 1

Trabalho de Computação Científica. Equipe: José Salgado, Renan Brêttas e Heitor Morais.

**Pergunta central:** para um problema de valor inicial bem posto, uma PINN (rede neural informada pela física)
consegue igualar um Runge–Kutta adaptativo em custo × precisão?

**Exercício 1:** uma PINN mínima em PyTorch para y′ = f(t, y), comparada com `scipy.integrate.solve_ivp`
(RK45, tolerância 10⁻¹⁰) em erro, tempo e número de avaliações de f, em três problemas:

| | Problema | Intervalo |
|---|---|---|
| (i) | y′ = −50y, y(0) = 1 | t ∈ [0, 1] |
| (ii) | y′ = cos(20t), y(0) = 0 | t ∈ [0, 1] |
| (iii) | coelhos e raposas (Lotka–Volterra): c′ = c − cr, r′ = −r + cr, c(0) = 2, r(0) = 1 | t ∈ [0, 10] |

## Como rodar (um comando, sem intervenção)

```bash
python -m pip install -r requirements.txt
python exercicio1.py
```

Leva cerca de 2 minutos numa CPU comum e gera tudo em `resultados/`:

| Arquivo | Conteúdo |
|---|---|
| `tabela.md`, `tabela.csv`, `tabela.tex` | Erro máximo, tempo e avaliações de f de cada método |
| `fig1_solucoes.png` | Resposta certa, passos do RK45 e PINN nos três problemas |
| `fig2_custo_precisao.png` | Custo × precisão: curva do RK45 (tolerâncias 10⁻³ a 10⁻¹²) e pontos das PINNs |
| `fig3_remedios_rigido.png` | Problema (i): a falha da PINN e dois remédios para a condição inicial |
| `fig4_fase_coelhos_raposas.png` | Problema (iii) no plano de fase |
| `numeros.tex` | Todos os números citados no relatório, como comandos LaTeX |
| `resultados.json` | Todos os números, mais versões das bibliotecas e dados do computador |

## Reprodutibilidade

- A semente aleatória é fixada **uma única vez**, no início de `exercicio1.py` (`torch.manual_seed(0)`), fora de qualquer laço.
- Nenhum número do relatório é digitado à mão: o texto usa os comandos de `resultados/numeros.tex`.
- Os erros são idênticos de uma rodada para outra; os tempos variam um pouco com a carga do computador
  (o tempo do RK45 é a mediana de 20 execuções).
- Mesma régua para os dois métodos: erro máximo numa grade de 1000 pontos, contra a mesma resposta certa.
  No problema (iii), que não tem fórmula, a referência é o DOP853 com tolerância 10⁻¹³, conferido contra o Radau
  e pela quantidade conservada do sistema.

## Organização

| Caminho | O que é |
|---|---|
| `exercicio1.py` | Script oficial: roda tudo e gera tabelas, figuras e números |
| `problemas.py` | Ficha dos três problemas (regra, ponto de partida, intervalo, resposta certa) e a régua de erro |
| `rk.py` | RK45 do SciPy (referência). Sozinho: `python rk.py oscilante` |
| `pinn.py` | A PINN, implementada pela equipe em PyTorch. Sozinho: `python pinn.py oscilante` |
| `relatorio/` | Texto do exercício 1 em LaTeX (`exercicio1.tex` para o relatório do grupo; `exercicio1_avulso.tex` compila sozinho) |
| `apresentacao/` | Cópia dos slides e o script que gera os dados das animações. A versão oficial é a publicada (link abaixo) |
| `estudo/` | Exemplos didáticos em Python puro, usados para estudar o tema. Não fazem parte da entrega |

Slides: https://claude.ai/artifact/HbjDY4XbhzEbuaLB1LVZTo

## Gerar o PDF do relatório

Depois de rodar `exercicio1.py`, compile dentro de `relatorio/` (duas vezes, por causa das referências):

```bash
pdflatex exercicio1_avulso.tex
```

Ou envie as pastas `relatorio/` e `resultados/` para o Overleaf, mantendo as duas lado a lado.
