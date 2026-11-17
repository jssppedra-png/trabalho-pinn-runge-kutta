# Apresentação do exercício 1

Cópia de segurança dos slides publicados em https://claude.ai/artifact/HbjDY4XbhzEbuaLB1LVZTo
(tipo "Slides" do Claude; o link é privado até ser compartilhado pelo menu Share da página).

- `project/deck.json`: índice (título, ordem dos 15 slides, seções, fontes).
- `project/deck.css`: estilo comum a todos os slides (cores, letras, cartões, animações de entrada).
- `project/slides/*.html`: um arquivo por slide. As animações vivas ficam dentro de `<x-embed>`,
  e o roteiro de fala fica no `<aside>` no fim de cada arquivo.
- `dados_animacoes.py`: refaz os treinos com a mesma semente e ordem de `exercicio1.py` e salva as
  "fotos" usadas nas animações. Os erros batem com `resultados/`.

Se alguém editar os slides pela página, esta cópia fica desatualizada: a versão que vale é a do link.
