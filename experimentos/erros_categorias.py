"""
Categorizacao MANUAL dos 79 erros entre Positivo e Neutro no conjunto de teste.

Por que manual: nenhuma heuristica automatica distingue "exposicao didatica com lexico
afetivo" de "elogio", que e justamente a fronteira em questao. A atribuicao abaixo foi
feita por inspecao das 79 sentencas, e fica registrada aqui para ser audivel — cada
indice remete a posicao na lista de erros de erros_taxonomia.json (1-based, na ordem em
que o script os produz).

Reatribuida em 2026-08-15 sobre o corpus re-rotulado. A atribuicao anterior cobria os 72
erros da execucao antiga e nao pode ser reaproveitada: os indices apontam para outras
sentencas. Como toda categorizacao por inspecao, esta e um julgamento e nao uma medida —
convem que um segundo leitor a confira antes da submissao.

Categorias:
  ruido        a transcricao esta corrompida e a sentenca nao e interpretavel; erro de
               dado, nao de modelo
  exposicao    funcao informativa com lexico afetivo, ou descricao de acao cujo tom
               positivo vem do contexto do video e nao da sentenca
  interrogativa  pergunta de engajamento didatico, comum na fala dirigida a criancas
  formula      cantiga, refrao ou maxima prescritiva — texto de forma fixa, sem
               proposicao avaliativa propria

Uso: python erros_categorias.py   (Python global; le erros_taxonomia.json)
"""

from __future__ import annotations

import json
from collections import Counter

from caminhos import RESULTADOS, prepara_saidas

prepara_saidas()

CATEGORIAS = {
    "ruido": [17, 18, 19, 22, 28, 30, 33, 35, 37, 58],
    "exposicao": [
        1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 23, 24, 25, 26, 27,
        31, 34, 36, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 59, 60, 62, 63, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74,
    ],
    "interrogativa": [2, 6, 7],
    "formula": [29, 32, 38, 40, 61, 66, 75, 76, 77, 78, 79],
}

ROTULOS = {
    "ruido": "Ruído de transcrição",
    "exposicao": "Exposição didática com léxico afetivo",
    "interrogativa": "Interrogativa de engajamento",
    "formula": "Fórmula fixa (cantiga ou máxima)",
}


def main():
    d = json.loads((RESULTADOS / "erros_taxonomia.json").read_text(encoding="utf-8"))
    par = [
        e for e in d["erros"]
        if {e["verdadeiro"], e["predito"]} == {"Positivo", "Neutro"}
    ]

    atribuido = {}
    for cat, idxs in CATEGORIAS.items():
        for i in idxs:
            if i in atribuido:
                raise SystemExit(f"indice {i} atribuido duas vezes")
            atribuido[i] = cat
    faltando = set(range(1, len(par) + 1)) - set(atribuido)
    if faltando:
        raise SystemExit(f"sem categoria: {sorted(faltando)}")
    if len(atribuido) != len(par):
        raise SystemExit(f"{len(atribuido)} categorias para {len(par)} erros")

    saida = {"n": len(par), "categorias": {}, "itens": []}
    cont = Counter(atribuido.values())
    for cat in CATEGORIAS:
        saida["categorias"][cat] = {
            "rotulo": ROTULOS[cat],
            "n": cont[cat],
            "pct": cont[cat] / len(par),
        }
    for i, e in enumerate(par, 1):
        saida["itens"].append(
            {
                "i": i,
                "categoria": atribuido[i],
                "direcao": f"{e['verdadeiro']}->{e['predito']}",
                "n_tokens": e["n_tokens"],
                "confianca": e["confianca"],
                "texto": e["texto"],
            }
        )

    (RESULTADOS / "erros_categorias.json").write_text(
        json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(par)} erros Positivo<->Neutro categorizados")
    for cat, d_ in saida["categorias"].items():
        print(f"  {ROTULOS[cat]:42s} {d_['n']:2d}  ({d_['pct']:.1%})")
    print("\nsalvo em erros_categorias.json")


if __name__ == "__main__":
    main()
