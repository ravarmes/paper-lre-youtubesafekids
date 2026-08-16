"""Recalcula as estatisticas do corpus reportadas na Secao de metodologia.

Confere, direto de corpus.csv, o que o artigo afirma sobre os dados: tamanho do
corpus, numero de videos, distribuicao das classes, concordancia entre os quatro
anotadores (kappa de Fleiss), concordancia media par a par e proporcao de itens
com voto unanime.

O rotulo final (coluna AS) e o voto majoritario de P1/A1/A2/A3, com P1 (docente)
como arbitro em caso de empate; o script tambem verifica quantos itens seguem a
maioria simples, para que a regra de desempate fique auditavel.

Saidas: estatisticas_corpus.json
Uso: python estatisticas_corpus.py   (so a biblioteca padrao)
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from itertools import combinations

from caminhos import CORPUS, RESULTADOS, prepara_saidas

prepara_saidas()
LABELS = ["Negativo", "Neutro", "Positivo"]
ANOTADORES = ["P1", "A1", "A2", "A3"]


def carrega():
    with open(CORPUS, encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh, delimiter=";")]


def fleiss_kappa(matriz):
    """kappa de Fleiss sobre a matriz n_itens x n_categorias de contagens."""
    n_itens = len(matriz)
    n_avaliacoes = sum(matriz[0])

    # concordancia observada: proporcao de pares concordantes dentro de cada item
    p_i = [
        (sum(c * c for c in linha) - n_avaliacoes) / (n_avaliacoes * (n_avaliacoes - 1))
        for linha in matriz
    ]
    p_obs = sum(p_i) / n_itens

    # concordancia esperada: proporcao marginal de cada categoria, ao quadrado
    total = n_itens * n_avaliacoes
    p_cat = [sum(linha[j] for linha in matriz) / total for j in range(len(LABELS))]
    p_esp = sum(p * p for p in p_cat)

    return (p_obs - p_esp) / (1 - p_esp), p_obs, p_esp


def main():
    linhas = carrega()

    # so entram itens com os quatro anotadores preenchidos e rotulos validos
    completos = [
        r for r in linhas
        if all((r.get(a) or "").strip() in LABELS for a in ANOTADORES)
    ]

    matriz = []
    unanimes = 0
    concordancia_pares = []
    maioria_simples = 0
    for r in completos:
        votos = [r[a].strip() for a in ANOTADORES]
        matriz.append([votos.count(rot) for rot in LABELS])
        if len(set(votos)) == 1:
            unanimes += 1
        pares = list(combinations(votos, 2))
        concordancia_pares.append(sum(a == b for a, b in pares) / len(pares))

        contagem = Counter(votos)
        topo = max(contagem.values())
        vencedores = [rot for rot, c in contagem.items() if c == topo]
        if len(vencedores) == 1 and vencedores[0] == (r.get("AS") or "").strip():
            maioria_simples += 1

    kappa, p_obs, p_esp = fleiss_kappa(matriz)
    dist = Counter((r.get("AS") or "").strip() for r in linhas)
    videos = {(r.get("ID") or "").strip() for r in linhas}

    saida = {
        "sentencas": len(linhas),
        "videos": len(videos),
        "itens_com_quatro_anotadores": len(completos),
        "distribuicao": {
            rot: {"n": dist[rot], "pct": round(100 * dist[rot] / len(linhas), 1)}
            for rot in LABELS
        },
        "fleiss_kappa": round(kappa, 4),
        "concordancia_observada": round(p_obs, 4),
        "concordancia_esperada": round(p_esp, 4),
        "concordancia_media_pares_pct": round(
            100 * sum(concordancia_pares) / len(concordancia_pares), 1
        ),
        "unanimes": unanimes,
        "unanimes_pct": round(100 * unanimes / len(completos), 1),
        "rotulo_final_igual_a_maioria_simples": maioria_simples,
    }

    (RESULTADOS / "estatisticas_corpus.json").write_text(
        json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"sentencas ............. {saida['sentencas']}")
    print(f"videos ................ {saida['videos']}")
    for rot in LABELS:
        d = saida["distribuicao"][rot]
        print(f"  {rot:9s} ........... {d['n']} ({d['pct']}%)")
    print(f"kappa de Fleiss ....... {saida['fleiss_kappa']}")
    print(f"concordancia media .... {saida['concordancia_media_pares_pct']}%")
    print(f"votos unanimes ........ {saida['unanimes']} ({saida['unanimes_pct']}%)")
    print(f"\nsalvo em estatisticas_corpus.json")


if __name__ == "__main__":
    main()
