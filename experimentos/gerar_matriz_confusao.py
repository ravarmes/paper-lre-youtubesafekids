"""
Gera latex/figuras/fig02-confusion-matrix.png a partir das predicoes reais do ensemble.

A matriz sai de erros_taxonomia.json, produzido por significancia_e_erros.py sobre
predicoes_teste.npz — as MESMAS predicoes que alimentam as Tabelas 4 e 5, o teste de
significancia e a taxonomia dos erros. Antes, esta figura vinha de valores digitados a
mao e nao correspondia a nenhuma execucao do artigo.

Roda com o Python GLOBAL (precisa apenas de numpy e matplotlib), nao com o venv de
.venv, que tem torch e transformers mas nao matplotlib — mesmo arranjo de
gerar_tabelas_estudo_caso.py.

Uso: python gerar_matriz_confusao.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from caminhos import FIGURAS, RESULTADOS, prepara_saidas

prepara_saidas()
TAXONOMIA = RESULTADOS / "erros_taxonomia.json"

FONT_SIZE = 18  # numeros dentro da matriz
LABEL_SIZE = 16  # rotulos dos eixos
TICK_SIZE = 15  # nomes das classes


def main():
    if not TAXONOMIA.exists():
        raise SystemExit(
            f"{TAXONOMIA.name} nao encontrado. Rode significancia_e_erros.py antes."
        )
    dados = json.loads(TAXONOMIA.read_text(encoding="utf-8"))
    cm = np.array(dados["matriz_confusao"])
    labels = dados["labels"]
    n = int(cm.sum())

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = cm.max()
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)

    for i in range(3):
        for j in range(3):
            val = int(cm[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=FONT_SIZE,
                color="white" if val > vmax * 0.55 else "black",
                fontweight="bold",
            )

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.set_yticklabels(labels, fontsize=TICK_SIZE)
    ax.set_xlabel("Classe predita", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel("Classe verdadeira", fontsize=LABEL_SIZE, labelpad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 2)

    plt.tight_layout()
    saida = FIGURAS / "fig02-confusion-matrix.png"
    plt.savefig(saida, dpi=300, bbox_inches="tight")
    acertos = int(np.trace(cm))
    print(f"Salvo: {saida}  ({n} sentencas, {acertos} acertos, {n - acertos} erros)")


if __name__ == "__main__":
    main()
