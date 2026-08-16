"""
Gera as tabelas LaTeX e a figura do estudo de caso de reordenacao real.

Le estudo_caso_ordenacao.json (produzido por estudo_caso_ordenacao.py) e escreve, em
../latex quando roda dentro do monorepo do artigo e em saidas/ fora dele:
  tab-estudo-caso.tex          distribuicao do Score por consulta e por estrato
  tab-ordem.tex                mudanca de ordem em relacao ao YouTube
  figuras/fig06-case-study-scores.png
  figuras/fig07-case-study-order.png

Roda com o Python GLOBAL (precisa apenas de numpy e matplotlib), nao com o venv de
.venv, que tem torch e transformers mas nao matplotlib. E o mesmo arranjo de
avaliar_reranking.py.

Uso: python gerar_tabelas_estudo_caso.py
"""

from __future__ import annotations

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from caminhos import FIGURAS, LATEX, RESULTADOS, prepara_saidas

prepara_saidas()
DADOS = RESULTADOS / "estudo_caso_ordenacao.json"

AVISO = "% GERADO POR src/experimentos/gerar_tabelas_estudo_caso.py — nao editar a mao.\n"


def num(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def tabela_estratos(d: dict) -> str:
    por_termo = d["por_termo"]
    e = d["separacao_estratos"]["score_bloco"]
    linhas = []
    for termo, v in sorted(por_termo.items(), key=lambda kv: (kv[1]["estrato"], -kv[1]["media"])):
        est = "A" if v["estrato"].startswith("A") else "B"
        linhas.append(
            f"{est} & \\emph{{{termo}}} & {v['n']} & {num(v['media'])} & "
            f"{num(v['min'])} & {num(v['max'])} & {num(v['amplitude'])} \\\\"
        )
    corpo = "\n".join(linhas)
    return f"""{AVISO}\\begin{{table}}[!t]
\\caption{{Distribuição do \\emph{{Score de Segurança}} nas quatro consultas do estudo de caso, dez vídeos cada. No estrato~A o comportamento desejável é que os escores sejam altos e próximos entre si, pois não há risco a corrigir; no estrato~B, que sejam baixos. As consultas são reproduzidas literalmente em português por serem entradas do sistema.}}\\label{{tab:estudo-caso}}
\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}LLRRRRR@{{}}}}
\\toprule
Estrato & Consulta & $n$ & Média & Mínimo & Máximo & Amplitude \\\\
\\midrule
{corpo}
\\midrule
\\textbf{{A}} & \\emph{{agregado}} & 20 & \\textbf{{{num(e['media_A'])}}} & {num(e['min_A'])} & {num(e['max_A'])} & {num(e['max_A'] - e['min_A'])} \\\\
\\textbf{{B}} & \\emph{{agregado}} & 20 & \\textbf{{{num(e['media_B'])}}} & {num(e['min_B'])} & {num(e['max_B'])} & {num(e['max_B'] - e['min_B'])} \\\\
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""


def tabela_ordem(d: dict) -> str:
    linhas = []
    for termo, v in sorted(d["mudanca_de_ordem"].items(), key=lambda kv: kv[1]["estrato"]):
        est = "A" if v["estrato"].startswith("A") else "B"
        linhas.append(
            f"{est} & \\emph{{{termo}}} & {num(v['kendall_tau'])} & "
            f"{num(v['top3_sobreposicao'], 2)} & {num(v['deslocamento_medio'], 1)} & "
            f"{v['primeiro_do_youtube']['para']}\\textsuperscript{{o}} & "
            f"{num(v['amplitude_score'])} \\\\"
        )
    corpo = "\n".join(linhas)
    return f"""{AVISO}\\begin{{table}}[!t]
\\caption{{Efeito da reordenação sobre a ordem devolvida pela plataforma. $\\tau$ é o coeficiente de Kendall entre as duas ordenações: $1$ indica ordem idêntica à do YouTube e valores próximos de zero, ordem sem relação com ela. ``Top-3'' é a fração dos três primeiros da plataforma que permanece entre os três primeiros; ``Deslocamento'' é a média das mudanças de posição em valor absoluto; ``1\\textsuperscript{{o}} do YouTube'' é a posição para a qual o vídeo mais bem colocado pela plataforma foi movido. A última coluna traz a amplitude dos escores dentro da consulta, que é o que dá ou retira sentido à reordenação.}}\\label{{tab:ordem}}
\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}LLRRRRR@{{}}}}
\\toprule
Estrato & Consulta & $\\tau$ & Top-3 & Deslocamento & 1\\textsuperscript{{o}} do YouTube & Amplitude \\\\
\\midrule
{corpo}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""


def figura(d: dict) -> None:
    """Duas figuras separadas, e nao dois paineis de uma so.

    Uma figura larga com os dois paineis nao cabe em nenhuma pagina junto ao texto
    desta secao e acaba sozinha numa pagina de floats, longe do que a discute. Em
    coluna unica, dois floats menores tambem ficam mais legiveis.
    """
    videos = d["videos"]
    termos = list(dict.fromkeys(v["termo"] for v in videos))
    cores = {"A": "#2b7bba", "B": "#c1443c"}

    fig1, ax1 = plt.subplots(figsize=(7.2, 3.6))

    # As duas faixas da Eq. (2) sombreadas ao fundo. Sao o argumento da formulacao
    # em forma visual: a distancia entre elas e tres vezes a variacao dentro de
    # cada uma, de modo que a gradacao por tom nunca cruza a fronteira de risco.
    ax1.axhspan(0.70, 0.85, color="#2b7bba", alpha=0.09, zorder=0)
    ax1.axhspan(0.10, 0.2333, color="#c1443c", alpha=0.09, zorder=0)
    ax1.text(len(termos) - 0.42, 0.775, "faixa sem risco", fontsize=6.5,
             color="#2b7bba", ha="right", va="center", zorder=1)
    ax1.text(len(termos) - 0.42, 0.167, "faixa de risco", fontsize=6.5,
             color="#c1443c", ha="right", va="center", zorder=1)

    # --- primeira figura: score de cada video, agrupado por consulta
    for i, termo in enumerate(termos):
        vs = [v for v in videos if v["termo"] == termo]
        est = "A" if vs[0]["estrato"].startswith("A") else "B"
        y = [v["score_bloco"] for v in vs]
        x = np.random.default_rng(7 + i).normal(i, 0.07, len(y))
        ax1.scatter(x, y, s=42, alpha=0.8, color=cores[est], edgecolor="white", linewidth=0.6, zorder=3)
        ax1.plot([i - 0.28, i + 0.28], [np.mean(y)] * 2, color="#333", lw=2, zorder=4)

    ax1.set_xticks(range(len(termos)))
    ax1.set_xticklabels([textwrap.fill(t, 15) for t in termos], fontsize=7.5)
    ax1.set_ylabel("Score de Segurança")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.5, color="#999", ls=":", lw=1)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_title("Escore por consulta", fontsize=10)
    for est, rot in (("A", "Estrato A (responsável)"), ("B", "Estrato B (apelo infantil)")):
        ax1.scatter([], [], color=cores[est], label=rot, s=42)
    ax1.legend(fontsize=8, loc="lower left", framealpha=0.9)

    fig1.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig1.savefig(FIGURAS / "fig06-case-study-scores.png", dpi=300)

    # --- segunda figura: ordem do YouTube contra ordem por Score, um par de colunas
    # por consulta. Sobrepor as quatro consultas num unico par torna o grafico ilegivel.
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.0))
    cor_classe = {"Negativo": "#c1443c", "Neutro": "#9a9a9a", "Positivo": "#2b7bba"}
    largura, folga = 0.72, 1.55
    for i, termo in enumerate(termos):
        x0 = i * folga
        x1 = x0 + largura
        for v in [v for v in videos if v["termo"] == termo]:
            d_ = next(
                x
                for x in d["deslocamentos"]
                if x["titulo"] == v["titulo"] and x["termo"] == v["termo"]
            )
            ax2.plot(
                [x0, x1],
                [d_["de"], d_["para"]],
                color=cor_classe[v["classe"]],
                alpha=0.75,
                lw=1.3,
                marker="o",
                ms=3.0,
                zorder=3,
            )
        # O rotulo do estrato vai ACIMA da moldura, nao dentro dela: em coordenadas de
        # dados ele encostava na borda superior, porque o eixo y esta invertido e sobra
        # pouco espaco entre a posicao 1 e o limite. A transformacao mista fixa o x nos
        # dados (para alinhar com o par de colunas da consulta) e o y na fracao dos
        # eixos. O `pad` do titulo abre a faixa em que esses rotulos ficam.
        est = "A" if [v for v in videos if v["termo"] == termo][0]["estrato"].startswith("A") else "B"
        ax2.text(
            x0 + largura / 2,
            1.02,
            est,
            transform=blended_transform_factory(ax2.transData, ax2.transAxes),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=cores[est],
        )

    ax2.set_xlim(-0.35, (len(termos) - 1) * folga + largura + 0.35)
    ax2.set_xticks([i * folga + largura / 2 for i in range(len(termos))])
    ax2.set_xticklabels([textwrap.fill(t, 15) for t in termos], fontsize=7.5)
    ax2.set_ylabel("Posição")
    ax2.invert_yaxis()
    ax2.set_yticks(range(1, 11))
    ax2.set_ylim(10.6, -0.6)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_title(
        "Ordem do YouTube (esq.) $\\rightarrow$ ordem por Score (dir.)", fontsize=10, pad=18
    )
    # legenda das classes como rodape: dentro dos eixos ela cobre as linhas
    manipuladores = [
        plt.Line2D([], [], color=cor, lw=2, label=cl) for cl, cor in cor_classe.items()
    ]
    fig2.legend(
        handles=manipuladores,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        title="Classe predita para o vídeo",
        title_fontsize=8,
    )

    fig2.tight_layout(rect=(0, 0.10, 1, 1))
    fig2.savefig(FIGURAS / "fig07-case-study-order.png", dpi=300)
    print("figuras salvas em", FIGURAS)


def main():
    d = json.loads(DADOS.read_text(encoding="utf-8"))
    (LATEX / "tab-estudo-caso.tex").write_text(tabela_estratos(d), encoding="utf-8")
    (LATEX / "tab-ordem.tex").write_text(tabela_ordem(d), encoding="utf-8")
    print("tabelas salvas em", LATEX)
    figura(d)


if __name__ == "__main__":
    main()
