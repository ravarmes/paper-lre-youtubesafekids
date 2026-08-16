"""
Ablacao da funcao de pontuacao (Equacao 2 do artigo).

MOTIVACAO
---------
A Eq. (2) faz tres escolhas de projeto que precisam de justificativa empirica:
separar risco de tom em duas FAIXAS disjuntas; graduar a faixa segura pela
esperanca de adequacao A = P(Pos) + 0.5*P(Neu); e graduar a faixa negativa pela
confianca. Este script mede o que cada uma rende, reaproveitando o protocolo de
avaliar_reranking.py (264 videos, 32 consultas, predicao out-of-fold com TF-IDF +
regressao logistica).

NAO retreina o BERTimbau: a ablacao roda sobre o mesmo classificador leve da
Secao 6.3, em segundos. Todas as variantes sao aplicadas sobre as MESMAS predicoes
out-of-fold, de modo que qualquer diferenca observada vem da funcao de escore.

DESEMPATE ALEATORIO
-------------------
Ponto metodologico que decide a comparacao. Variantes que atribuem o mesmo valor a
muitos videos deixam a ordem entre eles a cargo do desempate, e neste corpus nao ha
ordem da plataforma para desempatar: gera_consultas() devolve os videos em ordem de
ID do YouTube, que e arbitraria em relacao a adequacao. Pontuar essas variantes numa
unica ordem de entrada credita-lhes um sorteio especifico — favoravel ou nao — em vez
do valor esperado. Por isso toda variante e avaliada como a media de N_DESEMPATES
sorteios da ordem de entrada, com o desvio-padrao registrado ao lado. Para variantes
sem empates o procedimento e inocuo (desvio zero), e a comparacao passa a ser justa
entre as que empatam e as que nao empatam.

VARIANTES
---------
  proposta        Eq. (2): 0.10+0.20*(1-P(Neg)) se Negativo ; 0.70+0.15*A caso contrario
  plato_constante faixa segura sem gradacao (0.85 fixo): isola o que A rende
  sem_confianca   faixa negativa sem gradacao (0.20 fixo): isola o que a confianca rende
  neutro_interm   Neutro como risco intermediario: 0.5 ; Positivo 0.7+0.3C
  esperanca       so a esperanca A, sem as faixas: isola o que o portao de risco rende

Saidas: ablacao_score.json, tab-ablacao.tex (em ../latex no monorepo, em saidas/ fora dele)
Uso: python ablacao_score.py   (Python global; usa apenas numpy)
"""

from __future__ import annotations

import json
import random

import numpy as np

import avaliar_reranking as ar

from caminhos import LATEX, RESULTADOS, prepara_saidas

prepara_saidas()
AVISO = "% GERADO POR src/experimentos/ablacao_score.py — nao editar a mao.\n"

N_DESEMPATES = 200


# ---------------------------------------------------------------- variantes


def _A(P):
    """Esperanca de adequacao (Eq. 3)."""
    return float(P[2] + 0.5 * P[1])


def s_proposta(P):
    """Eq. (2) como proposta: faixas disjuntas, tom graduando a faixa segura."""
    if int(P.argmax()) == 0:
        return 0.10 + 0.20 * (1.0 - float(P[0]))
    return 0.70 + 0.15 * _A(P)


def s_plato_constante(P):
    """Faixa segura sem gradacao: mede o que a ordenacao por tom rende DENTRO da
    faixa em que nao ha risco a corrigir. Empata todo conteudo nao-Negativo."""
    if int(P.argmax()) == 0:
        return 0.10 + 0.20 * (1.0 - float(P[0]))
    return 0.85


def s_sem_confianca(P):
    """Faixa negativa sem gradacao: isola o que a confianca contribui."""
    if int(P.argmax()) == 0:
        return 0.20
    return 0.70 + 0.15 * _A(P)


def s_neutro_intermediario(P):
    """Alternativa de projeto: Neutro posicionado ENTRE Negativo e Positivo, como se
    a ausencia de carga afetiva fosse risco intermediario. E a variante que a Eq. (2)
    rejeita, e esta aqui para medir o custo dessa rejeicao."""
    c = int(P.argmax())
    conf = float(P[c])
    if c == 0:
        return 0.10 + 0.20 * (1.0 - conf)
    return 0.5 if c == 1 else 0.7 + 0.3 * conf


def s_esperanca(P):
    """So a esperanca, sem as faixas: mede o que o portao de risco rende. Aqui a
    gradacao por tom pode competir com a decisao de risco, que e exatamente o que
    as faixas disjuntas da Eq. (2) impedem."""
    return _A(P)


VARIANTES = {
    "proposta": ("Equação~\\ref{eq:score}, como proposta", s_proposta),
    "plato_constante": ("Faixa segura sem gradação por tom", s_plato_constante),
    "sem_confianca": ("Faixa negativa sem o termo de confiança", s_sem_confianca),
    "neutro_intermediario": ("Neutro como risco intermediário", s_neutro_intermediario),
    "esperanca": ("Só a esperança, sem as faixas", s_esperanca),
}

METRICAS = ("ndcg@3", "ndcg@10", "p@3", "map")


# ---------------------------------------------------------------- execucao


def avalia(escores, consultas, videos, rnd):
    """Media sobre N_DESEMPATES sorteios da ordem de entrada.

    Retorna (media, desvio) por metrica. Variantes sem empates dao desvio zero.
    """
    amostras = {m: [] for m in METRICAS}
    for _ in range(N_DESEMPATES):
        acc = []
        for vids in consultas.values():
            emb = list(vids)
            rnd.shuffle(emb)
            acc.append(ar.avalia_ordenacao(sorted(emb, key=lambda v: -escores[v]), videos))
        for m in METRICAS:
            amostras[m].append(float(np.mean([a[m] for a in acc])))
    return ({m: float(np.mean(amostras[m])) for m in METRICAS},
            {m: float(np.std(amostras[m])) for m in METRICAS})


def main():
    random.seed(ar.SEED)
    rng = np.random.default_rng(ar.SEED)

    videos = ar.carrega_videos()
    ids = sorted(videos)
    rng.shuffle(ids)
    folds = [ids[i :: ar.N_FOLDS] for i in range(ar.N_FOLDS)]

    # Uma unica passada out-of-fold guarda a distribuicao de probabilidade de cada
    # video; todas as variantes sao entao aplicadas sobre as MESMAS predicoes.
    probs = {}
    for ids_teste in folds:
        ids_treino = [v for v in ids if v not in set(ids_teste)]
        X_txt, y = [], []
        for vid in ids_treino:
            X_txt.extend(videos[vid]["frases"])
            y.extend(videos[vid]["rotulos"])
        tfidf = ar.TfIdf().fit(X_txt)
        clf = ar.RegressaoLogistica().fit(tfidf.transform(X_txt), np.array(y))
        for vid in ids_teste:
            v = videos[vid]
            texto = v["titulo"] + " " + " ".join(v["frases"])
            probs[vid] = clf.predict_proba(tfidf.transform([texto]))[0]

    consultas = ar.gera_consultas(videos)
    print(f"{len(videos)} videos, {len(consultas)} consultas, "
          f"{N_DESEMPATES} sorteios de desempate por variante")

    resultados = {}
    for chave, (rotulo, fn) in VARIANTES.items():
        escores = {vid: fn(P) for vid, P in probs.items()}
        rnd = random.Random(12345)  # mesma sequencia de sorteios para toda variante
        media, desvio = avalia(escores, consultas, videos, rnd)
        n_dist = len(set(round(v, 6) for v in escores.values()))
        resultados[chave] = {
            "rotulo": rotulo,
            "metricas": media,
            "desvio_desempate": desvio,
            "n_valores_distintos": n_dist,
            "n_videos": len(escores),
        }

    (RESULTADOS / "ablacao_score.json").write_text(
        json.dumps({"n_desempates": N_DESEMPATES, "variantes": resultados},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------- tabela
    def num(x, c=3):
        return f"{x:.{c}f}".replace(".", ",")

    linhas = []
    for chave, d in resultados.items():
        m, s = d["metricas"], d["desvio_desempate"]
        nome = f"\\textbf{{{d['rotulo']}}}" if chave == "proposta" else d["rotulo"]
        cels = " & ".join(
            f"{num(m[k])}" if s[k] < 5e-4 else f"{num(m[k])}\\,$\\pm$\\,{num(s[k])}"
            for k in METRICAS
        )
        linhas.append(f"{nome} & {cels} & {d['n_valores_distintos']}/{d['n_videos']} \\\\")
    corpo = "\n".join(linhas)
    tabela = f"""{AVISO}\\begin{{table}}[!t]
\\caption{{Ablação da função de pontuação sobre as mesmas predições \\emph{{out-of-fold}} e as mesmas {len(consultas)} consultas da Seção~\\ref{{sec:reranking}}. Cada valor é a média de {N_DESEMPATES} sorteios da ordem de entrada, com o desvio-padrão quando a variante empata vídeos; sem o sorteio, as variantes que empatam seriam creditadas pela ordem arbitrária de ID do vídeo. A última coluna dá quantos valores distintos de escore a variante produz sobre os {len(probs)} vídeos.}}\\label{{tab:ablacao}}
\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}LRRRRR@{{}}}}
\\toprule
Função de pontuação & nDCG@3 & nDCG@10 & P@3 & MAP & Valores \\\\
\\midrule
{corpo}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    (LATEX / "tab-ablacao.tex").write_text(tabela, encoding="utf-8")

    print("\n=== ablacao (media +- dp sobre sorteios de desempate) ===")
    for chave, d in resultados.items():
        m, s = d["metricas"], d["desvio_desempate"]
        print(f"  {chave:22s} " + "  ".join(
            f"{k}={m[k]:.3f}+-{s[k]:.3f}" for k in METRICAS
        ) + f" | valores distintos: {d['n_valores_distintos']}")
    print(f"\nsalvo em ablacao_score.json e {LATEX / 'tab-ablacao.tex'}")


if __name__ == "__main__":
    main()
