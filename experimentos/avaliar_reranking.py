"""
Avaliacao quantitativa da REORDENACAO (re-ranking) por Score de Seguranca.

Motivacao: o trabalho propoe reordenar resultados de busca, mas ate aqui a
reordenacao era apenas ilustrada por capturas de tela. Este script mede se
ordenar por Score de Seguranca coloca os videos adequados no topo, com metricas
de recuperacao da informacao (nDCG@k, P@k, MAP).

PROTOCOLO
---------
Unidade de recuperacao: o VIDEO (o corpus tem 176 videos distintos, cada um com
~15 sentencas anotadas em Inicio/Meio/Fim).

Ground truth de adequacao: derivado dos rotulos humanos das sentencas do video.
  adequacao(v) = (n_positivas + 0.5 * n_neutras) / n_sentencas   em [0, 1]
Ganho graduado para o nDCG (escala 0-2, padrao em RI):
  rel(v) = 0 se adequacao < 0.34 ; 1 se < 0.67 ; 2 caso contrario
Relevancia binaria para P@k e MAP: rel(v) >= 1 (video nao-inadequado).

Consultas: derivadas dos proprios titulos do corpus. Um termo vira consulta se
aparece no titulo de >= MIN_RESULTADOS videos E o conjunto retornado tem
adequacao variada (senao nao ha o que reordenar).

Sistemas comparados:
  1. aleatorio  - media de N_PERMUT permutacoes (piso: ordenar sem informacao)
  2. score      - ordena por Score de Seguranca das predicoes do classificador
  3. oraculo    - ordena pelo Score calculado sobre os rotulos HUMANOS (teto)

Classificador: TF-IDF + regressao logistica multinomial, implementados em numpy
(o ambiente nao tem scikit-learn funcional). Split por VIDEO, nao por sentenca,
para evitar vazamento entre treino e teste.

LIMITACAO DECLARADA
-------------------
Nao ha, neste corpus, a ordem em que o YouTube devolveu os resultados. Portanto
esta avaliacao mede o ganho da reordenacao sobre uma ordenacao NAO INFORMADA
(aleatoria), e nao sobre a ordenacao real da plataforma. Medir o segundo exige
coletar, para cada consulta, a lista ordenada devolvida pela API.

Saidas: reranking_results.json, reranking_curvas.png, reranking_tabela.tex
Uso: python avaliar_reranking.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from caminhos import CORPUS, RESULTADOS, prepara_saidas

prepara_saidas()
SEED = 42
LABELS = ["Negativo", "Neutro", "Positivo"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}

MIN_RESULTADOS = 5      # minimo de videos para um termo virar consulta
MIN_DESVIO = 0.12       # desvio-padrao minimo da adequacao (garante conjunto misto)
N_PERMUT = 200          # permutacoes para o baseline aleatorio
KS = [3, 5, 10]         # cortes das metricas @k
N_FOLDS = 5             # validacao cruzada POR VIDEO (predicao out-of-fold)

SISTEMAS = ["aleatorio", "bloco", "sentenca", "oraculo"]
ROTULOS = {
    "aleatorio": "Ordenação aleatória",
    "bloco": "Score, bloco único (Eq. 1)",
    "sentenca": "Score, média por sentença",
    "oraculo": "Oráculo (rótulos humanos)",
}


# --------------------------------------------------------------- corpus

def normaliza(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    return txt.lower()


def carrega_videos() -> dict:
    """Agrupa as sentencas por video e calcula o ground truth de adequacao."""
    with open(CORPUS, encoding="utf-8") as fh:
        linhas = list(csv.DictReader(fh, delimiter=";"))

    videos = defaultdict(lambda: {"titulo": "", "frases": [], "rotulos": []})
    for r in linhas:
        vid, rot = (r.get("ID") or "").strip(), (r.get("AS") or "").strip()
        frase = (r.get("FRASE") or "").strip().strip('"').strip()
        if not vid or rot not in LABEL_TO_ID or not frase:
            continue
        videos[vid]["titulo"] = (r.get("TITULO") or "").strip()
        videos[vid]["frases"].append(frase)
        videos[vid]["rotulos"].append(LABEL_TO_ID[rot])

    for vid, v in videos.items():
        n = len(v["rotulos"])
        c = Counter(v["rotulos"])
        # adequacao: positivo vale 1, neutro 0.5, negativo 0
        v["adequacao"] = (c[2] + 0.5 * c[1]) / n
        v["rel"] = 0 if v["adequacao"] < 0.34 else (1 if v["adequacao"] < 0.67 else 2)
        v["n_frases"] = n
    return dict(videos)


# ------------------------------------------------- consultas a partir dos titulos

STOPWORDS = set("""de da do das dos e em no na nos nas o a os as um uma para com por
que se ao aos sem sob sobre completo completa episodio episodios temporada video
videos dublado dublada portugues brasil pt br em""".split())


def gera_consultas(videos: dict) -> dict:
    """Um termo dos titulos vira consulta se retorna videos suficientes e variados."""
    termo_para_videos = defaultdict(set)
    for vid, v in videos.items():
        tokens = re.findall(r"[a-z0-9]+", normaliza(v["titulo"]))
        for t in set(tokens):
            if len(t) >= 4 and t not in STOPWORDS:
                termo_para_videos[t].add(vid)

    consultas = {}
    for termo, vids in termo_para_videos.items():
        if len(vids) < MIN_RESULTADOS:
            continue
        adeq = [videos[v]["adequacao"] for v in vids]
        if float(np.std(adeq)) < MIN_DESVIO:
            continue  # conjunto homogeneo: nao ha reordenacao a avaliar
        consultas[termo] = sorted(vids)
    return consultas


# ------------------------------------------------- TF-IDF + regressao logistica (numpy)

def tokeniza(txt: str) -> list:
    return re.findall(r"[a-z0-9_]+", normaliza(txt))


class TfIdf:
    def __init__(self, min_df=2, max_features=8000):
        self.min_df, self.max_features, self.vocab, self.idf = min_df, max_features, {}, None

    def fit(self, docs):
        df = Counter()
        for d in docs:
            df.update(set(tokeniza(d)))
        termos = [t for t, c in df.most_common() if c >= self.min_df][: self.max_features]
        self.vocab = {t: i for i, t in enumerate(termos)}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in termos])
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for t, c in Counter(tokeniza(d)).items():
                j = self.vocab.get(t)
                if j is not None:
                    X[i, j] = c
        X *= self.idf
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm == 0, 1, norm)


class RegressaoLogistica:
    """Multinomial (softmax) com gradiente descendente e regularizacao L2."""

    def __init__(self, n_classes=3, lr=1.0, epocas=400, l2=1e-4, seed=SEED):
        self.n_classes, self.lr, self.epocas, self.l2 = n_classes, lr, epocas, l2
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        n, d = X.shape
        self.W = np.zeros((d, self.n_classes))
        self.b = np.zeros(self.n_classes)
        Y = np.eye(self.n_classes)[y]
        # pesos por classe: compensa o desbalanceamento do corpus
        cont = np.bincount(y, minlength=self.n_classes).astype(float)
        peso_cls = (n / (self.n_classes * np.maximum(cont, 1)))
        w = peso_cls[y][:, None]
        for _ in range(self.epocas):
            P = self._softmax(X @ self.W + self.b)
            G = (P - Y) * w
            self.W -= self.lr * (X.T @ G / n + self.l2 * self.W)
            self.b -= self.lr * G.mean(axis=0)
        return self

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)


# --------------------------------------------------------------- Score de Seguranca

def score_seguranca(P) -> float:
    """Equacao (2) do artigo: RISCO DOMINA TOM.

    Duas faixas disjuntas, sobre a distribuicao de probabilidade completa:

      Negativo      0.10 + 0.20*(1 - P(Neg))   -> [0.10, 0.233]
      nao-Negativo  0.70 + 0.15*A              -> [0.70, 0.85]

    com A = P(Positivo) + 0.5*P(Neutro), a esperanca de adequacao (Eq. 3).

    A separacao entre as faixas (0.467) e tres vezes a variacao dentro da faixa
    segura (0.15): nenhum item sem risco cai abaixo de um item com risco, por
    construcao e nao por calibracao. Dentro da faixa segura, A ordena por tom —
    entre dois videos que nao apresentam risco, o de carga afetiva positiva fica
    a frente do meramente informativo, sem que essa gradacao possa competir com
    a decisao de risco. O teto e 0.85 e nao 1.00: nao rebaixar nao e atestar
    adequacao, porque as demais dimensoes nao foram verificadas.
    """
    P = np.asarray(P, dtype=float)
    if int(P.argmax()) == 0:                    # Negativo: gradua pela confianca
        return 0.10 + 0.20 * (1.0 - float(P[0]))
    return 0.70 + 0.15 * float(P[2] + 0.5 * P[1])


# --------------------------------------------------------------- metricas de RI

def dcg(ganhos) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(ganhos))


def ndcg_at_k(rels_ordenados, k) -> float:
    ideal = sorted(rels_ordenados, reverse=True)
    d, i = dcg(rels_ordenados[:k]), dcg(ideal[:k])
    return d / i if i > 0 else 0.0


def precision_at_k(rels_ordenados, k) -> float:
    topo = rels_ordenados[:k]
    return sum(1 for r in topo if r >= 1) / len(topo) if topo else 0.0


def average_precision(rels_ordenados) -> float:
    acertos, soma = 0, 0.0
    for i, r in enumerate(rels_ordenados):
        if r >= 1:
            acertos += 1
            soma += acertos / (i + 1)
    total = sum(1 for r in rels_ordenados if r >= 1)
    return soma / total if total else 0.0


def avalia_ordenacao(vids_ordenados, videos) -> dict:
    rels = [videos[v]["rel"] for v in vids_ordenados]
    m = {f"ndcg@{k}": ndcg_at_k(rels, k) for k in KS}
    m.update({f"p@{k}": precision_at_k(rels, k) for k in KS})
    m["map"] = average_precision(rels)
    return m


# --------------------------------------------------------------- execucao

def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    videos = carrega_videos()
    print(f"Videos no corpus: {len(videos)}")
    print(f"Sentencas: {sum(v['n_frases'] for v in videos.values())}")
    dist_rel = Counter(v["rel"] for v in videos.values())
    print(f"Adequacao por video (0=inadequado,1=intermediario,2=adequado): {dict(sorted(dist_rel.items()))}")

    # --- predicao OUT-OF-FOLD por video ---
    # Cada video e pontuado por um modelo que nunca viu nenhuma sentenca dele.
    # Assim todos os 176 videos ficam disponiveis para as consultas sem vazamento.
    ids = sorted(videos)
    rng.shuffle(ids)
    folds = [ids[i::N_FOLDS] for i in range(N_FOLDS)]

    # Duas variantes de agregacao das sentencas para o Score do video:
    #   bloco     - Eq. (1) do artigo: concatena tudo num unico texto e classifica 1x
    #   sentenca  - classifica cada sentenca e tira a MEDIA dos Scores individuais
    # A comparacao entre as duas testa uma decisao de projeto do artigo.
    score_bloco, score_sentenca, score_oraculo = {}, {}, {}
    for f, ids_teste in enumerate(folds):
        ids_treino = [v for v in ids if v not in set(ids_teste)]
        X_txt, y = [], []
        for vid in ids_treino:
            X_txt.extend(videos[vid]["frases"])
            y.extend(videos[vid]["rotulos"])
        tfidf = TfIdf().fit(X_txt)
        clf = RegressaoLogistica().fit(tfidf.transform(X_txt), np.array(y))
        for vid in ids_teste:
            v = videos[vid]
            # (a) bloco unico, como na Eq. (1)
            texto = v["titulo"] + " " + " ".join(v["frases"])
            P = clf.predict_proba(tfidf.transform([texto]))[0]
            score_bloco[vid] = score_seguranca(P)
            # (b) media dos Scores por sentenca
            Ps = clf.predict_proba(tfidf.transform(v["frases"]))
            scores = [score_seguranca(p) for p in Ps]
            score_sentenca[vid] = float(np.mean(scores))
        print(f"  fold {f + 1}/{N_FOLDS}: treino {len(ids_treino)} videos "
              f"({len(X_txt)} sentencas) -> pontuados {len(ids_teste)} videos")

    # oraculo: mesma equacao do Score, mas sobre os rotulos HUMANOS. A distribuicao
    # empirica das classes anotadas entra no lugar da predita, de modo que o oraculo
    # e o teto da FORMULACAO — o que a Eq. (2) renderia se acertasse toda classe.
    for vid, v in videos.items():
        c = Counter(v["rotulos"])
        n = len(v["rotulos"])
        score_oraculo[vid] = score_seguranca([c[0] / n, c[1] / n, c[2] / n])

    # --- consultas ---
    consultas = gera_consultas(videos)
    print(f"Consultas avaliadas (>= {MIN_RESULTADOS} videos, adequacao variada): {len(consultas)}")
    for q, vs in sorted(consultas.items()):
        print(f"  '{q}': {len(vs)} videos")
    if not consultas:
        print("\nERRO: nenhuma consulta atende aos criterios. Ajuste MIN_RESULTADOS/TEST_FRAC.")
        return 1

    # --- avaliacao ---
    acumulado = defaultdict(lambda: defaultdict(list))
    for q, vids in consultas.items():
        # 1. aleatorio (media de N_PERMUT permutacoes)
        parcial = defaultdict(list)
        for _ in range(N_PERMUT):
            perm = list(vids)
            random.shuffle(perm)
            for k, val in avalia_ordenacao(perm, videos).items():
                parcial[k].append(val)
        for k, vals in parcial.items():
            acumulado["aleatorio"][k].append(float(np.mean(vals)))

        # 2. Score de Seguranca (predito) e 3. oraculo (rotulos humanos)
        #
        # O desempate tambem e sorteado. Este corpus nao registra a ordem em que a
        # plataforma devolveu os resultados, e gera_consultas() entrega os videos em
        # ordem de ID do YouTube — arbitraria em relacao a adequacao. Uma ordenacao
        # estavel sobre essa ordem creditaria ao sistema um sorteio especifico em vez
        # do valor esperado, e o vies seria tanto maior quanto mais a funcao empatasse.
        # Sortear a ordem de entrada torna a comparacao justa entre sistemas que
        # empatam em graus diferentes (o oraculo empata mais, por usar contagens de
        # rotulos). Para uma funcao sem empates o procedimento e inocuo.
        for nome, tabela in (("bloco", score_bloco), ("sentenca", score_sentenca),
                             ("oraculo", score_oraculo)):
            parcial = defaultdict(list)
            for _ in range(N_PERMUT):
                perm = list(vids)
                random.shuffle(perm)
                ordem = sorted(perm, key=lambda v: -tabela[v])
                for k, val in avalia_ordenacao(ordem, videos).items():
                    parcial[k].append(val)
            for k, vals in parcial.items():
                acumulado[nome][k].append(float(np.mean(vals)))

    resultados = {sis: {k: float(np.mean(v)) for k, v in met.items()}
                  for sis, met in acumulado.items()}

    print("\n=== Resultados (media sobre as consultas) ===")
    cab = ["sistema"] + [f"nDCG@{k}" for k in KS] + [f"P@{k}" for k in KS] + ["MAP"]
    print("  ".join(c.ljust(9) for c in cab))
    for sis in SISTEMAS:
        r = resultados[sis]
        vals = [r[f"ndcg@{k}"] for k in KS] + [r[f"p@{k}"] for k in KS] + [r["map"]]
        print(sis.ljust(9) + "  " + "  ".join(f"{v:.4f}".ljust(9) for v in vals))

    saida = {
        "protocolo": {
            "n_videos": len(videos),
            "n_sentencas": sum(v["n_frases"] for v in videos.values()),
            "n_folds": N_FOLDS,
            "n_consultas": len(consultas),
            "consultas": {q: len(vs) for q, vs in sorted(consultas.items())},
            "ks": KS,
            "n_permutacoes_aleatorio": N_PERMUT,
            "seed": SEED,
            "classificador": "TF-IDF + regressao logistica multinomial (numpy)",
            "limitacao": ("a ordem original devolvida pela plataforma nao esta no corpus; "
                          "o baseline e a ordenacao aleatoria, nao a do YouTube"),
        },
        "resultados": resultados,
    }
    (RESULTADOS / "reranking_results.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nGravado: reranking_results.json")

    gera_figura(resultados)
    gera_tex(resultados, saida["protocolo"])
    return 0


def gera_figura(resultados: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rotulos = ROTULOS
    cores = {"aleatorio": "#9aa0a6", "bloco": "#ea4335", "sentenca": "#1a73e8",
             "oraculo": "#34a853"}

    fig, eixos = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # nDCG@k
    for sis in SISTEMAS:
        eixos[0].plot(KS, [resultados[sis][f"ndcg@{k}"] for k in KS],
                      marker="o", label=rotulos[sis], color=cores[sis], linewidth=2)
    eixos[0].set_xlabel("k"); eixos[0].set_ylabel("nDCG@k")
    eixos[0].set_title("Qualidade da ordenação (nDCG@k)")
    eixos[0].set_xticks(KS); eixos[0].set_ylim(0, 1.05)
    eixos[0].grid(alpha=.3); eixos[0].legend(fontsize=8)

    # P@k + MAP em barras
    grupos = [f"P@{k}" for k in KS] + ["MAP"]
    chaves = [f"p@{k}" for k in KS] + ["map"]
    x = np.arange(len(grupos)); larg = 0.2
    for i, sis in enumerate(SISTEMAS):
        eixos[1].bar(x + (i - 1.5) * larg, [resultados[sis][c] for c in chaves],
                     larg, label=rotulos[sis], color=cores[sis])
    eixos[1].set_xticks(x); eixos[1].set_xticklabels(grupos)
    eixos[1].set_ylabel("valor"); eixos[1].set_title("Precisão no topo e MAP")
    eixos[1].set_ylim(0, 1.05); eixos[1].grid(alpha=.3, axis="y")

    fig.tight_layout()
    destino = RESULTADOS / "reranking_curvas.png"
    fig.savefig(destino, dpi=300)
    print(f"Gravado: {destino.name} (300 dpi)")


def gera_tex(resultados: dict, protocolo: dict):
    """Trecho .tex pronto para incluir na secao de Resultados."""
    rot = ROTULOS
    linhas = []
    for sis in SISTEMAS:
        r = resultados[sis]
        vals = [f"{r[f'ndcg@{k}']:.3f}" for k in KS] + [f"{r[f'p@{k}']:.3f}" for k in KS] + [f"{r['map']:.3f}"]
        # sem negrito: bloco vence em nDCG, sentenca vence em MAP e P@3 —
        # destacar uma das duas induziria o leitor ao erro
        linhas.append(rot[sis] + " & " + " & ".join(vals) + r" \\")

    cab = " & ".join([f"nDCG@{k}" for k in KS] + [f"P@{k}" for k in KS] + ["MAP"])
    tex = f"""% GERADO POR src/experimentos/avaliar_reranking.py — nao editar a mao.
\\begin{{table}}[!t]
\\caption{{Avaliação da reordenação sobre {protocolo['n_consultas']} consultas derivadas dos títulos do corpus ({protocolo['n_videos']} vídeos, predição out-of-fold). O baseline é a ordenação aleatória, média de {protocolo['n_permutacoes_aleatorio']} permutações; o oráculo ordena pelo Score calculado sobre os rótulos humanos.}}\\label{{tab:reranking}}
\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}LRRRRRRR@{{}}}}
\\toprule
Ordenação & {cab} \\\\
\\midrule
{chr(10).join(linhas)}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    destino = RESULTADOS / "reranking_tabela.tex"
    destino.write_text(tex, encoding="utf-8")
    print(f"Gravado: {destino.name}")


if __name__ == "__main__":
    sys.exit(main())
