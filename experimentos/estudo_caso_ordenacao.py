"""
Estudo de caso: reordenacao de resultados REAIS de busca por analise de sentimentos.

MOTIVACAO
---------
A avaliacao de reordenacao ja presente no artigo (avaliar_reranking.py, Secao 6.6)
compara a ordem por Score contra uma ordenacao NAO INFORMADA (aleatoria), porque o
corpus de sentimentos nao registra a ordem em que a plataforma devolveu os resultados.
Este script fecha essa lacuna: usa os 40 videos coletados em quatro buscas reais no
YouTube, que preservam a POSICAO original devolvida pela plataforma, e mede o que a
reordenacao por sentimento faz com essa ordem.

DADOS
-----
Fonte: reordenacao_buscas.csv, nesta mesma pasta.
  4 consultas x 10 videos, com ESTRATO, TERMO, POSICAO (ordem do YouTube), TITULO,
  DESCRICAO e tres frases da transcricao (inicio, meio, fim).

  Estrato A (responsavel): consultas que um responsavel digitaria procurando conteudo
    infantil. Comportamento desejavel: NAO reordenar (nao ha risco a corrigir).
  Estrato B (apelo infantil): consultas de animacao adulta com forte apelo visual
    infantil. Comportamento desejavel: rebaixar.

Os videos deste estudo sao EXTERNOS ao corpus de sentimentos, com duas excecoes de
titulo coincidente, que sao REMOVIDAS do treino para garantir disjuncao (ver
VIDEOS_EXCLUIDOS no relatorio de saida).

O CSV veio de uma coleta feita para outro estudo, e dela aproveita-se APENAS o insumo
bruto — os videos, a ordem devolvida pela plataforma e as frases transcritas. Nenhum
rotulo, escore ou resultado daquele estudo entra aqui: rotulagem, classificador, escore
e conclusoes sao todos de analise de sentimentos.

CLASSIFICADOR
-------------
O mesmo da Secao 5 do artigo: BERTimbau Base (cased) ajustado, 5 epocas, lote 8,
lr 3e-5, 100 passos de aquecimento, agregado em ensemble por soft voting sobre os
cinco modelos da validacao cruzada estratificada. Random Oversampling apenas na
particao de treino de cada fold.

O ensemble e avaliado no conjunto de teste retido antes de ser aplicado ao estudo de
caso, para verificar que reproduz as metricas publicadas (Tabela 4 do artigo).

SCORE
-----
Equacao 1 do artigo: T_input = Titulo + Descricao + Frases(inicio, meio, fim)
Equacao 2 do artigo: S = 0.1 + 0.2*(1-C) se Negativo ; 0.85 caso contrario
                         (Neutro e Positivo nao se distinguem em risco)

Duas agregacoes, como na Secao 6.6:
  bloco    - um unico T_input por video (a adotada no sistema)
  sentenca - media dos scores das tres frases isoladas

ANALISES
--------
1. Separacao entre estratos: media por estrato, Mann-Whitney U, AUC com IC bootstrap.
2. Deteccao por limiar: quantos videos do estrato B ficam abaixo do limiar, contra
   falsos alarmes no estrato A.
3. Mudanca de ordem contra o YouTube: Kendall tau, sobreposicao de Top-3,
   deslocamento medio, destino do primeiro colocado da plataforma.

Saidas: estudo_caso_ordenacao.json, tabelas em tex/, figura fig06-case-study-scores.png
Uso: python estudo_caso_ordenacao.py [--folds N]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from caminhos import BUSCAS, CORPUS, FIGURAS, LATEX, RESULTADOS, prepara_saidas

prepara_saidas()

MODELO = "neuralmind/bert-base-portuguese-cased"
LABELS = ["Negativo", "Neutro", "Positivo"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}

SEED = 42
EPOCAS = 5
BATCH = 8
LR = 3e-5
WARMUP = 100
MAX_LEN_FRASE = 64
MAX_LEN_BLOCO = 256
TESTE_FRAC = 0.20
N_BOOT = 2000

torch.set_num_threads(20)

# Treinar cinco folds de BERTimbau em CPU leva horas; na GPU, minutos. O dispositivo e
# escolhido em tempo de execucao e nada mais no script depende dele: a semente, os
# hiperparametros e a particao sao os mesmos nos dois casos. As diferencas de ultima
# casa decimal entre CPU e GPU sao de ordem de operacoes em ponto flutuante, da mesma
# natureza das ja documentadas em REPRODUCAO.md entre versoes de biblioteca.
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normaliza(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt or "")
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", txt).strip().lower()


# --------------------------------------------------------------------- dados


def carrega_corpus(titulos_excluir: set) -> tuple:
    """Le o corpus de sentimentos, descartando videos do estudo de caso."""
    with open(CORPUS, encoding="utf-8-sig") as fh:
        linhas = list(csv.DictReader(fh, delimiter=";"))

    textos, rotulos, excluidos = [], [], Counter()
    for r in linhas:
        rot = (r.get("AS") or "").strip()
        frase = (r.get("FRASE") or "").strip().strip('"').strip()
        titulo = (r.get("TITULO") or "").strip()
        if not frase or rot not in LABEL_TO_ID:
            continue
        if normaliza(titulo) in titulos_excluir:
            excluidos[titulo] += 1
            continue
        textos.append(frase)
        rotulos.append(LABEL_TO_ID[rot])
    return textos, np.array(rotulos), dict(excluidos)


def carrega_buscas() -> list:
    with open(BUSCAS, encoding="utf-8-sig") as fh:
        linhas = list(csv.DictReader(fh, delimiter=";"))
    videos = []
    for r in linhas:
        frases = [
            (r.get(c) or "").strip()
            for c in ("FRASE_INICIO", "FRASE_MEIO", "FRASE_FIM")
        ]
        videos.append(
            {
                "estrato": (r.get("ESTRATO") or "").strip(),
                "termo": (r.get("TERMO") or "").strip(),
                "posicao": int(r.get("POSICAO")),
                "titulo": (r.get("TITULO") or "").strip(),
                "descricao": (r.get("DESCRICAO") or "").strip(),
                "frases": [f for f in frases if f],
            }
        )
    return videos


def bloco_entrada(v: dict) -> str:
    """Equacao 1: Titulo + Descricao + Frases(inicio, meio, fim)."""
    return " ".join([v["titulo"], v["descricao"], *v["frases"]]).strip()


# ------------------------------------------------------------------- modelo


class Sentencas(Dataset):
    def __init__(self, textos, rotulos, tok, max_len):
        self.textos, self.rotulos, self.tok, self.max_len = textos, rotulos, tok, max_len

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, i):
        return self.textos[i], (-1 if self.rotulos is None else int(self.rotulos[i]))


def colador(tok, max_len):
    def _f(lote):
        textos = [t for t, _ in lote]
        rot = torch.tensor([r for _, r in lote])
        enc = tok(
            textos,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc["labels"] = rot
        return enc

    return _f


def oversample(textos, rotulos, rng):
    """Random Oversampling: iguala as classes a maior, so no treino."""
    por_classe = defaultdict(list)
    for i, r in enumerate(rotulos):
        por_classe[int(r)].append(i)
    alvo = max(len(v) for v in por_classe.values())
    idx = []
    for c, ids in por_classe.items():
        idx.extend(ids)
        if len(ids) < alvo:
            idx.extend(rng.choice(ids, size=alvo - len(ids), replace=True).tolist())
    rng.shuffle(idx)
    return [textos[i] for i in idx], rotulos[idx]


def treina_fold(textos, rotulos, tok, rng, etiqueta=""):
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO, num_labels=3, local_files_only=True
    ).to(DEV)
    tx, ry = oversample(textos, rotulos, rng)
    ds = Sentencas(tx, ry, tok, MAX_LEN_FRASE)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=colador(tok, MAX_LEN_FRASE))

    opt = torch.optim.AdamW(modelo.parameters(), lr=LR)
    total = len(dl) * EPOCAS
    sched = get_linear_schedule_with_warmup(opt, WARMUP, total)

    modelo.train()
    t0, passo = time.time(), 0
    for ep in range(EPOCAS):
        for lote in dl:
            lote = {k: v.to(DEV) for k, v in lote.items()}
            saida = modelo(**lote)
            saida.loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            passo += 1
            if passo % 200 == 0:
                print(
                    f"    {etiqueta} passo {passo}/{total} "
                    f"({(time.time()-t0)/60:.1f} min)",
                    flush=True,
                )
    print(f"    {etiqueta} concluido em {(time.time()-t0)/60:.1f} min", flush=True)
    modelo.eval()
    return modelo


@torch.no_grad()
def probabilidades(modelo, textos, tok, max_len, batch=16):
    saidas = []
    for i in range(0, len(textos), batch):
        enc = tok(
            textos[i : i + batch],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(DEV) for k, v in enc.items()}
        logits = modelo(**enc).logits
        saidas.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(saidas)


def score_seguranca(P) -> float:
    """Equacao 2 do artigo: risco domina tom.

    Negativo -> [0.10, 0.233] ; nao-Negativo -> [0.70, 0.85], graduado por
    A = P(Positivo) + 0.5*P(Neutro). As faixas nao se sobrepoem, de modo que a
    gradacao por tom nunca compete com a decisao de risco.
    """
    P = np.asarray(P, dtype=float)
    if int(P.argmax()) == 0:
        return 0.10 + 0.20 * (1.0 - float(P[0]))
    return 0.70 + 0.15 * float(P[2] + 0.5 * P[1])


# ------------------------------------------------------------------ metricas


def metricas(y, pred):
    out = {}
    for c, nome in enumerate(LABELS):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[nome] = {"precisao": p, "recall": r, "f1": f, "suporte": int((y == c).sum())}
    out["acuracia"] = float((pred == y).mean())
    out["f1_macro"] = float(np.mean([out[n]["f1"] for n in LABELS]))
    out["precisao_macro"] = float(np.mean([out[n]["precisao"] for n in LABELS]))
    out["recall_macro"] = float(np.mean([out[n]["recall"] for n in LABELS]))
    return out


def mannwhitney(a, b):
    """U de Mann-Whitney com correcao para empates e aproximacao normal."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    todos = np.concatenate([a, b])
    ordem = todos.argsort()
    postos = np.empty(len(todos), float)
    i = 0
    while i < len(todos):
        j = i
        while j + 1 < len(todos) and todos[ordem[j + 1]] == todos[ordem[i]]:
            j += 1
        postos[ordem[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n2 = len(a), len(b)
    r1 = postos[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = max(u1, n1 * n2 - u1)
    # empates
    _, cont = np.unique(todos, return_counts=True)
    corr = (cont**3 - cont).sum()
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(
        (n1 * n2 / 12.0)
        * ((n1 + n2 + 1) - corr / ((n1 + n2) * (n1 + n2 - 1)))
    )
    z = (u - mu - 0.5) / sigma if sigma else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"u": float(u1), "z": float(z), "p": float(p)}


def auc(pos, neg):
    """AUC = P(score de A > score de B), com empates valendo 0.5."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    m = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(m / (len(pos) * len(neg)))


def auc_ic(pos, neg, rng, n=N_BOOT):
    vals = []
    for _ in range(n):
        p = rng.choice(pos, size=len(pos), replace=True)
        q = rng.choice(neg, size=len(neg), replace=True)
        vals.append(auc(p, q))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def kendall_tau(x, y):
    """Tau-b entre duas ordenacoes."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / den) if den else 0.0


# --------------------------------------------------------------------- main


def main():
    global EPOCAS

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=EPOCAS, help="so para smoke test")
    args = ap.parse_args()
    EPOCAS = args.epocas

    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    videos = carrega_buscas()
    titulos_estudo = {normaliza(v["titulo"]) for v in videos}
    textos, rotulos, excluidos = carrega_corpus(titulos_estudo)
    print(f"corpus apos exclusao: {len(textos)} sentencas")
    print(f"videos excluidos por coincidir com o estudo de caso: {excluidos}")

    # particao estratificada: 20% teste retido
    idx_teste, idx_resto = [], []
    for c in range(3):
        ids = np.where(rotulos == c)[0]
        rng.shuffle(ids)
        corte = int(round(len(ids) * TESTE_FRAC))
        idx_teste.extend(ids[:corte].tolist())
        idx_resto.extend(ids[corte:].tolist())
    idx_teste, idx_resto = np.array(sorted(idx_teste)), np.array(sorted(idx_resto))
    print(f"teste retido: {len(idx_teste)} | treino+validacao: {len(idx_resto)}")

    tok = AutoTokenizer.from_pretrained(MODELO, local_files_only=True)

    # folds estratificados sobre os 80%
    folds = [[] for _ in range(args.folds)]
    for c in range(3):
        ids = idx_resto[rotulos[idx_resto] == c]
        rng.shuffle(ids)
        for k, i in enumerate(ids):
            folds[k % args.folds].append(int(i))

    txt_teste = [textos[i] for i in idx_teste]
    y_teste = rotulos[idx_teste]
    blocos = [bloco_entrada(v) for v in videos]
    frases_planas, mapa_frases = [], []
    for k, v in enumerate(videos):
        for f in v["frases"]:
            frases_planas.append(f)
            mapa_frases.append(k)

    probs_teste, probs_bloco, probs_frase = [], [], []
    for k in range(args.folds):
        # com um unico fold (smoke test) treina-se sobre todos os 80%
        treino_ids = (
            idx_resto
            if args.folds == 1
            else np.array([i for j in range(args.folds) if j != k for i in folds[j]])
        )
        print(f"\n[fold {k+1}/{args.folds}] treino={len(treino_ids)}", flush=True)
        modelo = treina_fold(
            [textos[i] for i in treino_ids],
            rotulos[treino_ids],
            tok,
            rng,
            etiqueta=f"fold {k+1}",
        )
        probs_teste.append(probabilidades(modelo, txt_teste, tok, MAX_LEN_FRASE))
        probs_bloco.append(probabilidades(modelo, blocos, tok, MAX_LEN_BLOCO))
        probs_frase.append(probabilidades(modelo, frases_planas, tok, MAX_LEN_FRASE))
        del modelo
        if DEV.type == "cuda":
            torch.cuda.empty_cache()

    # ---- verificacao no teste retido (ensemble por soft voting)
    p_teste = np.mean(probs_teste, axis=0)
    verificacao = metricas(y_teste, p_teste.argmax(1))

    # Guarda as predicoes por amostra: sao o insumo dos testes de significancia e da
    # analise qualitativa de erros (significancia_e_erros.py). Sem isso qualquer analise
    # posterior no nivel da amostra exige repetir o treino inteiro.
    np.savez_compressed(
        RESULTADOS / "predicoes_teste.npz",
        probs_ensemble=p_teste,
        probs_por_fold=np.stack(probs_teste),
        y=y_teste,
        idx=idx_teste,
        textos=np.array(txt_teste, dtype=object),
        labels=np.array(LABELS),
    )
    print("predicoes salvas em predicoes_teste.npz", flush=True)
    por_fold = [
        metricas(y_teste, p.argmax(1))["f1_macro"] for p in probs_teste
    ]
    print("\n=== verificacao no teste retido ===")
    print(f"acuracia {verificacao['acuracia']:.4f} | F1-macro {verificacao['f1_macro']:.4f}")
    for n in LABELS:
        d = verificacao[n]
        print(f"  {n:9s} P={d['precisao']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f}")

    # ---- pontuacao dos 40 videos
    p_bloco = np.mean(probs_bloco, axis=0)
    p_frase = np.mean(probs_frase, axis=0)

    scores_frase = defaultdict(list)
    for i, k in enumerate(mapa_frases):
        scores_frase[k].append(score_seguranca(p_frase[i]))

    for k, v in enumerate(videos):
        c = int(p_bloco[k].argmax())
        v["classe"] = LABELS[c]
        v["confianca"] = float(p_bloco[k][c])
        # A adequacao predita (Eq. 3) e o que gradua a faixa segura; fica registrada
        # para que a figura e as tabelas possam ser auditadas sem reexecutar o ensemble.
        v["adequacao_predita"] = float(p_bloco[k][2] + 0.5 * p_bloco[k][1])
        v["probs"] = [float(x) for x in p_bloco[k]]
        v["score_bloco"] = score_seguranca(p_bloco[k])
        v["score_sentenca"] = float(np.mean(scores_frase[k])) if scores_frase[k] else 0.5
        v["classes_frases"] = [
            LABELS[int(p_frase[i].argmax())]
            for i in range(len(mapa_frases))
            if mapa_frases[i] == k
        ]

    # ---- separacao entre estratos
    estratos = {}
    for chave in ("score_bloco", "score_sentenca"):
        a = [v[chave] for v in videos if v["estrato"].startswith("A")]
        b = [v[chave] for v in videos if v["estrato"].startswith("B")]
        mw = mannwhitney(a, b)
        estratos[chave] = {
            "media_A": float(np.mean(a)),
            "media_B": float(np.mean(b)),
            "min_A": float(np.min(a)),
            "min_B": float(np.min(b)),
            "max_A": float(np.max(a)),
            "max_B": float(np.max(b)),
            "diferenca": float(np.mean(a) - np.mean(b)),
            "mannwhitney": mw,
            "auc": auc(a, b),
            "auc_ic95": auc_ic(np.array(a), np.array(b), rng),
        }

    por_termo = {}
    for termo in dict.fromkeys(v["termo"] for v in videos):
        vs = [v for v in videos if v["termo"] == termo]
        s = [v["score_bloco"] for v in vs]
        por_termo[termo] = {
            "estrato": vs[0]["estrato"],
            "n": len(vs),
            "media": float(np.mean(s)),
            "min": float(np.min(s)),
            "max": float(np.max(s)),
            "amplitude": float(np.max(s) - np.min(s)),
        }

    # ---- deteccao por limiar (score BAIXO = inadequado)
    deteccao = []
    for lim in (0.30, 0.40, 0.50, 0.60, 0.70):
        det = sum(
            1 for v in videos if v["estrato"].startswith("B") and v["score_bloco"] <= lim
        )
        falso = sum(
            1 for v in videos if v["estrato"].startswith("A") and v["score_bloco"] <= lim
        )
        deteccao.append(
            {"limiar": lim, "detectados_b": det, "total_b": 20, "falsos_a": falso, "total_a": 20}
        )

    # ---- mudanca de ordem contra o YouTube
    ordem = {}
    deslocamentos = []
    for termo in dict.fromkeys(v["termo"] for v in videos):
        vs = sorted([v for v in videos if v["termo"] == termo], key=lambda x: x["posicao"])
        nova = sorted(vs, key=lambda x: -x["score_bloco"])
        pos_nova = {id(v): i + 1 for i, v in enumerate(nova)}
        deltas = []
        for v in vs:
            d = pos_nova[id(v)] - v["posicao"]
            deltas.append(d)
            deslocamentos.append(
                {
                    "termo": termo,
                    "estrato": v["estrato"],
                    "titulo": v["titulo"],
                    "de": v["posicao"],
                    "para": pos_nova[id(v)],
                    "delta": d,
                    "score": v["score_bloco"],
                    "classe": v["classe"],
                }
            )
        top3_yt = {v["titulo"] for v in vs[:3]}
        top3_no = {v["titulo"] for v in nova[:3]}
        primeiro = vs[0]
        ordem[termo] = {
            "estrato": vs[0]["estrato"],
            "kendall_tau": kendall_tau(
                [v["posicao"] for v in vs], [pos_nova[id(v)] for v in vs]
            ),
            "deslocamento_medio": float(np.mean(np.abs(deltas))),
            "deslocamento_max": int(np.max(np.abs(deltas))),
            "top3_sobreposicao": len(top3_yt & top3_no) / 3.0,
            "primeiro_do_youtube": {
                "titulo": primeiro["titulo"],
                "para": pos_nova[id(primeiro)],
                "score": primeiro["score_bloco"],
                "classe": primeiro["classe"],
            },
            "amplitude_score": por_termo[termo]["amplitude"],
        }

    saida = {
        "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # so o nome do arquivo: o caminho absoluto da maquina nao interessa a quem
        # reproduz e vazaria a estrutura de diretorios de quem gerou o relatorio
        "fonte_videos": BUSCAS.name,
        "n_videos": len(videos),
        "n_consultas": len(ordem),
        "classificador": {
            "modelo": MODELO,
            "folds": args.folds,
            "epocas": EPOCAS,
            "batch": BATCH,
            "lr": LR,
            "warmup": WARMUP,
            "agregacao": "soft voting",
        },
        "videos_excluidos_do_treino": excluidos,
        "n_sentencas_treino_apos_exclusao": len(textos),
        "verificacao_teste_retido": verificacao,
        "f1_macro_por_fold": por_fold,
        "separacao_estratos": estratos,
        "por_termo": por_termo,
        "deteccao_por_limiar": deteccao,
        "mudanca_de_ordem": ordem,
        "deslocamentos": deslocamentos,
        "videos": [
            {
                k: v[k]
                for k in (
                    "estrato",
                    "termo",
                    "posicao",
                    "titulo",
                    "classe",
                    "confianca",
                    "adequacao_predita",
                    "probs",
                    "score_bloco",
                    "score_sentenca",
                    "classes_frases",
                )
            }
            for v in videos
        ],
    }
    (RESULTADOS / "estudo_caso_ordenacao.json").write_text(
        json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== separacao entre estratos (bloco) ===")
    e = estratos["score_bloco"]
    print(
        f"media A {e['media_A']:.3f} | media B {e['media_B']:.3f} | "
        f"dif {e['diferenca']:.3f} | AUC {e['auc']:.3f} "
        f"IC95 [{e['auc_ic95'][0]:.3f}; {e['auc_ic95'][1]:.3f}] | p {e['mannwhitney']['p']:.2e}"
    )
    print("\n=== mudanca de ordem ===")
    for t, d in ordem.items():
        print(
            f"  {t:32s} [{d['estrato'][:1]}] tau={d['kendall_tau']:+.3f} "
            f"top3={d['top3_sobreposicao']:.2f} desl={d['deslocamento_medio']:.1f} "
            f"amplitude={d['amplitude_score']:.3f}"
        )
    print("\nJSON salvo em estudo_caso_ordenacao.json")


if __name__ == "__main__":
    main()
