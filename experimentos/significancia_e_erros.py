"""
Testes de significancia estatistica e analise qualitativa dos erros.

Fecha duas pendencias do manuscrito:

  (1) SIGNIFICANCIA — as diferencas entre o modelo proposto e os baselines eram
      reportadas como valores pontuais. Aqui elas sao acompanhadas de teste pareado,
      como o ESWA exige ("incluir testes estatisticos para sustentar afirmacoes
      empiricas").

  (2) TAXONOMIA DOS ERROS — a confusao entre Positivo e Neutro era descrita apenas
      como "proximidade semantica". Aqui os casos sao extraidos e caracterizados.

INSUMO
------
predicoes_teste.npz, produzido por estudo_caso_ordenacao.py: probabilidades do
ensemble por amostra, rotulos verdadeiros e os textos do conjunto de teste retido.
Sem ele nao ha como fazer teste PAREADO — comparar medias agregadas de execucoes
diferentes nao sustenta afirmacao de significancia.

Os baselines sao reexecutados aqui sobre EXATAMENTE a mesma particao, porque o
teste de McNemar exige as predicoes item a item dos dois sistemas comparados.

TESTES
------
  McNemar (com correcao de continuidade) sobre acertos/erros pareados: testa se a
  diferenca de ACURACIA e atribuivel ao acaso.
  Bootstrap pareado (10.000 reamostragens) sobre a diferenca de F1-MACRO: da
  intervalo de confianca e p-valor bilateral para a metrica primaria do artigo.

Saidas: significancia.json, erros_taxonomia.json e as tabelas tab-significancia.tex,
tab-teste.tex e tab-baselines.tex (em ../latex no monorepo, em saidas/ fora dele)
Uso: .venv/Scripts/python.exe significancia_e_erros.py
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from caminhos import BUSCAS, CORPUS, LATEX, RESULTADOS, SENTILEX as LEXICO, prepara_saidas

prepara_saidas()
PRED = RESULTADOS / "predicoes_teste.npz"

LABELS = ["Negativo", "Neutro", "Positivo"]
SEED = 42
N_BOOT = 10000
AVISO = "% GERADO POR src/experimentos/significancia_e_erros.py — nao editar a mao.\n"


def normaliza(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "")
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def tokeniza(t: str) -> list:
    return re.findall(r"[a-z0-9_]+", normaliza(t))


# ------------------------------------------------------------------ metricas


def por_classe(y, pred) -> list:
    """Precisao, recall e F1 de cada classe, na ordem de LABELS."""
    out = []
    for c in range(3):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append(
            {
                "precisao": p,
                "recall": r,
                "f1": 2 * p * r / (p + r) if p + r else 0.0,
                "suporte": tp + fn,
            }
        )
    return out


def f1_macro(y, pred) -> float:
    return float(np.mean([c["f1"] for c in por_classe(y, pred)]))


def metricas(y, pred) -> dict:
    """Bloco completo de metricas de um sistema: global, macro, ponderada e por classe."""
    cls = por_classe(y, pred)
    peso = np.array([c["suporte"] for c in cls], dtype=float)
    peso = peso / peso.sum()
    return {
        "acuracia": float((pred == y).mean()),
        "f1_macro": float(np.mean([c["f1"] for c in cls])),
        "precisao_macro": float(np.mean([c["precisao"] for c in cls])),
        "recall_macro": float(np.mean([c["recall"] for c in cls])),
        "precisao_ponderada": float(sum(p * c["precisao"] for p, c in zip(peso, cls))),
        "recall_ponderada": float(sum(p * c["recall"] for p, c in zip(peso, cls))),
        "f1_ponderada": float(sum(p * c["f1"] for p, c in zip(peso, cls))),
        "por_classe": {LABELS[i]: cls[i] for i in range(3)},
    }


def mcnemar(acertos_a: np.ndarray, acertos_b: np.ndarray) -> dict:
    """Testa se dois classificadores diferem em acuracia, item a item."""
    b = int((acertos_a & ~acertos_b).sum())  # so A acerta
    c = int((~acertos_a & acertos_b).sum())  # so B acerta
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # correcao de continuidade
    # p-valor de qui-quadrado com 1 grau de liberdade
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": float(chi2), "p": float(p)}


def bootstrap_f1(y, pred_a, pred_b, rng, n=N_BOOT) -> dict:
    """IC e p-valor bilateral para a diferenca de F1-macro, com reamostragem pareada."""
    obs = f1_macro(y, pred_a) - f1_macro(y, pred_b)
    n_amostras = len(y)
    difs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_amostras, n_amostras)
        difs[i] = f1_macro(y[idx], pred_a[idx]) - f1_macro(y[idx], pred_b[idx])
    # p bilateral: fracao de reamostragens que cruzam o zero, centrando em zero
    centradas = difs - difs.mean()
    p = float((np.abs(centradas) >= abs(obs)).mean())
    return {
        "diferenca": float(obs),
        "ic95": [float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))],
        "p": p,
    }


# ------------------------------------------------------------------ baselines


# Os baselines sao reexecutados com a configuracao EXATA de baselines.py — sklearn,
# TF-IDF com bigramas e sublinear_tf, regressao logistica com class_weight="balanced"
# e max_iter=3000. Reimplementar por conta propria produz baselines mais fracos e
# infla artificialmente a vantagem do modelo proposto.


def carrega_sentilex() -> dict:
    """SentiLex-PT: forma -> polaridade (-1, 0, +1), consolidada por POL:N0."""
    if not LEXICO.exists():
        return {}
    lex = {}
    with open(LEXICO, encoding="latin-1") as fh:
        for linha in fh:
            if ".PoS" not in linha or "POL" not in linha:
                continue
            forma = linha.split(",")[0].strip()
            m = re.search(r"POL:N0=(-?\d)", linha)
            if not m:
                continue
            lex[normaliza(forma)] = int(m.group(1))
    return lex


def prediz_sentilex(textos, lex) -> np.ndarray:
    out = []
    for t in textos:
        s = sum(lex.get(tok, 0) for tok in tokeniza(t))
        out.append(0 if s < 0 else (2 if s > 0 else 1))
    return np.array(out)


# ------------------------------------------------------------------ execucao


def carrega_corpus():
    """Recompoe o corpus na MESMA ordem e com a MESMA exclusao de estudo_caso_ordenacao.py.

    Aquele script remove as sentencas dos videos que aparecem no estudo de caso, para que
    os 40 videos fiquem fora da amostra. Sem repetir a exclusao aqui, os indices salvos no
    npz apontariam para linhas erradas e o teste pareado compararia amostras diferentes.
    """
    import csv

    buscas = BUSCAS
    # Sem o CSV nao ha como repetir a exclusao, e o resultado sairia sobre outra
    # amostra sem aviso; e melhor parar do que publicar numeros incomparaveis.
    if not buscas.exists():
        raise SystemExit(
            "reordenacao_buscas.csv nao encontrado — sem ele a exclusao dos videos "
            "do estudo de caso nao pode ser repetida e o teste pareado compararia "
            "amostras diferentes."
        )

    excluir = set()
    with open(buscas, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            excluir.add(re.sub(r"\s+", " ", normaliza(r.get("TITULO") or "")).strip())

    textos, rot = [], []
    with open(CORPUS, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            frase = (r.get("FRASE") or "").strip().strip('"').strip()
            rotulo = (r.get("AS") or "").strip()
            titulo = (r.get("TITULO") or "").strip()
            if not frase or rotulo not in LABELS:
                continue
            if re.sub(r"\s+", " ", normaliza(titulo)).strip() in excluir:
                continue
            textos.append(frase)
            rot.append(LABELS.index(rotulo))
    return textos, np.array(rot)


def main():
    if not PRED.exists():
        raise SystemExit(
            f"{PRED.name} nao encontrado. Rode estudo_caso_ordenacao.py antes: "
            "ele produz as predicoes por amostra de que este script depende."
        )
    rng = np.random.default_rng(SEED)
    d = np.load(PRED, allow_pickle=True)
    probs = d["probs_ensemble"]
    y = d["y"]
    idx_teste = d["idx"]
    txt_teste = list(d["textos"])
    pred_proposto = probs.argmax(1)
    print(f"teste retido: {len(y)} sentencas")

    # --- reexecuta os baselines na MESMA particao
    todos_textos, todos_rot = carrega_corpus()
    print(f"corpus (com a mesma exclusao do estudo de caso): {len(todos_rot)} sentencas")
    mask_teste = np.zeros(len(todos_rot), dtype=bool)
    mask_teste[idx_teste] = True

    # O teste so e pareado se as amostras forem exatamente as mesmas: conferimos texto
    # a texto contra o que foi salvo no npz, e abortamos se divergir.
    tx_teste = [todos_textos[i] for i in range(len(todos_rot)) if mask_teste[i]]
    divergentes = sum(1 for a, b in zip(tx_teste, txt_teste) if a != b)
    if divergentes:
        raise SystemExit(
            f"{divergentes} de {len(tx_teste)} textos do teste divergem do npz. "
            "A particao nao e a mesma; o teste pareado seria invalido."
        )

    tx_treino = [todos_textos[i] for i in range(len(todos_rot)) if not mask_teste[i]]
    y_treino = todos_rot[~mask_teste]
    print(f"baselines: treino {len(tx_treino)} | teste {len(tx_teste)}")
    assert np.array_equal(todos_rot[mask_teste], y), "rotulos do teste divergem do npz"

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def reg_log():
        return LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=SEED
        )

    sistemas = {}

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True, sublinear_tf=True)
    Xtr = vec.fit_transform(tx_treino)
    sistemas["TF-IDF + regressão logística"] = (
        reg_log().fit(Xtr, y_treino).predict(vec.transform(tx_teste))
    )

    lex = carrega_sentilex()
    if lex:
        sistemas["SentiLex-PT"] = prediz_sentilex(tx_teste, lex)
        print(f"SentiLex-PT: {len(lex)} formas")
    else:
        print("AVISO: lexico ausente em _recursos/, SentiLex-PT sera pulado")

    # BERTimbau congelado como extrator de atributos
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(20)
        tok = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased", local_files_only=True
        )
        bert = AutoModel.from_pretrained(
            "neuralmind/bert-base-portuguese-cased", local_files_only=True
        ).eval()

        @torch.no_grad()
        def embed(textos, batch=16, max_len=128):
            """Mean pooling com mascara de atencao — igual a baselines.py."""
            saida = []
            for i in range(0, len(textos), batch):
                enc = tok(
                    [str(x) for x in textos[i : i + batch]],
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                h = bert(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                saida.append(((h * m).sum(1) / m.sum(1)).numpy())
            return np.vstack(saida)

        print("extraindo embeddings do BERTimbau congelado...", flush=True)
        E_tr, E_te = embed(tx_treino), embed(tx_teste)
        sistemas["BERTimbau congelado + RL"] = (
            reg_log().fit(E_tr, y_treino).predict(E_te)
        )
    except Exception as e:  # pragma: no cover
        print(f"AVISO: BERTimbau congelado pulado ({e})")

    # --- testes pareados contra o proposto
    ac_prop = pred_proposto == y
    f1_prop = f1_macro(y, pred_proposto)
    resultados = {
        "n_teste": int(len(y)),
        "proposto": metricas(y, pred_proposto),
        "comparacoes": {},
    }
    print(f"\nproposto: acc={ac_prop.mean():.4f} F1-macro={f1_prop:.4f}\n")
    for nome, pred in sistemas.items():
        ac = pred == y
        mc = mcnemar(ac_prop, ac)
        bs = bootstrap_f1(y, pred_proposto, pred, rng)
        resultados["comparacoes"][nome] = {
            **metricas(y, pred),
            "delta_acuracia": float(ac_prop.mean() - ac.mean()),
            "mcnemar": mc,
            "bootstrap_f1_macro": bs,
        }
        print(
            f"{nome:32s} acc={ac.mean():.4f} F1={f1_macro(y, pred):.4f} | "
            f"McNemar b={mc['b']} c={mc['c']} chi2={mc['chi2']:.1f} p={mc['p']:.2e} | "
            f"dF1={bs['diferenca']:+.4f} IC95[{bs['ic95'][0]:+.4f};{bs['ic95'][1]:+.4f}] "
            f"p={bs['p']:.4f}"
        )

    (RESULTADOS / "significancia.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------ taxonomia dos erros
    conf = np.zeros((3, 3), dtype=int)
    for t, p in zip(y, pred_proposto):
        conf[t, p] += 1

    erros = []
    for i, (t, p) in enumerate(zip(y, pred_proposto)):
        if t == p:
            continue
        texto = txt_teste[i]
        toks = tokeniza(texto)
        erros.append(
            {
                "texto": texto,
                "verdadeiro": LABELS[int(t)],
                "predito": LABELS[int(p)],
                "confianca": float(probs[i][p]),
                "n_tokens": len(toks),
                "interrogativa": texto.strip().endswith("?"),
                "exclamativa": texto.strip().endswith("!"),
                "tem_negacao": any(x in toks for x in ("nao", "nem", "sem", "nunca")),
                "tem_neg_marcado": "neg_" in normaliza(texto),
            }
        )

    par = [e for e in erros if {e["verdadeiro"], e["predito"]} == {"Positivo", "Neutro"}]
    resumo = {
        "matriz_confusao": conf.tolist(),
        "labels": LABELS,
        "n_erros": len(erros),
        "n_erros_pos_neu": len(par),
        "fracao_pos_neu": len(par) / len(erros) if erros else 0.0,
        "por_direcao": dict(
            Counter(f"{e['verdadeiro']}->{e['predito']}" for e in erros)
        ),
        "comprimento_mediano": {
            "acertos": float(
                np.median([len(tokeniza(txt_teste[i])) for i in range(len(y)) if y[i] == pred_proposto[i]])
            ),
            "erros_pos_neu": float(np.median([e["n_tokens"] for e in par])) if par else None,
        },
        "confianca_mediana": {
            "acertos": float(
                np.median([probs[i][pred_proposto[i]] for i in range(len(y)) if y[i] == pred_proposto[i]])
            ),
            "erros_pos_neu": float(np.median([e["confianca"] for e in par])) if par else None,
        },
        "interrogativas_pct": {
            "erros_pos_neu": float(np.mean([e["interrogativa"] for e in par])) if par else None,
        },
        "erros": erros,
    }
    (RESULTADOS / "erros_taxonomia.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n=== erros ===")
    print(f"total {len(erros)} | Positivo<->Neutro {len(par)} ({len(par)/max(len(erros),1):.1%})")
    print(f"por direcao: {resumo['por_direcao']}")
    print(
        f"comprimento mediano: acertos {resumo['comprimento_mediano']['acertos']:.0f} tokens, "
        f"erros Pos<->Neu {resumo['comprimento_mediano']['erros_pos_neu']} tokens"
    )
    print(
        f"confianca mediana: acertos {resumo['confianca_mediana']['acertos']:.3f}, "
        f"erros Pos<->Neu {resumo['confianca_mediana']['erros_pos_neu']:.3f}"
    )

    # ------------------------------------------------------ tabela
    def num(x, c=4):
        return f"{x:.{c}f}".replace(".", ",")

    def pval(p):
        if p < 1e-4:
            return "$< 10^{-4}$"
        return f"${num(p, 4)}$".replace(".", ",")

    linhas = []
    ordem = ["SentiLex-PT", "TF-IDF + regressão logística", "BERTimbau congelado + RL"]
    for nome in ordem:
        if nome not in resultados["comparacoes"]:
            continue
        c = resultados["comparacoes"][nome]
        bs = c["bootstrap_f1_macro"]
        linhas.append(
            f"{nome} & {num(c['f1_macro']*100,2)} & "
            f"{num(bs['diferenca']*100,2)} & "
            f"[{num(bs['ic95'][0]*100,2)}; {num(bs['ic95'][1]*100,2)}] & "
            f"{pval(bs['p'])} & {pval(c['mcnemar']['p'])} \\\\"
        )
    corpo = "\n".join(linhas)
    tabela = f"""{AVISO}\\begin{{table}}[!t]
\\caption{{Significância das diferenças entre o \\emph{{ensemble}} proposto e cada \\emph{{baseline}}, sobre a mesma partição de teste ({len(y)} sentenças). $\\Delta$F1 é a diferença de F1-macro em pontos percentuais, em favor do modelo proposto; o intervalo e o $p$ vêm de \\emph{{bootstrap}} pareado com 10.000 reamostragens. A última coluna traz o $p$ do teste de McNemar sobre a acurácia.}}\\label{{tab:significancia}}
\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}LRRRRR@{{}}}}
\\toprule
\\emph{{Baseline}} & F1-macro & $\\Delta$F1 & IC 95\\% & $p$ (\\emph{{bootstrap}}) & $p$ (McNemar) \\\\
\\midrule
{corpo}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    (LATEX / "tab-significancia.tex").write_text(tabela, encoding="utf-8")

    # ------------------------------------------------- tabelas 4 e 5 do manuscrito
    # Geradas aqui para que TODAS as tabelas do artigo venham da mesma execucao:
    # mesmo ensemble, mesma particao, mesmos baselines. Enquanto a Tabela 4 vinha do
    # treino original (550 sentencas) e a analise de erros desta secao, das predicoes
    # de predicoes_teste.npz (544), as duas se contradiziam na matriz de confusao.
    p = resultados["proposto"]
    linhas_teste = [
        f"{c} & {num(p['por_classe'][c]['precisao']*100,1)} & "
        f"{num(p['por_classe'][c]['recall']*100,1)} & "
        f"{num(p['por_classe'][c]['f1']*100,1)} \\\\"
        for c in LABELS
    ]
    tab_teste = f"""{AVISO}\\begin{{table}}[!t]
\\caption{{Métricas do \\emph{{ensemble}} proposto no conjunto de teste retido ({len(y)} sentenças). Valores em porcentagem.}}\\label{{tab:teste}}
\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}LRRR@{{}}}}
\\toprule
Classe / agregação & Precisão & \\emph{{Recall}} & F1 \\\\
\\midrule
{chr(10).join(linhas_teste)}
\\midrule
Macro & {num(p['precisao_macro']*100,2)} & {num(p['recall_macro']*100,2)} & {num(p['f1_macro']*100,2)} \\\\
Ponderada & {num(p['precisao_ponderada']*100,2)} & {num(p['recall_ponderada']*100,2)} & {num(p['f1_ponderada']*100,2)} \\\\
\\midrule
\\multicolumn{{4}}{{@{{}}l}}{{Acurácia: {num(p['acuracia']*100,2)}}} \\\\
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    (LATEX / "tab-teste.tex").write_text(tab_teste, encoding="utf-8")

    rotulo_tabela = {
        "SentiLex-PT": "SentiLex-PT",
        "TF-IDF + regressão logística": "TF-IDF + RL",
        "BERTimbau congelado + RL": "BERTimbau congelado + RL",
    }
    linhas_base = []
    for nome in ordem:
        if nome not in resultados["comparacoes"]:
            continue
        c = resultados["comparacoes"][nome]
        linhas_base.append(
            f"{rotulo_tabela[nome]} & {num(c['acuracia']*100,2)} & {num(c['f1_macro']*100,2)} & "
            + " & ".join(num(c["por_classe"][k]["f1"] * 100, 1) for k in LABELS)
            + " \\\\"
        )
    linhas_base.append(
        f"BERTimbau ajustado (\\emph{{ensemble}}) & \\textbf{{{num(p['acuracia']*100,2)}}} & "
        f"\\textbf{{{num(p['f1_macro']*100,2)}}} & "
        + " & ".join(f"\\textbf{{{num(p['por_classe'][k]['f1']*100,1)}}}" for k in LABELS)
        + " \\\\"
    )
    tab_base = f"""{AVISO}\\begin{{table}}[!t]
\\caption{{Comparação com os \\emph{{baselines}} no conjunto de teste retido ({len(y)} sentenças). Valores em porcentagem.}}\\label{{tab:baselines}}
\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}LRRRRR@{{}}}}
\\toprule
Método & Acurácia & F1-macro & F1 Neg. & F1 Neu. & F1 Pos. \\\\
\\midrule
{chr(10).join(linhas_base)}
\\bottomrule
\\end{{tabular*}}
\\end{{table}}
"""
    (LATEX / "tab-baselines.tex").write_text(tab_base, encoding="utf-8")

    print(
        "\nsalvo: significancia.json, erros_taxonomia.json, "
        f"{LATEX}/{{tab-significancia,tab-teste,tab-baselines}}.tex"
    )


if __name__ == "__main__":
    main()
