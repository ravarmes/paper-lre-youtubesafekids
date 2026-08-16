"""Caminhos compartilhados pelos scripts de experimento.

Um unico lugar decide onde estao os insumos e para onde vao as saidas, de modo que
mover a pasta nao exija editar dez scripts.

  dados/        corpus.csv, reordenacao_buscas.csv — entram, nunca sao escritos
  resultados/   .json e .npz versionados: e o que permite auditar os numeros do
                artigo sem reexecutar nada
  _recursos/    insumos nao redistribuidos (SentiLex-PT); fora do controle de versao
  saidas/       tabelas .tex e figuras do manuscrito, quando esta pasta e usada
                sozinha. Dentro do monorepo do artigo elas vao direto para ../../latex,
                que e de onde o main.tex as le.
"""

from __future__ import annotations

from pathlib import Path

EXPERIMENTOS = Path(__file__).parent
BASE = EXPERIMENTOS.parent

DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"
RECURSOS = EXPERIMENTOS / "_recursos"

CORPUS = DADOS / "corpus.csv"
BUSCAS = DADOS / "reordenacao_buscas.csv"
SENTILEX = RECURSOS / "SentiLex-flex-PT02.txt"

_LATEX_ARTIGO = BASE.parent / "latex"
LATEX = _LATEX_ARTIGO if _LATEX_ARTIGO.is_dir() else BASE / "saidas"
FIGURAS = LATEX / "figuras"


def prepara_saidas() -> None:
    """Cria os diretorios de escrita. Chamar antes de gravar qualquer coisa."""
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
