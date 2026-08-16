# Plataforma YouTube Safe Kids

A aplicação descrita no artigo: recebe uma consulta, busca na YouTube Data API, pontua cada
resultado com os filtros habilitados e devolve a listagem **reordenada**. Nenhum vídeo é
removido — o que muda é a posição.

É a mesma base de código do protótipo mostrado na figura da interface do artigo.

## Como rodar

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate no Linux/Mac
pip install -r requirements.txt

cp .env.example .env            # e preencha YOUTUBE_API_KEY
uvicorn app.main:app --reload
```

A aplicação sobe em `http://localhost:8000`.

Duas coisas precisam existir antes de a busca funcionar de ponta a ponta:

1. **Chave da YouTube Data API v3**, em `.env`. Sem ela a busca não retorna nada: a
   plataforma consulta a API oficial, não raspa a interface.
2. **Modelo de sentimentos ajustado**, em `app/nlp/models/trained/AS_*`. Os pesos
   **não são versionados** (são ~440 MB). Sem eles o `SentimentFilter` registra o erro no
   log e devolve 0,5 para todo vídeo — a aplicação sobe, mas o filtro de sentimento fica
   inerte. Para gerar os pesos, ver `../experimentos/`.

## Arquitetura

A divisão cliente/servidor é a da seção *Arquitetura do protótipo* do artigo: ao servidor
cabe a cadeia inteira, da consulta ao escore; ao cliente, apresentar a listagem já
ordenada com escore e classe visíveis.

```
app/
├── main.py                 sobe o FastAPI e registra os dez filtros
├── api/endpoints/videos.py o endpoint de busca: consulta, pontua e ordena
├── core/
│   ├── config.py           configurações e chave da API
│   ├── youtube.py          cliente da YouTube Data API + transcrição
│   └── logging.py
├── filters/
│   ├── base.py             contrato de um filtro: process(video) -> escore em [0,1]
│   ├── __init__.py         FilterManager: agrega os escores dos filtros habilitados
│   ├── sentiment.py        o filtro deste artigo
│   └── ...                 duration, age_rating, educational, toxicity, language,
│                           diversity, interactivity, engagement, sensitive
├── nlp/
│   ├── config.py           tarefas, hiperparâmetros e caminho do corpus
│   ├── models/             classificadores BERTimbau, um por tarefa
│   ├── training/           treino
│   ├── evaluation/         avaliação
│   └── utils/
├── static/, templates/     interface
scripts/                    treino, avaliação e comparação por linha de comando
config/                     configurações de experimento
```

Como o contrato entre cliente e servidor trafega **escore e classe**, e não a saída bruta
do modelo, acrescentar um filtro equivale a acrescentar um produtor de escores: sem efeito
sobre a interface nem sobre os filtros já em operação. É o que `base.py` formaliza.

## Onde estão as equações do artigo

| No artigo | No código |
|---|---|
| Equação (1) — bloco único: título + descrição + trechos da transcrição | `app/filters/sentiment.py`, início de `process` |
| Equação (2) — Score de Segurança por classe e confiança $C$ | `app/filters/sentiment.py`, o bloco `if predicted_class == 0 ... else` |
| Equação (3) — escore global como média dos $N$ filtros habilitados | `app/filters/__init__.py`, `FilterManager.process_video` |
| Reordenação sem remoção | `app/api/endpoints/videos.py`, a ordenação por `final_score` |
| Cor do indicador — verde/amarelo/vermelho pela **classe** | `app/static/js/main.js`, o mapa `CORES` |

Sobre a cor do indicador: ela sinaliza a **classe**, não o escore — que é o que a Seção 4.7
do artigo descreve. As duas coisas respondem perguntas diferentes. O **escore** diz como
ordenar, e a Equação (2) de propósito **não** separa Neutro de Positivo, porque a diferença
entre eles é de tom e não de risco: separá-los rebaixaria conteúdo informativo em favor de
conteúdo afetivo sem que houvesse diferença de adequação. A **cor** diz o que o item é, e aí
as três classes valem a pena continuar visíveis — quem supervisiona a criança se beneficia
de saber que o vídeo é informativo em vez de afetivo, mesmo que isso não deva mexer na
ordem.

Por isso o indicador tem **três** cores mesmo com um escore de dois valores: verde para
Positivo, amarelo para Neutro, vermelho para Negativo. O amarelo é alcançável por
construção, e não por acidente de faixa. Quando não há classe disponível — filtro de
sentimento desligado, ou vários filtros e o escore sendo a média da Equação (3) — a cor
volta a sair do escore, pelos limites das faixas.

A classe viaja do servidor ao cliente no campo `sentiment_class`, escrito por
`app/filters/sentiment.py`: o contrato entre as duas pontas carrega **escore e classe**,
como o artigo descreve, e não a saída bruta do modelo.

Sobre a Equação (3): o código calcula uma **média ponderada**, com o peso de cada filtro
vindo do cliente. Com todos os pesos em 1,0 — o padrão, e a configuração usada no artigo —
ela é exatamente a média aritmética da Equação (3). Os pesos existem para o trabalho
futuro discutido no artigo, quando houver evidência sobre a importância relativa de cada
dimensão de risco.

## Corpus

A plataforma **não guarda cópia do corpus**. `app/nlp/config.py` aponta para
`../dados/corpus.csv`, o único corpus do repositório — o mesmo que os experimentos usam.

Uma cópia própria, multitarefa e desatualizada, existia na versão de origem desta
aplicação e já havia divergido: 1.801 sentenças contra as 2.749 do artigo. Ela não foi
publicada aqui, para que não haja dois corpora em circulação. Pelo mesmo motivo ficaram de
fora os resultados de avaliação antigos, calculados sobre aquele recorte.

`../dados/corpus-rubrica.md` descreve o significado de cada classe nas quatro tarefas
(sentimento, toxicidade, linguagem imprópria e tópicos educacionais) — é a rubrica usada
na anotação.

## Escopo

Dos dez filtros registrados, **apenas o de sentimento é objeto deste artigo**. Os demais
estão aqui porque são parte da plataforma e porque a Equação (3) só faz sentido diante de
múltiplos filtros; seus modelos e sua avaliação pertencem a outros trabalhos e não são
reivindicados aqui.

`README_EXPERIMENTS.md` documenta a interface de linha de comando de `scripts/`, herdada
da aplicação. Para reproduzir os números **do artigo**, o caminho é `../experimentos/`,
que é autocontido e não depende de subir a aplicação.
