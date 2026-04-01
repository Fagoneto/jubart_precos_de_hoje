from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import pandas as pd
import unicodedata
import re

CSV_PATH = "precos_do_dia.csv"

produtos_geral = [
    "camarão", "tilápia", "salmão", "bacalhau", "cerveja",
    "picanha", "azeite", "frango", "merluza", "café"
]

TIPOS_TABELA = [
    {"value": "mais_barato", "label": "Produtos mais baratos"},
    {"value": "maior_desconto", "label": "Maiores descontos"},
]


def norm(txt: str) -> str:
    if txt is None:
        return ""
    txt = str(txt).strip().lower()
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("utf-8")
    txt = re.sub(r"\s+", " ", txt)
    return txt


def trabalho_com_medidas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai peso/volume a partir de 'produto' e calcula:
    - peso_gramas
    - volume_ml
    - preco_kg
    - preco_l
    - preco_padronizado
    - unidade_padrao
    """

    def extrair_medidas(texto: str) -> dict:
        t = str(texto).lower().replace(",", ".").strip()

        peso_gramas = None
        volume_ml = None

        # peso em kg
        m = re.search(r"(\d+(?:\.\d+)?)\s*(kg)\b", t)
        if m:
            peso_gramas = float(m.group(1)) * 1000

        # peso em g/gr
        if peso_gramas is None:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(g|gr)\b", t)
            if m:
                peso_gramas = float(m.group(1))

        # volume em litros
        m = re.search(r"(\d+(?:\.\d+)?)\s*(l)\b", t)
        if m:
            volume_ml = float(m.group(1)) * 1000

        # volume em ml
        if volume_ml is None:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(ml)\b", t)
            if m:
                volume_ml = float(m.group(1))

        return {
            "peso_gramas": peso_gramas,
            "volume_ml": volume_ml,
        }

    df = df.copy()

    if "produto" not in df.columns:
        raise ValueError("A coluna 'produto' não foi encontrada no CSV.")

    if "preco" not in df.columns:
        raise ValueError("A coluna 'preco' não foi encontrada no CSV.")

    medidas = df["produto"].apply(extrair_medidas).apply(pd.Series)
    df = pd.concat([df, medidas], axis=1)

    # ajustes manuais
    mask = df["produto"].str.contains(
        "Bacalhau Porto 08X10 Gadus Morhua 1,480 Kg",
        case=False,
        na=False,
    )
    df.loc[mask, "peso_gramas"] = 1480

    mask = df["produto"].str.contains(
        r"Bacalhau Porto Cod 7/9 Kg Pas",
        case=False,
        na=False,
    )
    df.loc[mask, "peso_gramas"] = 700

    df.loc[df["produto"] == "Filé De Tilápia Oba", "peso_gramas"] = 800

    df["preco"] = pd.to_numeric(df["preco"], errors="coerce")

    df["preco_kg"] = pd.NA
    df["preco_l"] = pd.NA
    df["preco_padronizado"] = pd.NA
    df["unidade_padrao"] = pd.NA

    mask_kg = df["peso_gramas"].notna() & (df["peso_gramas"] > 0) & df["preco"].notna()
    df.loc[mask_kg, "preco_kg"] = (
        df.loc[mask_kg, "preco"] / (df.loc[mask_kg, "peso_gramas"] / 1000)
    ).round(2)

    mask_l = df["volume_ml"].notna() & (df["volume_ml"] > 0) & df["preco"].notna()
    df.loc[mask_l, "preco_l"] = (
        df.loc[mask_l, "preco"] / (df.loc[mask_l, "volume_ml"] / 1000)
    ).round(2)

    df.loc[mask_kg, "preco_padronizado"] = df.loc[mask_kg, "preco_kg"]
    df.loc[mask_kg, "unidade_padrao"] = "kg"

    mask_somente_l = (~mask_kg) & mask_l
    df.loc[mask_somente_l, "preco_padronizado"] = df.loc[mask_somente_l, "preco_l"]
    df.loc[mask_somente_l, "unidade_padrao"] = "l"

    # força colunas numéricas
    df["preco_kg"] = pd.to_numeric(df["preco_kg"], errors="coerce")
    df["preco_l"] = pd.to_numeric(df["preco_l"], errors="coerce")
    df["preco_padronizado"] = pd.to_numeric(df["preco_padronizado"], errors="coerce")

    return df


def preparar_ordenacao(df_filtrado: pd.DataFrame) -> pd.DataFrame:
    dff = df_filtrado.copy()

    dff["preco"] = pd.to_numeric(dff["preco"], errors="coerce")
    dff["preco_padronizado"] = pd.to_numeric(dff["preco_padronizado"], errors="coerce")

    dff["tem_padronizacao"] = dff["preco_padronizado"].notna().astype(int)
    dff["criterio_ordem"] = dff["preco_padronizado"].fillna(dff["preco"])
    dff["criterio_ordem"] = pd.to_numeric(dff["criterio_ordem"], errors="coerce")

    return dff


def limpar_para_json(df_out: pd.DataFrame) -> list[dict]:
    if df_out.empty:
        return []
    df_out = df_out.copy()
    df_out = df_out.where(pd.notna(df_out), None)
    return df_out.to_dict(orient="records")


app = FastAPI(title="Jubart Preços")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# leitura da base
df = pd.read_csv(CSV_PATH)
df = trabalho_com_medidas(df)

# validações mínimas
if "cidade" not in df.columns:
    raise ValueError("A coluna 'cidade' não foi encontrada no CSV.")

if "data" not in df.columns:
    raise ValueError("A coluna 'data' não foi encontrada no CSV.")

df["preco"] = pd.to_numeric(df["preco"], errors="coerce")
df["preco_kg"] = pd.to_numeric(df["preco_kg"], errors="coerce")
df["preco_l"] = pd.to_numeric(df["preco_l"], errors="coerce")
df["preco_padronizado"] = pd.to_numeric(df["preco_padronizado"], errors="coerce")

if "desconto" in df.columns:
    df["desconto"] = pd.to_numeric(df["desconto"], errors="coerce")
else:
    df["desconto"] = pd.NA

# data de referência
data_ref = pd.to_datetime(df["data"], errors="coerce").max()
data_ref_str = data_ref.strftime("%d/%m/%Y") if pd.notna(data_ref) else ""

# colunas normalizadas
df["_cidade_norm"] = df["cidade"].map(norm)
df["_produto_norm"] = df["produto"].map(norm)

# listas para selects
CIDADES = sorted(df["cidade"].dropna().unique().tolist(), key=lambda x: norm(x))
PRODUTOS = produtos_geral[:]


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "cidades": CIDADES,
            "produtos": PRODUTOS,
            "tipos_tabela": TIPOS_TABELA,
            "data_referencia": data_ref_str,
        },
    )


@app.get("/api/pesquisa", response_class=JSONResponse)
def pesquisa(
    cidade: str = Query(...),
    produto: str = Query(...),
    palavra_chave: str = Query(""),
    tipo_tabela: str = Query("mais_barato", pattern="^(mais_barato|maior_desconto)$"),
    limit: int = Query(100, ge=5, le=500),
):
    cidade_n = norm(cidade)
    produto_n = norm(produto)
    palavra_n = norm(palavra_chave)

    dff = df[
        (df["_cidade_norm"] == cidade_n) &
        (df["_produto_norm"].str.contains(produto_n, na=False))
    ].copy()

    if palavra_n:
        dff = dff[dff["_produto_norm"].str.contains(palavra_n, na=False)].copy()

    dff = dff.dropna(subset=["preco"])

    if dff.empty:
        return {
            "cidade": cidade,
            "produto": produto,
            "palavra_chave": palavra_chave,
            "tipo_tabela": tipo_tabela,
            "data_referencia": data_ref_str,
            "total_encontrado": 0,
            "lista_precos": [],
            "lista_descontos": [],
            "ranking_escolhido": [],
        }

    dff = preparar_ordenacao(dff)

    # ranking principal
    dff_ordenado = dff.sort_values(
        by=["tem_padronizacao", "criterio_ordem", "preco"],
        ascending=[False, True, True],
    ).head(limit)

    # descontos
    dff_desc = dff[dff["desconto"].notna() & (dff["desconto"] != 0)].copy()

    if not dff_desc.empty:
        dff_desc = dff_desc.sort_values(
            by=["desconto", "tem_padronizacao", "criterio_ordem", "preco"],
            ascending=[True, False, True, True],
        ).head(limit)

    cols_lista = [
        "rede",
        "loja",
        "produto",
        "preco",
        "preco_padronizado",
        "unidade_padrao",
    ]

    cols_desc = [
        "rede",
        "loja",
        "produto",
        "preco",
        "preco_padronizado",
        "unidade_padrao",
        "desconto",
    ]

    cols_lista = [c for c in cols_lista if c in dff_ordenado.columns]
    cols_desc = [c for c in cols_desc if c in dff_desc.columns]

    lista_precos_json = limpar_para_json(dff_ordenado[cols_lista])
    lista_descontos_json = limpar_para_json(dff_desc[cols_desc]) if not dff_desc.empty else []

    ranking_escolhido_json = (
        lista_precos_json if tipo_tabela == "mais_barato" else lista_descontos_json
    )

    return {
        "cidade": cidade,
        "produto": produto,
        "palavra_chave": palavra_chave,
        "tipo_tabela": tipo_tabela,
        "data_referencia": data_ref_str,
        "total_encontrado": int(len(dff)),
        "lista_precos": lista_precos_json,
        "lista_descontos": lista_descontos_json,
        "ranking_escolhido": ranking_escolhido_json,
    }