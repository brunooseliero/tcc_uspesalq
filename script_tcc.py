# -*- coding: utf-8 -*-
# %% [markdown]
# # Machine Learning na Predição de Ações
#
# Este notebook implementa o pipeline principal do TCC em duas etapas:
#
# - **Etapa 1:** Engenharia de Dados (ETL) e construção do dataset.
# - **Etapa 2:** Machine Learning, Deep Learning e testes estatísticos.
#
# Fontes usadas:
# - Ações: preços oficiais da B3 via `COTAHIST`.
# - Ibovespa: fallback documentado via `yfinance`, usado apenas como benchmark.
# - Fundamentos: CVM DFP anual consolidada.
# - Macroeconomia: BCB SGS.
#
# A organização foi pensada para notebook: cada bloco `# %%` vira uma célula no
# Jupyter/VS Code/Colab.

# %% [markdown]
# ## Etapa 0 - Configuração Inicial
#
# Nesta célula concentramos imports, parâmetros e nomes de arquivos.

# %%
import base64
import json
import re
import unicodedata
import warnings
from io import BytesIO
from pathlib import Path
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

try:
    from IPython.display import display
except ImportError:
    display = print
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.stats import chi2_contingency, mannwhitneyu
import xgboost as xgb

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Período do estudo.
DATA_INICIO = "2010-01-01"
DATA_FIM = "2025-12-31"
ANOS_ESPERADOS_TCC = set(range(2010, 2026))

# Parâmetros metodológicos.
ALIQUOTA_IR = 0.34
MIN_ANOS_TREINO = 8
MIN_ANOS_TESTE = 3
MIN_ANOS_PRECO_VALIDO = 5
MIN_PREGOES_ANO = 120
MIN_OBSERVACOES_BASE = 400

# Controle de execução.
EXECUTAR_ETAPA_1 = True
VALIDAR_PERIODO_TCC = True
PERMITIR_FALLBACK_YFINANCE_IBOV = True
APLICAR_FILTRO_SIGNIFICANCIA = True
ALPHA_SIGNIFICANCIA_FEATURES = 0.05
MIN_FEATURES_APOS_FILTRO = 5

# Arquivos gerados pelo pipeline.
ARQUIVO_BASE = Path("DATABASE_MESTRE_FINAL.csv")
ARQUIVO_UNIVERSO = Path("universo_empresas_b3_cvm.csv")
ARQUIVO_DIAGNOSTICO_ETL = Path("diagnostico_cobertura.csv")
ARQUIVO_DIAGNOSTICO_SETOR = Path("diagnostico_cobertura_setor.csv")
ARQUIVO_DIAGNOSTICO_SEGMENTO = Path("diagnostico_cobertura_segmento.csv")
ARQUIVO_TICKERS_REMOVIDOS = Path("diagnostico_tickers_removidos.csv")
ARQUIVO_RESULTADOS = Path("resultados_modelos.csv")
ARQUIVO_SELECAO_FEATURES = Path("selecao_features_significancia.csv")
ARQUIVO_IMPORTANCIA = Path("importancia_variaveis.csv")
ARQUIVO_ESTATISTICAS = Path("estatisticas_descritivas.csv")
ARQUIVO_MATRIZ_CONFUSAO_RF = Path("matriz_confusao_random_forest.csv")

GRAFICO_CORRELACAO = Path("grafico_correlacao.png")
GRAFICO_DESEMPENHO = Path("grafico_desempenho_modelos.png")
GRAFICO_IMPORTANCIA = Path("grafico_feature_importance.png")
GRAFICO_MATRIZ_CONFUSAO = Path("grafico_matriz_confusao_random_forest.png")

# Cache local para não baixar arquivos grandes repetidamente.
DIR_CACHE_COTAHIST = Path("cotahist")
DIR_CACHE_CVM = Path("cvm_dfp")
DIR_CACHE_COTAHIST.mkdir(parents=True, exist_ok=True)
DIR_CACHE_CVM.mkdir(parents=True, exist_ok=True)

# Fontes oficiais/públicas.
COTAHIST_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP"
CVM_DFP_URL = (
    "https://dados.cvm.gov.br/dados/cia_aberta/DOC/DFP/DADOS/dfp_cia_aberta_{ano}.zip"
)
BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"
B3_INDEX_URL = "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/{payload}"
B3_COMPANIES_URL = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetInitialCompanies/{payload}"
B3_CLASSIFICACAO_SETORIAL_URL = (
    "https://bvmf.bmfbovespa.com.br/InstDados/InformacoesEmpresas/ClassifSetorial.zip"
)

INDICE_UNIVERSO_B3 = "IBRA"
SERIE_SELIC_META = 432
SERIE_IPCA_MENSAL = 433

SEGMENTOS_LISTAGEM_ESPERADOS = [
    "Segmento_Novo_Mercado",
    "Segmento_Nivel_1",
    "Segmento_Nivel_2",
    "Segmento_Bovespa_Mais",
    "Segmento_Bovespa_Mais_Nivel_2",
    "Segmento_Tradicional",
]

# %% [markdown]
# ## Funções auxiliares
#
# As funções abaixo são pequenas de propósito: elas deixam claro o que é limpeza,
# coleta, cálculo financeiro e validação.


# %%
def display_df(df: pd.DataFrame, linhas: int = 5) -> None:
    """Mostra DataFrame no notebook ou no terminal."""
    display(df.head(linhas))


def normalizar_texto(texto: object) -> str:
    """Remove acentos e padroniza textos para comparações robustas."""
    texto = "" if pd.isna(texto) else str(texto).upper()
    texto = "".join(
        caractere
        for caractere in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(caractere)
    )
    return " ".join(texto.replace("-", " ").replace("/", " ").split())


def normalizar_nome_coluna(texto: object) -> str:
    """Cria nomes de coluna simples para dummies de setor."""
    texto = normalizar_texto(texto).title().replace(" ", "_")
    texto = re.sub(r"[^A-Za-z0-9_]", "", texto)
    return texto or "Nao_Classificado"


def retorno_composto_anual(retornos_diarios: pd.Series) -> float:
    """Calcula retorno composto no ano a partir dos retornos diários."""
    retornos_validos = retornos_diarios.dropna()
    if retornos_validos.empty:
        return np.nan
    return float((1 + retornos_validos).prod() - 1)


def max_drawdown(precos: pd.Series) -> float:
    """Calcula a maior queda percentual a partir do pico no ano."""
    precos = precos.dropna()
    if precos.empty:
        return np.nan
    acumulado_maximo = precos.cummax()
    drawdown = precos / acumulado_maximo - 1
    return float(drawdown.min())


def beta_contra_benchmark(retornos_acao: pd.Series, retornos_ibov: pd.Series) -> float:
    """Calcula beta anual da ação contra o benchmark."""
    dados = pd.concat([retornos_acao, retornos_ibov], axis=1).dropna()
    if len(dados) < 30:
        return np.nan
    variancia_benchmark = dados.iloc[:, 1].var()
    if pd.isna(variancia_benchmark) or variancia_benchmark == 0:
        return np.nan
    return float(dados.iloc[:, 0].cov(dados.iloc[:, 1]) / variancia_benchmark)


def divisao_segura(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    """Evita divisão por zero substituindo zero por NaN."""
    return numerador / denominador.replace(0, np.nan)


def reportar_falhas(etapa: str, falhas: list[tuple[str, str]], limite: int = 8) -> None:
    """Mostra falhas de coleta sem esconder o tamanho do problema."""
    if not falhas:
        return
    print(f"[Aviso] {etapa}: {len(falhas)} falha(s).")
    for item, mensagem in falhas[:limite]:
        print(f" - {item}: {mensagem}")
        if len(falhas) > limite:
            print(f" - ... e mais {len(falhas) - limite} falha(s).")


def b3_payload(payload: dict) -> str:
    """Codifica payload no formato esperado pelos endpoints públicos da B3."""
    texto = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return base64.b64encode(texto.encode("utf-8")).decode("utf-8")


def baixar_json_b3(url_template: str, payload: dict) -> dict:
    """Baixa JSON da B3 com cabeçalhos de navegador."""
    url = url_template.format(payload=b3_payload(payload))
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"}
    with urlopen(Request(url, headers=headers), timeout=60) as resposta:
        return json.loads(resposta.read().decode("utf-8"))


def extrair_raiz_ticker(ticker: str) -> str:
    """Remove o número final do ticker para casar com o emissor B3."""
    return re.sub(r"\d+$", "", ticker.replace(".SA", ""))


def normalizar_segmento_listagem(valor: object) -> str:
    """Padroniza segmento de listagem para dummies interpretáveis."""
    texto = normalizar_texto(valor)
    if texto in {"NM", "NOVO MERCADO"}:
        return "Novo Mercado"
    if texto in {"N1", "NIVEL 1", "NIVEL1"}:
        return "Nivel 1"
    if texto in {"N2", "NIVEL 2", "NIVEL2"}:
        return "Nivel 2"
    if texto in {"MA", "MB", "BOVESPA MAIS"}:
        return "Bovespa Mais"
    if texto in {"M2", "BOVESPA MAIS NIVEL 2", "BOVESPA MAIS NIVEL II"}:
        return "Bovespa Mais Nivel 2"
    return "Tradicional"


def criar_dummies_segmento(df: pd.DataFrame) -> pd.DataFrame:
    """Cria dummies de governança sem impor ranking ordinal."""
    df = df.copy()
    mapa_colunas = {
        "Novo Mercado": "Segmento_Novo_Mercado",
        "Nivel 1": "Segmento_Nivel_1",
        "Nivel 2": "Segmento_Nivel_2",
        "Bovespa Mais": "Segmento_Bovespa_Mais",
        "Bovespa Mais Nivel 2": "Segmento_Bovespa_Mais_Nivel_2",
        "Tradicional": "Segmento_Tradicional",
    }
    for segmento, coluna in mapa_colunas.items():
        df[coluna] = (df["Segmento_Listagem"] == segmento).astype(int)
    return df


def criar_dummies_setor(df: pd.DataFrame) -> pd.DataFrame:
    """Cria dummies macro de setor para capturar diferenças estruturais entre empresas."""
    df = df.copy()
    setor_normalizado = (
        df["Setor"].fillna("Nao Classificado").map(normalizar_nome_coluna)
    )
    dummies = pd.get_dummies(setor_normalizado, prefix="Setor", dtype=int)
    return pd.concat([df, dummies], axis=1)


def remover_multiindex_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    """Mantido apenas para o fallback do Ibovespa via yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# %% [markdown]
# # ETAPA 1: Engenharia de Dados
#
# Nesta etapa, o notebook coleta preços oficiais B3, fundamentos CVM,
# macroeconomia BCB e informações de setor/governança.

# %% [markdown]
# ## 1.1 Universo B3/CVM
#
# Universo líquido ampliado: carteira atual do IBrA, com `CD_CVM`, CNPJ, setor,
# segmento de listagem e dummies de governança.


# %%
def baixar_carteira_indice_b3(indice: str) -> pd.DataFrame:
    """Baixa a carteira atual do índice escolhido como universo líquido inicial."""
    dados = baixar_json_b3(B3_INDEX_URL, {"index": indice, "language": "pt-br"})
    carteira = pd.DataFrame(dados["results"])
    carteira = carteira.rename(
        columns={
            "cod": "Ticker_B3",
            "asset": "Nome_Pregao",
            "type": "Tipo_Ativo",
            "part": "Participacao_Indice",
        }
    )
    carteira["Ticker"] = carteira["Ticker_B3"] + ".SA"
    carteira["Codigo_Emissor"] = carteira["Ticker_B3"].map(extrair_raiz_ticker)
    return carteira[
        [
            "Ticker",
            "Ticker_B3",
            "Codigo_Emissor",
            "Nome_Pregao",
            "Tipo_Ativo",
            "Participacao_Indice",
        ]
    ]


def baixar_cadastro_empresas_b3() -> pd.DataFrame:
    """Baixa cadastro público da B3 para obter CD_CVM e CNPJ por emissor."""
    paginas = []
    pagina_atual = 1
    while True:
        dados = baixar_json_b3(
            B3_COMPANIES_URL,
            {"language": "pt-br", "pageNumber": pagina_atual, "pageSize": 120},
        )
        resultados = dados.get("results", [])
        if not resultados:
            break
        paginas.append(pd.DataFrame(resultados))
        total_paginas = dados.get("page", {}).get("totalPages") or pagina_atual
        if pagina_atual >= total_paginas:
            break
        pagina_atual += 1

    cadastro = pd.concat(paginas, ignore_index=True)
    cadastro = cadastro.rename(
        columns={
            "codeCVM": "CD_CVM",
            "issuingCompany": "Codigo_Emissor",
            "companyName": "DENOM_CIA",
            "tradingName": "Nome_Negociacao_B3",
            "cnpj": "CNPJ_CIA",
            "segment": "Subsetor_B3_Cadastro",
            "market": "Segmento_Listagem_B3",
        }
    )
    cadastro["CD_CVM"] = pd.to_numeric(cadastro["CD_CVM"], errors="coerce").astype(
        "Int64"
    )
    cadastro["Codigo_Emissor"] = cadastro["Codigo_Emissor"].astype(str).str.strip()
    return cadastro


def baixar_classificacao_setorial_b3() -> pd.DataFrame:
    """Baixa a planilha oficial de classificação setorial da B3."""
    headers = {"User-Agent": "Mozilla/5.0"}
    with urlopen(
        Request(B3_CLASSIFICACAO_SETORIAL_URL, headers=headers), timeout=60
    ) as resposta:
        zip_bytes = resposta.read()

    with ZipFile(BytesIO(zip_bytes)) as arquivo_zip:
        nome_planilha = arquivo_zip.namelist()[0]
        planilha_bytes = arquivo_zip.read(nome_planilha)

    bruto = pd.read_excel(BytesIO(planilha_bytes), header=None, engine="openpyxl")
    registros = []
    setor_atual = None
    subsetor_atual = None
    segmento_atual = None

    for _, linha in bruto.iterrows():
        col0, col1, col2, col3, col4 = linha.iloc[:5]
        if (
            isinstance(col0, str)
            and col0.strip()
            and normalizar_texto(col0) != "SETOR ECONOMICO"
        ):
            setor_atual = col0.strip()
            subsetor_atual = (
                col1.strip() if isinstance(col1, str) and col1.strip() else None
            )
            segmento_atual = (
                col2.strip() if isinstance(col2, str) and col2.strip() else None
            )
            continue
        if (
            isinstance(col1, str)
            and col1.strip()
            and normalizar_texto(col1) != "SUBSETOR"
        ):
            subsetor_atual = col1.strip()
            segmento_atual = (
                col2.strip()
                if isinstance(col2, str) and col2.strip()
                else segmento_atual
            )
            continue
        if isinstance(col2, str) and col2.strip() and pd.isna(col3):
            segmento_atual = col2.strip()
            continue
        if (
            isinstance(col3, str)
            and col3.strip()
            and normalizar_texto(col3) != "CODIGO"
        ):
            registros.append(
                {
                    "Codigo_Emissor": col3.strip(),
                    "Setor": setor_atual,
                    "Subsetor": subsetor_atual,
                    "Segmento_Setorial": segmento_atual,
                    "Segmento_Listagem_Classificacao": (
                        col4.strip() if isinstance(col4, str) and col4.strip() else ""
                    ),
                }
            )

    return pd.DataFrame(registros).drop_duplicates(
        subset=["Codigo_Emissor"], keep="last"
    )


def montar_universo_empresas_b3() -> pd.DataFrame:
    """Monta universo auditável com empresas líquidas da B3."""
    carteira = baixar_carteira_indice_b3(INDICE_UNIVERSO_B3)
    cadastro = baixar_cadastro_empresas_b3()
    classificacao = baixar_classificacao_setorial_b3()

    universo = carteira.merge(cadastro, on="Codigo_Emissor", how="left")
    universo = universo.merge(classificacao, on="Codigo_Emissor", how="left")
    universo["Segmento_Listagem"] = universo["Segmento_Listagem_B3"].combine_first(
        universo["Segmento_Listagem_Classificacao"]
    )
    universo["Segmento_Listagem"] = universo["Segmento_Listagem"].map(
        normalizar_segmento_listagem
    )
    universo["Setor"] = universo["Setor"].fillna("Nao Classificado")
    universo["Subsetor"] = (
        universo["Subsetor"]
        .fillna(universo["Subsetor_B3_Cadastro"])
        .fillna("Nao Classificado")
    )
    universo["Segmento_Setorial"] = universo["Segmento_Setorial"].fillna(
        "Nao Classificado"
    )

    # Controle estatal não é segmento de listagem; mantemos como variável separada.
    tickers_estatais = {
        "BBAS3",
        "PETR3",
        "PETR4",
        "CMIG3",
        "CMIG4",
        "AXIA3",
        "AXIA6",
        "SBSP3",
    }
    universo["Eh_Estatal"] = universo["Ticker_B3"].isin(tickers_estatais).astype(int)

    universo = universo[
        [
            "Ticker",
            "Ticker_B3",
            "Codigo_Emissor",
            "CD_CVM",
            "CNPJ_CIA",
            "DENOM_CIA",
            "Nome_Pregao",
            "Setor",
            "Subsetor",
            "Segmento_Setorial",
            "Segmento_Listagem",
            "Eh_Estatal",
            "Participacao_Indice",
        ]
    ].copy()
    universo = universo.dropna(subset=["Ticker", "CD_CVM"]).drop_duplicates(
        subset=["Ticker"], keep="first"
    )
    universo = criar_dummies_segmento(universo)
    universo = universo.sort_values("Ticker").reset_index(drop=True)
    universo.to_csv(ARQUIVO_UNIVERSO, index=False)
    return universo


if EXECUTAR_ETAPA_1:
    print("Montando universo expandido B3/CVM...")
    df_universo = montar_universo_empresas_b3()
else:
    df_universo = pd.read_csv(ARQUIVO_UNIVERSO)
    print(f"Universo carregado de {ARQUIVO_UNIVERSO.resolve()}")

tickers_fmt = df_universo["Ticker"].dropna().astype(str).unique().tolist()
tickers_b3 = df_universo["Ticker_B3"].dropna().astype(str).unique().tolist()
display_df(df_universo)

# %% [markdown]
# ## 1.2 Preços oficiais B3 COTAHIST
#
# Esta célula troca o preço das ações para a fonte oficial B3. O arquivo
# `COTAHIST` usa layout de largura fixa; por isso fazemos parsing direto das
# posições relevantes.
#
# Observação metodológica: o `COTAHIST` traz preço oficial de negociação. Ele é
# excelente para auditabilidade, mas não representa retorno total com dividendos.


# %%
def baixar_arquivo_cotahist(ano: int) -> Path:
    """Baixa o ZIP anual COTAHIST da B3 para cache local."""
    caminho = DIR_CACHE_COTAHIST / f"COTAHIST_A{ano}.ZIP"
    if caminho.exists() and caminho.stat().st_size > 0:
        return caminho

    url = COTAHIST_URL.format(ano=ano)
    headers = {"User-Agent": "Mozilla/5.0"}
    print(f"Baixando COTAHIST B3 {ano}...")
    with urlopen(Request(url, headers=headers), timeout=180) as resposta:
        caminho.write_bytes(resposta.read())
    return caminho


def inteiro_cotahist(texto: bytes) -> int:
    """Converte campos numéricos inteiros do COTAHIST."""
    valor = texto.decode("latin1", errors="ignore").strip()
    return int(valor or 0)


def preco_cotahist(texto: bytes) -> float:
    """Converte campos de preço/volume monetário do COTAHIST, escalados por 100."""
    return inteiro_cotahist(texto) / 100


def parsear_cotahist_para_tickers(ano: int, tickers_alvo: set[str]) -> pd.DataFrame:
    """Lê apenas os tickers do universo para economizar memória."""
    caminho = baixar_arquivo_cotahist(ano)
    registros = []

    with ZipFile(caminho) as arquivo_zip:
        nomes_txt = [
            nome for nome in arquivo_zip.namelist() if nome.upper().endswith(".TXT")
        ]
        if not nomes_txt:
            raise RuntimeError(f"Nenhum TXT encontrado em {caminho.name}")

        with arquivo_zip.open(nomes_txt[0]) as arquivo:
            for linha in arquivo:
                # Registro 01 = cotação. TPMERC 010 = mercado à vista.
                if linha[0:2] != b"01" or linha[24:27] != b"010":
                    continue

                ticker_b3 = linha[12:24].decode("latin1", errors="ignore").strip()
                if ticker_b3 not in tickers_alvo:
                    continue

                data = pd.to_datetime(
                    linha[2:10].decode("latin1"), format="%Y%m%d", errors="coerce"
                )
                if pd.isna(data):
                    continue

                registros.append(
                    {
                        "Data": data,
                        "Ano": int(data.year),
                        "Ticker_B3": ticker_b3,
                        "Abertura": preco_cotahist(linha[56:69]),
                        "Maxima": preco_cotahist(linha[69:82]),
                        "Minima": preco_cotahist(linha[82:95]),
                        "Media": preco_cotahist(linha[95:108]),
                        "Fechamento": preco_cotahist(linha[108:121]),
                        "Negocios": inteiro_cotahist(linha[147:152]),
                        "Quantidade_Titulos": inteiro_cotahist(linha[152:170]),
                        "Volume_Financeiro": preco_cotahist(linha[170:188]),
                    }
                )

    return pd.DataFrame(registros)


def baixar_ibovespa_yfinance() -> pd.DataFrame:
    """Fallback explícito para o benchmark, mantendo ações na fonte oficial B3."""
    if not PERMITIR_FALLBACK_YFINANCE_IBOV:
        raise RuntimeError("Fallback yfinance para Ibovespa desativado.")

    print("Baixando Ibovespa via yfinance apenas como benchmark/fallback...")
    ibov = yf.download(
        "^BVSP",
        start=DATA_INICIO,
        end=DATA_FIM,
        progress=False,
        auto_adjust=False,
        threads=False,
        timeout=30,
    )
    ibov = remover_multiindex_yfinance(ibov)
    ibov = ibov[ibov["Close"].notna()].copy()
    ibov["Retorno_Ibov_Diario"] = ibov["Close"].pct_change(fill_method=None)
    ibov["Ano"] = ibov.index.year
    ibov["Data"] = pd.to_datetime(ibov.index).tz_localize(None)
    return ibov[["Data", "Ano", "Close", "Retorno_Ibov_Diario"]].rename(
        columns={"Close": "Fechamento_Ibov"}
    )


def montar_precos_b3(
    df_universo_base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consolida preços oficiais B3 e calcula target anual."""
    tickers_alvo = set(df_universo_base["Ticker_B3"].dropna().astype(str))
    dados_diarios = []
    falhas = []

    for ano in sorted(ANOS_ESPERADOS_TCC):
        try:
            print(f"Processando COTAHIST {ano}...")
            df_ano = parsear_cotahist_para_tickers(ano, tickers_alvo)
            if df_ano.empty:
                raise ValueError("Nenhum ticker do universo encontrado no COTAHIST.")
            dados_diarios.append(df_ano)
        except Exception as erro:
            falhas.append((str(ano), str(erro)))

    reportar_falhas("COTAHIST B3", falhas)
    if not dados_diarios:
        raise RuntimeError("Nenhuma cotação B3 foi consolidada.")

    df_diario = pd.concat(dados_diarios, ignore_index=True)
    df_diario = df_diario.merge(
        df_universo_base[["Ticker", "Ticker_B3"]], on="Ticker_B3", how="inner"
    )
    df_diario = df_diario.sort_values(["Ticker", "Data"]).reset_index(drop=True)
    df_diario["Retorno_Diario"] = df_diario.groupby("Ticker")["Fechamento"].pct_change(
        fill_method=None
    )
    df_diario["Amplitude_Intradiaria"] = divisao_segura(
        df_diario["Maxima"] - df_diario["Minima"],
        df_diario["Fechamento"],
    )

    ibov_diario = baixar_ibovespa_yfinance()
    df_diario = df_diario.merge(
        ibov_diario[["Data", "Retorno_Ibov_Diario"]], on="Data", how="left"
    )

    registros_anuais = []
    for (ticker, ano), grupo in df_diario.groupby(["Ticker", "Ano"]):
        grupo = grupo.sort_values("Data")
        pregoes = int(grupo["Fechamento"].notna().sum())
        if pregoes < MIN_PREGOES_ANO:
            continue
        registros_anuais.append(
            {
                "Ticker": ticker,
                "Ano": int(ano),
                "Pregoes_Ano": pregoes,
                "Volatilidade_Anual": grupo["Retorno_Diario"].std() * np.sqrt(252),
                "Retorno_Anual": retorno_composto_anual(grupo["Retorno_Diario"]),
                "Volume_Financeiro_Medio_Diario": grupo["Volume_Financeiro"].mean(),
                "Negocios_Medio_Diario": grupo["Negocios"].mean(),
                "Liquidez_Anual_Log": np.log1p(grupo["Volume_Financeiro"].sum()),
                "Amplitude_Intradiaria_Media": grupo["Amplitude_Intradiaria"].mean(),
                "Max_Drawdown_Anual": max_drawdown(grupo["Fechamento"]),
                "Beta_Ibov_Anual": beta_contra_benchmark(
                    grupo["Retorno_Diario"], grupo["Retorno_Ibov_Diario"]
                ),
            }
        )

    df_precos = pd.DataFrame(registros_anuais)
    if df_precos.empty:
        raise RuntimeError("Nenhum ano/ticker atingiu o mínimo de pregões na B3.")

    anos_validos_por_ticker = df_precos.groupby("Ticker")["Ano"].nunique()
    tickers_suficientes = anos_validos_por_ticker[
        anos_validos_por_ticker >= MIN_ANOS_PRECO_VALIDO
    ].index
    tickers_removidos_preco = sorted(
        set(df_precos["Ticker"]) - set(tickers_suficientes)
    )
    df_precos = df_precos[df_precos["Ticker"].isin(tickers_suficientes)].copy()

    ibov_anual = (
        ibov_diario.groupby("Ano")["Retorno_Ibov_Diario"]
        .apply(retorno_composto_anual)
        .rename("Retorno_Ibov")
        .reset_index()
    )
    df_precos = df_precos.merge(ibov_anual, on="Ano", how="left")
    df_precos["Target_Venceu"] = (
        df_precos["Retorno_Anual"] > df_precos["Retorno_Ibov"]
    ).astype(int)
    df_precos = df_precos.sort_values(["Ticker", "Ano"]).reset_index(drop=True)
    df_precos["Retorno_Ano_Anterior"] = df_precos.groupby("Ticker")[
        "Retorno_Anual"
    ].shift(1)

    df_tickers_removidos = pd.DataFrame(
        [
            {
                "Ticker": ticker,
                "Motivo": f"Menos de {MIN_ANOS_PRECO_VALIDO} anos com preço oficial B3 válido",
            }
            for ticker in tickers_removidos_preco
        ],
        columns=["Ticker", "Motivo"],
    )

    return df_precos, df_tickers_removidos


if EXECUTAR_ETAPA_1:
    print("Iniciando preços oficiais B3 e target anual...")
    df_precos, df_tickers_removidos = montar_precos_b3(df_universo)
    display_df(df_precos)
else:
    print("Etapa 1 desativada. O notebook usará os CSVs existentes.")

# %% [markdown]
# ## 1.3 Fundamentos CVM DFP
#
# A CVM DFP fornece os fundamentos contábeis usados para calcular rentabilidade,
# alavancagem, F-Score adaptado e variáveis adicionais simples.


# %%
def baixar_zip_dfp_cvm(ano: int) -> ZipFile:
    """Baixa ZIP anual da CVM DFP com cache local."""
    caminho = DIR_CACHE_CVM / f"dfp_cia_aberta_{ano}.zip"
    if not caminho.exists() or caminho.stat().st_size == 0:
        url = CVM_DFP_URL.format(ano=ano)
        print(f"Baixando DFP CVM {ano}...")
        with urlopen(url, timeout=120) as resposta:
            caminho.write_bytes(resposta.read())
    return ZipFile(caminho)


def ler_arquivo_dfp(zip_dfp: ZipFile, demonstrativo: str, ano: int) -> pd.DataFrame:
    """Lê demonstrativo consolidado da CVM dentro do ZIP anual."""
    nome_arquivo = f"dfp_cia_aberta_{demonstrativo}_con_{ano}.csv"
    with zip_dfp.open(nome_arquivo) as arquivo:
        df = pd.read_csv(arquivo, sep=";", encoding="ISO-8859-1")
        df["CD_CVM"] = pd.to_numeric(df["CD_CVM"], errors="coerce").astype("Int64")
        df["VL_CONTA"] = pd.to_numeric(df["VL_CONTA"], errors="coerce")
        df["Ano"] = pd.to_datetime(df["DT_FIM_EXERC"], errors="coerce").dt.year
        df["ORDEM_NORMALIZADA"] = df["ORDEM_EXERC"].map(normalizar_texto)
        df = df[(df["ORDEM_NORMALIZADA"] == "ULTIMO") & (df["Ano"] == ano)].copy()
        df = df.sort_values("VERSAO")
        df = df.drop_duplicates(subset=["CD_CVM", "Ano", "CD_CONTA"], keep="last")
    return df


def valor_por_codigo(df: pd.DataFrame, codigo: str) -> pd.Series:
    """Extrai conta CVM por código exato."""
    conta = df[df["CD_CONTA"].astype(str) == codigo]
    return conta.set_index(["CD_CVM", "Ano"])["VL_CONTA"]


def soma_por_codigos(df: pd.DataFrame, codigos: list[str]) -> pd.Series:
    """Soma contas CVM por código exato."""
    conta = df[df["CD_CONTA"].astype(str).isin(codigos)]
    if conta.empty:
        return pd.Series(dtype="float64")
    return conta.groupby(["CD_CVM", "Ano"])["VL_CONTA"].sum()


def primeira_conta_disponivel(df: pd.DataFrame, codigos: list[str]) -> pd.Series:
    """Usa a primeira alternativa de conta que exista no demonstrativo."""
    for codigo in codigos:
        serie = valor_por_codigo(df, codigo)
        if not serie.empty:
            return serie
    return pd.Series(dtype="float64")


def montar_fundamentos_cvm_para_ano(ano: int, codigos_cvm: set[int]) -> pd.DataFrame:
    """Transforma DFP CVM em indicadores fundamentalistas anuais."""
    zip_dfp = baixar_zip_dfp_cvm(ano)
    dre = ler_arquivo_dfp(zip_dfp, "DRE", ano)
    bpa = ler_arquivo_dfp(zip_dfp, "BPA", ano)
    bpp = ler_arquivo_dfp(zip_dfp, "BPP", ano)
    dfc = ler_arquivo_dfp(zip_dfp, "DFC_MI", ano)

    dre = dre[dre["CD_CVM"].isin(codigos_cvm)]
    bpa = bpa[bpa["CD_CVM"].isin(codigos_cvm)]
    bpp = bpp[bpp["CD_CVM"].isin(codigos_cvm)]
    dfc = dfc[dfc["CD_CVM"].isin(codigos_cvm)]

    indice = pd.MultiIndex.from_frame(
        pd.DataFrame({"CD_CVM": sorted(codigos_cvm), "Ano": ano})
    )
    fundamentos = pd.DataFrame(index=indice)

    fundamentos["Receita"] = valor_por_codigo(dre, "3.01")
    fundamentos["Lucro_Bruto"] = valor_por_codigo(dre, "3.03")
    fundamentos["EBIT"] = valor_por_codigo(dre, "3.05")
    fundamentos["Lucro"] = primeira_conta_disponivel(dre, ["3.11.01", "3.11"])
    fundamentos["Ativo_Total"] = valor_por_codigo(bpa, "1")
    fundamentos["Caixa"] = valor_por_codigo(bpa, "1.01.01")
    fundamentos["Patrimonio"] = valor_por_codigo(bpp, "2.03")
    fundamentos["Divida"] = soma_por_codigos(bpp, ["2.01.04", "2.02.01"])
    fundamentos["Caixa_Operacional"] = valor_por_codigo(dfc, "6.01")
    fundamentos["CAPEX"] = soma_por_codigos(dfc, ["6.02.02", "6.02.03"])

    fundamentos = fundamentos.reset_index()
    fundamentos["Ano"] = ano

    capital_investido = fundamentos["Divida"] + fundamentos["Patrimonio"]
    fundamentos["ROIC"] = divisao_segura(
        fundamentos["EBIT"] * (1 - ALIQUOTA_IR), capital_investido
    )
    fundamentos["DivLiq_EBITDA"] = divisao_segura(
        fundamentos["Divida"] - fundamentos["Caixa"], fundamentos["EBIT"]
    )
    fundamentos["ROE"] = divisao_segura(fundamentos["Lucro"], fundamentos["Patrimonio"])
    fundamentos["ROA"] = divisao_segura(
        fundamentos["Lucro"], fundamentos["Ativo_Total"]
    )
    fundamentos["Alavancagem_Ativo"] = divisao_segura(
        fundamentos["Divida"], fundamentos["Ativo_Total"]
    )
    fundamentos["Margem_Bruta"] = divisao_segura(
        fundamentos["Lucro_Bruto"], fundamentos["Receita"]
    )
    fundamentos["Giro_Ativo"] = divisao_segura(
        fundamentos["Receita"], fundamentos["Ativo_Total"]
    )
    fundamentos["FCF"] = fundamentos["Caixa_Operacional"] + fundamentos["CAPEX"]

    # Variáveis adicionais leves para enriquecer a base.
    fundamentos["Log_Ativo_Total"] = np.where(
        fundamentos["Ativo_Total"] > 0,
        np.log1p(fundamentos["Ativo_Total"]),
        np.nan,
    )
    fundamentos["Margem_Liquida"] = divisao_segura(
        fundamentos["Lucro"], fundamentos["Receita"]
    )
    fundamentos["Divida_Ativo"] = divisao_segura(
        fundamentos["Divida"], fundamentos["Ativo_Total"]
    )

    return fundamentos


def montar_fundamentos_cvm(df_universo_base: pd.DataFrame) -> pd.DataFrame:
    """Consolida fundamentos CVM para todos os anos do estudo."""
    df_mapa_cvm = df_universo_base.copy()
    df_mapa_cvm["CD_CVM"] = pd.to_numeric(
        df_mapa_cvm["CD_CVM"], errors="coerce"
    ).astype("Int64")
    codigos_cvm = set(df_mapa_cvm["CD_CVM"].dropna().astype(int))
    dados_fundamentos = []
    falhas_cvm = []

    for ano in sorted(ANOS_ESPERADOS_TCC):
        try:
            print(f"Processando DFP CVM {ano}...")
            dados_fundamentos.append(montar_fundamentos_cvm_para_ano(ano, codigos_cvm))
        except Exception as erro:
            falhas_cvm.append((str(ano), str(erro)))

    reportar_falhas("CVM DFP", falhas_cvm)
    if not dados_fundamentos:
        raise RuntimeError("Nenhum demonstrativo CVM foi consolidado.")

    df_fund = pd.concat(dados_fundamentos, ignore_index=True)
    df_fund = df_fund.merge(df_mapa_cvm, on="CD_CVM", how="inner")
    df_fund = df_fund.sort_values(["Ticker", "Ano"]).reset_index(drop=True)

    df_fund["ROA_Anterior"] = df_fund.groupby("Ticker")["ROA"].shift(1)
    df_fund["Alavancagem_Anterior"] = df_fund.groupby("Ticker")[
        "Alavancagem_Ativo"
    ].shift(1)
    df_fund["Margem_Anterior"] = df_fund.groupby("Ticker")["Margem_Bruta"].shift(1)
    df_fund["Giro_Anterior"] = df_fund.groupby("Ticker")["Giro_Ativo"].shift(1)
    df_fund["Receita_Anterior"] = df_fund.groupby("Ticker")["Receita"].shift(1)
    df_fund["Crescimento_Receita"] = divisao_segura(
        df_fund["Receita"] - df_fund["Receita_Anterior"],
        df_fund["Receita_Anterior"].abs(),
    )

    # F-Score adaptado: rentabilidade, caixa, alavancagem e eficiência.
    df_fund["F_Score"] = (
        (df_fund["ROA"] > 0).astype(int)
        + (df_fund["FCF"] > 0).astype(int)
        + (df_fund["FCF"] > df_fund["Lucro"]).astype(int)
        + (df_fund["ROA"] > df_fund["ROA_Anterior"]).astype(int)
        + (df_fund["Alavancagem_Ativo"] < df_fund["Alavancagem_Anterior"]).astype(int)
        + (df_fund["Margem_Bruta"] > df_fund["Margem_Anterior"]).astype(int)
        + (df_fund["Giro_Ativo"] > df_fund["Giro_Anterior"]).astype(int)
    )

    colunas_fund = [
        "Ticker",
        "CD_CVM",
        "CNPJ_CIA",
        "DENOM_CIA",
        "Nome_Pregao",
        "Setor",
        "Subsetor",
        "Segmento_Setorial",
        "Segmento_Listagem",
        "Ano",
        "ROIC",
        "F_Score",
        "DivLiq_EBITDA",
        "ROE",
        "FCF",
        "ROA",
        "Margem_Bruta",
        "Giro_Ativo",
        "Log_Ativo_Total",
        "Crescimento_Receita",
        "Margem_Liquida",
        "Divida_Ativo",
        "Eh_Estatal",
        *SEGMENTOS_LISTAGEM_ESPERADOS,
    ]
    return df_fund[colunas_fund].copy()


if EXECUTAR_ETAPA_1:
    print("Iniciando fundamentos oficiais CVM DFP...")
    df_fund = montar_fundamentos_cvm(df_universo)
    display_df(df_fund)

# %% [markdown]
# ## 1.4 Macroeconomia BCB SGS
#
# Selic média anual e IPCA acumulado anual continuam vindo do BCB SGS.


# %%
def baixar_serie_bcb(serie: int, data_inicial: str, data_final: str) -> pd.DataFrame:
    """Baixa uma série temporal do SGS/BCB em JSON."""
    parametros = urlencode(
        {"formato": "json", "dataInicial": data_inicial, "dataFinal": data_final}
    )
    url = f"{BCB_SGS_URL.format(serie=serie)}?{parametros}"
    ultimo_erro = None
    for tentativa in range(3):
        try:
            with urlopen(url, timeout=60) as resposta:
                df = pd.read_json(BytesIO(resposta.read()))
                break
        except Exception as erro:
            ultimo_erro = erro
            sleep(1 + tentativa)
    else:
        raise RuntimeError(
            f"Falha ao baixar série SGS {serie}: {ultimo_erro}"
        ) from ultimo_erro

    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["Ano"] = df["data"].dt.year
    return df


def montar_macro_bcb() -> pd.DataFrame:
    """Calcula Selic média anual e IPCA acumulado anual."""
    series_selic = []
    series_ipca = []
    for ano in sorted(ANOS_ESPERADOS_TCC):
        data_inicial = f"01/01/{ano}"
        data_final = f"31/12/{ano}"
        series_selic.append(
            baixar_serie_bcb(SERIE_SELIC_META, data_inicial, data_final)
        )
        series_ipca.append(
            baixar_serie_bcb(SERIE_IPCA_MENSAL, data_inicial, data_final)
        )

    selic = pd.concat(series_selic, ignore_index=True)
    ipca = pd.concat(series_ipca, ignore_index=True)
    df_selic = (
        selic.groupby("Ano")["valor"]
        .mean()
        .div(100)
        .rename("Selic_Media")
        .reset_index()
    )
    df_ipca = (
        ipca.assign(fator=1 + ipca["valor"] / 100)
        .groupby("Ano")["fator"]
        .prod()
        .sub(1)
        .rename("IPCA_Anual")
        .reset_index()
    )
    return df_selic.merge(df_ipca, on="Ano", how="inner")


# %% [markdown]
# ## 1.5 Consolidação da Base Final
#
# Aqui juntamos preços B3, fundamentos CVM, macro BCB, governança e setor.


# %%
def montar_lista_features(df_base: pd.DataFrame) -> list[str]:
    """Lista única de features, mantendo só colunas existentes."""
    setor_cols = sorted([col for col in df_base.columns if col.startswith("Setor_")])
    features = [
        "Volatilidade_Anual",
        "F_Score",
        "DivLiq_EBITDA",
        "ROE_Premio_Risco",
        "ROIC_Premio_Risco",
        "Custo_Divida_Real",
        "IPCA_Anual",
        "Volume_Financeiro_Medio_Diario",
        "Negocios_Medio_Diario",
        "Liquidez_Anual_Log",
        "Amplitude_Intradiaria_Media",
        "Max_Drawdown_Anual",
        "Beta_Ibov_Anual",
        "Retorno_Ano_Anterior",
        "Log_Ativo_Total",
        "Crescimento_Receita",
        "Margem_Liquida",
        "Divida_Ativo",
        "Eh_Estatal",
        *SEGMENTOS_LISTAGEM_ESPERADOS,
        *setor_cols,
    ]
    return [feature for feature in features if feature in df_base.columns]


if EXECUTAR_ETAPA_1:
    print("Baixando macroeconomia BCB e consolidando base final...")
    df_macro = montar_macro_bcb()

    df_final_pre_dropna = df_precos.merge(df_fund, on=["Ticker", "Ano"], how="inner")
    df_final = df_final_pre_dropna.merge(df_macro, on="Ano", how="left")

    df_final["ROE_Premio_Risco"] = df_final["ROE"] - df_final["Selic_Media"]
    df_final["ROIC_Premio_Risco"] = df_final["ROIC"] - df_final["Selic_Media"]
    df_final["ROE_Ajustado_Selic"] = divisao_segura(
        df_final["ROE"], df_final["Selic_Media"]
    )
    df_final["Custo_Divida_Real"] = df_final["DivLiq_EBITDA"] * df_final["Selic_Media"]
    df_final = criar_dummies_setor(df_final)

    features = montar_lista_features(df_final)

    # A base contemporânea precisa preservar 2010-2025. Algumas features novas,
    # como Retorno_Ano_Anterior e Crescimento_Receita, nascem naturalmente sem
    # valor no primeiro ano de cada empresa. Por isso, o ETL exige apenas o
    # núcleo metodológico essencial; a Etapa 2 preenche missings das features
    # pela mediana, mantendo o painel auditável sem criar anos artificiais.
    colunas_obrigatorias = [
        "Ticker",
        "Ano",
        "Volatilidade_Anual",
        "Retorno_Anual",
        "Retorno_Ibov",
        "Target_Venceu",
        "F_Score",
        "DivLiq_EBITDA",
        "ROE",
        "ROIC",
        "ROE_Premio_Risco",
        "ROIC_Premio_Risco",
        "Custo_Divida_Real",
        "Selic_Media",
        "IPCA_Anual",
        "Eh_Estatal",
    ]

    tamanho_pre_dropna = len(df_final)
    df_final = df_final.replace([np.inf, -np.inf], np.nan)
    df_final = df_final.dropna(subset=colunas_obrigatorias)
    df_final = df_final.sort_values(["Ano", "Ticker"]).reset_index(drop=True)

    diagnostico_etl = (
        df_precos.groupby("Ano")
        .size()
        .rename("Linhas_Preco_B3")
        .reset_index()
        .merge(
            df_final_pre_dropna.groupby("Ano")
            .size()
            .rename("Linhas_Pos_Merge_CVM")
            .reset_index(),
            on="Ano",
            how="left",
        )
        .merge(
            df_final.groupby("Ano").size().rename("Linhas_Validas").reset_index(),
            on="Ano",
            how="left",
        )
    )
    diagnostico_etl[["Linhas_Pos_Merge_CVM", "Linhas_Validas"]] = (
        diagnostico_etl[["Linhas_Pos_Merge_CVM", "Linhas_Validas"]]
        .fillna(0)
        .astype(int)
    )
    diagnostico_etl["Cobertura_Tickers_%"] = (
        diagnostico_etl["Linhas_Validas"] / len(tickers_fmt) * 100
    ).round(2)
    diagnostico_etl.to_csv(ARQUIVO_DIAGNOSTICO_ETL, index=False)

    diagnostico_setor = (
        df_final.groupby("Setor")
        .agg(Tickers=("Ticker", "nunique"), Observacoes=("Ticker", "size"))
        .sort_values(["Observacoes", "Tickers"], ascending=False)
        .reset_index()
    )
    diagnostico_setor.to_csv(ARQUIVO_DIAGNOSTICO_SETOR, index=False)

    diagnostico_segmento = (
        df_final.groupby("Segmento_Listagem")
        .agg(Tickers=("Ticker", "nunique"), Observacoes=("Ticker", "size"))
        .sort_values(["Observacoes", "Tickers"], ascending=False)
        .reset_index()
    )
    diagnostico_segmento.to_csv(ARQUIVO_DIAGNOSTICO_SEGMENTO, index=False)

    tickers_sem_fundamento_valido = sorted(
        set(df_precos["Ticker"]) - set(df_final["Ticker"])
    )
    if tickers_sem_fundamento_valido:
        df_removidos_fund = pd.DataFrame(
            {
                "Ticker": tickers_sem_fundamento_valido,
                "Motivo": "Sem fundamentos CVM válidos após filtros do modelo",
            }
        )
        df_tickers_removidos = pd.concat(
            [df_tickers_removidos, df_removidos_fund], ignore_index=True
        )

    df_tickers_removidos = df_tickers_removidos.drop_duplicates().sort_values(
        ["Motivo", "Ticker"]
    )
    df_tickers_removidos.to_csv(ARQUIVO_TICKERS_REMOVIDOS, index=False)

    print(f"Base consolidada com {len(df_final)} linhas válidas.")
    print(f"Linhas removidas por nulos/infinitos: {tamanho_pre_dropna - len(df_final)}")
    print("Diagnóstico por ano:")
    display(diagnostico_etl)
    print("Diagnóstico por setor:")
    display(diagnostico_setor)
    print("Diagnóstico por segmento:")
    display(diagnostico_segmento)

    anos_disponiveis_etl = set(df_final["Ano"].dropna().astype(int).unique())
    anos_ausentes_etl = sorted(ANOS_ESPERADOS_TCC - anos_disponiveis_etl)
    minimo_observacoes_tcc = max(
        MIN_OBSERVACOES_BASE, int(len(tickers_fmt) * len(ANOS_ESPERADOS_TCC) * 0.30)
    )

    if VALIDAR_PERIODO_TCC and anos_ausentes_etl:
        raise ValueError(
            f"A base não cobre todo o período 2010-2025. Anos ausentes: {anos_ausentes_etl}."
        )
    if VALIDAR_PERIODO_TCC and len(df_final) < minimo_observacoes_tcc:
        raise ValueError(
            f"A base ficou com {len(df_final)} observações, abaixo do mínimo proporcional {minimo_observacoes_tcc}."
        )
    if VALIDAR_PERIODO_TCC and df_final["F_Score"].nunique(dropna=True) <= 1:
        raise ValueError("F_Score ficou sem variação; verifique a leitura CVM.")

    df_final.to_csv(ARQUIVO_BASE, index=False)
else:
    df_final = pd.read_csv(ARQUIVO_BASE)
    features = montar_lista_features(df_final)
    print(f"Dataset carregado de {ARQUIVO_BASE.resolve()}")

display_df(df_final)

# %% [markdown]
# # ETAPA 2: Machine Learning
#
# Nesta etapa treinamos e avaliamos os quatro modelos definidos na metodologia:
# Regressão Logística, Random Forest, XGBoost e MLP.

# %% [markdown]
# ## 2.1 Funções de treino e avaliação
#
# A seleção estatística univariada é feita apenas no conjunto de treino. Isso
# reduz risco de overfitting e evita escolher variáveis olhando o período de
# teste.


# %%
def feature_binaria(serie: pd.Series) -> bool:
    """Identifica dummies/binárias para aplicar teste qui-quadrado."""
    valores = set(pd.Series(serie).dropna().unique())
    return len(valores) <= 2 and valores.issubset({0, 1, 0.0, 1.0, False, True})


def testar_significancia_feature(
    feature: str, X_train: pd.DataFrame, y_train: pd.Series
) -> dict:
    """Testa associação univariada entre uma feature e o target usando apenas treino."""
    serie = X_train[feature]
    grupo_0 = serie[y_train == 0].dropna()
    grupo_1 = serie[y_train == 1].dropna()
    resultado = {
        "Variável": feature,
        "Teste": "",
        "P_Valor": np.nan,
        "Media_Classe_0": grupo_0.mean() if len(grupo_0) else np.nan,
        "Media_Classe_1": grupo_1.mean() if len(grupo_1) else np.nan,
        "Diferenca_Medias_Abs": np.nan,
    }

    if len(grupo_0) == 0 or len(grupo_1) == 0 or serie.nunique(dropna=True) <= 1:
        resultado["Teste"] = "Sem variação suficiente"
        return resultado

    resultado["Diferenca_Medias_Abs"] = abs(
        resultado["Media_Classe_1"] - resultado["Media_Classe_0"]
    )

    try:
        if feature_binaria(serie):
            tabela = pd.crosstab(serie, y_train)
            if tabela.shape[0] < 2 or tabela.shape[1] < 2:
                resultado["Teste"] = "Qui-quadrado sem tabela 2x2"
                return resultado
            _, p_valor, _, _ = chi2_contingency(tabela)
            resultado["Teste"] = "Qui-quadrado"
            resultado["P_Valor"] = float(p_valor)
        else:
            _, p_valor = mannwhitneyu(grupo_0, grupo_1, alternative="two-sided")
            resultado["Teste"] = "Mann-Whitney U"
            resultado["P_Valor"] = float(p_valor)
    except Exception as erro:
        resultado["Teste"] = f"Falha no teste: {erro}"

    return resultado


def selecionar_features_significantes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: list[str],
    nome_base: str,
) -> tuple[list[str], pd.DataFrame]:
    """Seleciona features estatisticamente significantes sem olhar o teste."""
    registros = [
        testar_significancia_feature(feature, X_train, y_train) for feature in features
    ]
    df_significancia = pd.DataFrame(registros)
    df_significancia["Base"] = nome_base
    df_significancia["Alpha"] = ALPHA_SIGNIFICANCIA_FEATURES
    df_significancia["Selecionada"] = (
        df_significancia["P_Valor"] <= ALPHA_SIGNIFICANCIA_FEATURES
    )
    df_significancia["Motivo_Selecao"] = np.where(
        df_significancia["Selecionada"],
        "p-valor <= alpha no treino",
        "Não selecionada pelo filtro estatístico",
    )

    selecionadas = (
        df_significancia.loc[df_significancia["Selecionada"], "Variável"]
        .dropna()
        .astype(str)
        .tolist()
    )

    # Proteção operacional: se o alpha for restritivo demais, mantemos as menores
    # p-values para não quebrar o treino, deixando isso marcado no diagnóstico.
    if len(selecionadas) < MIN_FEATURES_APOS_FILTRO:
        candidatas = (
            df_significancia.dropna(subset=["P_Valor"])
            .sort_values("P_Valor")
            .head(MIN_FEATURES_APOS_FILTRO)["Variável"]
            .astype(str)
            .tolist()
        )
        selecionadas = list(dict.fromkeys([*selecionadas, *candidatas]))
        df_significancia.loc[
            df_significancia["Variável"].isin(candidatas), "Selecionada"
        ] = True
        df_significancia.loc[
            df_significancia["Variável"].isin(candidatas), "Motivo_Selecao"
        ] = "Fallback: menor p-valor para preservar mínimo de features"

    df_significancia = df_significancia.sort_values(
        ["Selecionada", "P_Valor", "Diferenca_Medias_Abs"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    return selecionadas, df_significancia


def corte_temporal_valido(df_base: pd.DataFrame, ano_corte: int) -> bool:
    """Verifica anos mínimos e duas classes em treino/teste."""
    treino = df_base.loc[df_base["Ano"] <= ano_corte, "Target_Venceu"]
    teste = df_base.loc[df_base["Ano"] > ano_corte, "Target_Venceu"]
    anos_treino = df_base.loc[df_base["Ano"] <= ano_corte, "Ano"].nunique()
    anos_teste = df_base.loc[df_base["Ano"] > ano_corte, "Ano"].nunique()
    return (
        anos_treino >= MIN_ANOS_TREINO
        and anos_teste >= MIN_ANOS_TESTE
        and len(treino) > 0
        and len(teste) > 0
        and treino.nunique() == 2
        and teste.nunique() == 2
    )


def escolher_corte_temporal(df_base: pd.DataFrame) -> int:
    """Escolhe o maior corte temporal válido, sem olhar métricas dos modelos."""
    anos_disponiveis = sorted(df_base["Ano"].dropna().unique())
    candidatos = [
        ano for ano in anos_disponiveis[:-1] if corte_temporal_valido(df_base, int(ano))
    ]
    if not candidatos:
        raise ValueError("Não foi possível encontrar corte temporal válido.")
    return int(max(candidatos))


def criar_modelos(X_train_scaled: np.ndarray, y_train: pd.Series) -> dict:
    """Cria os quatro modelos avaliados no estudo."""
    modelos = {
        "Regressão Logística": (
            LogisticRegression(max_iter=2000, random_state=42),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
            ),
            False,
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                eval_metric="logloss",
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            ),
            False,
        ),
    }

    print("Otimizando MLP com TimeSeriesSplit...")
    mlp_base = MLPClassifier(max_iter=1500, random_state=42, early_stopping=False)
    if len(X_train_scaled) < 15 or y_train.value_counts().min() < 3:
        print("Base pequena para GridSearch robusto. Usando MLP padrão.")
        modelos["Deep Learning (MLP)"] = (mlp_base, True)
    else:
        tscv = TimeSeriesSplit(n_splits=3 if len(X_train_scaled) >= 40 else 2)
        mlp_param_grid = {
            "hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
        }
        grid_mlp = GridSearchCV(
            mlp_base, mlp_param_grid, cv=tscv, scoring="roc_auc", n_jobs=1
        )
        grid_mlp.fit(X_train_scaled, y_train)
        modelos["Deep Learning (MLP)"] = (grid_mlp.best_estimator_, True)
        print(f"Melhor configuração MLP: {grid_mlp.best_params_}")
    return modelos


def avaliar_base_ml(
    df_base: pd.DataFrame, nome_base: str, salvar_artefatos_principais: bool = False
) -> dict:
    """Treina e avalia os quatro modelos para uma base específica."""
    features = montar_lista_features(df_base)
    if not features:
        raise ValueError(f"{nome_base}: nenhuma feature válida foi encontrada.")

    df_modelo = df_base.sort_values(["Ano", "Ticker"]).reset_index(drop=True).copy()
    X = df_modelo[features].replace([np.inf, -np.inf], np.nan)
    y = df_modelo["Target_Venceu"].copy()

    print(f"\n=== {nome_base} | Distribuição do target por ano ===")
    display(pd.crosstab(df_modelo["Ano"], df_modelo["Target_Venceu"], margins=True))

    anos_disponiveis = set(df_modelo["Ano"].dropna().astype(int).unique())
    anos_ausentes = sorted(ANOS_ESPERADOS_TCC - anos_disponiveis)
    if VALIDAR_PERIODO_TCC and anos_ausentes:
        raise ValueError(f"{nome_base}: anos ausentes na base: {anos_ausentes}.")

    ano_corte_usado = escolher_corte_temporal(df_modelo)
    treino_idx = df_modelo["Ano"] <= ano_corte_usado
    teste_idx = df_modelo["Ano"] > ano_corte_usado
    X_train = X.loc[treino_idx].reset_index(drop=True)
    y_train = y.loc[treino_idx].reset_index(drop=True)
    X_test = X.loc[teste_idx].reset_index(drop=True)
    y_test = y.loc[teste_idx].reset_index(drop=True)

    # Imputação sem vazamento: medianas calculadas no treino e aplicadas ao teste.
    medianas_treino = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medianas_treino).fillna(0)
    X_test = X_test.fillna(medianas_treino).fillna(0)

    print(f"{nome_base} | Corte temporal usado: {ano_corte_usado}")
    print(
        f"{nome_base} | Treino: {len(X_train)} obs | Classe positiva: {y_train.mean():.2%}"
    )
    print(
        f"{nome_base} | Teste: {len(X_test)} obs | Classe positiva: {y_test.mean():.2%}"
    )

    if APLICAR_FILTRO_SIGNIFICANCIA:
        features_originais = features.copy()
        features, df_significancia = selecionar_features_significantes(
            X_train, y_train, features, nome_base
        )
        X_train = X_train[features].copy()
        X_test = X_test[features].copy()
        print(
            f"{nome_base} | Filtro estatístico: "
            f"{len(features)} de {len(features_originais)} features selecionadas "
            f"(alpha={ALPHA_SIGNIFICANCIA_FEATURES})."
        )
        display(df_significancia.head(20))
    else:
        df_significancia = pd.DataFrame(
            {
                "Base": nome_base,
                "Variável": features,
                "Selecionada": True,
                "Motivo_Selecao": "Filtro estatístico desativado",
            }
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    modelos = criar_modelos(X_train_scaled, y_train)

    resultados = []
    probabilidades = {}
    predicoes = {}
    modelos_treinados = {}

    print(f"\n=== {nome_base} | RELATÓRIO DE PERFORMANCE ===")
    for nome_modelo, (modelo, usar_dados_padronizados) in modelos.items():
        X_fit = X_train_scaled if usar_dados_padronizados else X_train
        X_eval = X_test_scaled if usar_dados_padronizados else X_test
        modelo.fit(X_fit, y_train)
        y_prob = modelo.predict_proba(X_eval)[:, 1]
        y_pred = modelo.predict(X_eval)

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        precision_1, recall_1, f1_1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=[1],
            average=None,
            zero_division=0,
        )

        probabilidades[nome_modelo] = y_prob
        predicoes[nome_modelo] = y_pred
        modelos_treinados[nome_modelo] = modelo

        print(f"\n--- {nome_base} | {nome_modelo} ---")
        print(f"ROC-AUC: {auc:.4f} | Acurácia: {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        resultados.append(
            {
                "Base": nome_base,
                "Modelo": nome_modelo,
                "Acurácia": acc,
                "ROC-AUC": auc,
                "Precision Classe 1": float(precision_1[0]),
                "Recall Classe 1": float(recall_1[0]),
                "F1 Classe 1": float(f1_1[0]),
                "Ano_Corte": ano_corte_usado,
                "Observacoes_Treino": len(X_train),
                "Observacoes_Teste": len(X_test),
                "Features_Selecionadas": len(features),
            }
        )

    df_resultados = pd.DataFrame(resultados).sort_values(by="ROC-AUC", ascending=False)
    display(df_resultados)

    n_bootstraps = 1000
    intervalos_bootstrap = []
    y_test_np = y_test.to_numpy()
    for nome_modelo, y_prob in probabilidades.items():
        scores = []
        for seed in range(n_bootstraps):
            indices = resample(np.arange(len(y_test_np)), random_state=seed)
            if len(np.unique(y_test_np[indices])) < 2:
                continue
            scores.append(roc_auc_score(y_test_np[indices], y_prob[indices]))

        if scores:
            scores = np.sort(np.array(scores))
            intervalos_bootstrap.append(
                {
                    "Base": nome_base,
                    "Modelo": nome_modelo,
                    "IC_95_Inferior": scores[int(0.025 * len(scores))],
                    "IC_95_Superior": scores[int(0.975 * len(scores))],
                }
            )

    df_bootstrap = pd.DataFrame(intervalos_bootstrap)
    display(df_bootstrap)

    rf_model = modelos_treinados["Random Forest"]
    df_importancia = (
        pd.DataFrame(
            {"Variável": features, "Importância": rf_model.feature_importances_}
        )
        .sort_values(by="Importância", ascending=False)
        .reset_index(drop=True)
    )

    cols_categoricas = {"Eh_Estatal", *SEGMENTOS_LISTAGEM_ESPERADOS}
    cols_categoricas.update({col for col in features if col.startswith("Setor_")})
    cols_desc = [col for col in X_train.columns if col not in cols_categoricas]
    estatisticas = X_train[cols_desc].describe().T[["mean", "std", "min", "50%", "max"]]
    estatisticas.columns = ["Média", "Desvio Padrão", "Mínimo", "Mediana", "Máximo"]

    matriz_rf = confusion_matrix(y_test, predicoes["Random Forest"], labels=[0, 1])
    df_matriz_rf = pd.DataFrame(
        matriz_rf,
        index=["Real_0_Não_Venceu", "Real_1_Venceu"],
        columns=["Predito_0_Não_Venceu", "Predito_1_Venceu"],
    )

    if salvar_artefatos_principais:
        df_resultados.to_csv(ARQUIVO_RESULTADOS, index=False)
        df_importancia.to_csv(ARQUIVO_IMPORTANCIA, index=False)
        estatisticas.round(4).to_csv(ARQUIVO_ESTATISTICAS)
        df_matriz_rf.to_csv(ARQUIVO_MATRIZ_CONFUSAO_RF)

    return {
        "nome_base": nome_base,
        "features": features,
        "df_resultados": df_resultados,
        "df_bootstrap": df_bootstrap,
        "df_significancia": df_significancia,
        "df_importancia": df_importancia,
        "estatisticas": estatisticas,
        "matriz_rf": matriz_rf,
        "X_train": X_train,
        "cols_desc": cols_desc,
    }


# %% [markdown]
# ## 2.2 Avaliação dos Modelos
#
# Esta célula treina os modelos, aplica a seleção estatística de variáveis e
# gera as principais tabelas de resultado.

# %%
resultado_principal = avaliar_base_ml(
    df_final, "Base Principal", salvar_artefatos_principais=True
)
df_selecao_features = resultado_principal["df_significancia"].copy()
df_selecao_features.to_csv(ARQUIVO_SELECAO_FEATURES, index=False)
print("Seleção estatística de features salva em:")
print(f"- {ARQUIVO_SELECAO_FEATURES.resolve()}")

# %% [markdown]
# ## 2.3 Gráficos e arquivos finais
#
# Os gráficos finais são salvos em PNG para uso direto no TCC.

# %%
df_resultados = resultado_principal["df_resultados"]
df_importancia = resultado_principal["df_importancia"]
estatisticas = resultado_principal["estatisticas"]
matriz_rf = resultado_principal["matriz_rf"]
X_train = resultado_principal["X_train"]
cols_desc = resultado_principal["cols_desc"]

plt.figure(figsize=(12, 9))
corr_matrix = X_train[cols_desc].corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriz de Correlação das Variáveis Preditivas", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(GRAFICO_CORRELACAO, dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x="ROC-AUC", y="Modelo", data=df_resultados, palette="Blues_r")
plt.title("Desempenho dos Modelos Preditivos (ROC-AUC)", fontsize=14, pad=15)
plt.xlim(0, 1)
plt.xlabel("Área Sob a Curva ROC (AUC)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(GRAFICO_DESEMPENHO, dpi=300)
plt.show()

plt.figure(figsize=(10, 8))
df_importancia_plot = df_importancia.head(20)
sns.barplot(x="Importância", y="Variável", data=df_importancia_plot, palette="magma")
plt.title("Importância das Variáveis (Random Forest)", fontsize=14, pad=15)
plt.xlabel("Nível de Importância Relativa")
plt.ylabel("Variáveis Preditivas")
plt.tight_layout()
plt.savefig(GRAFICO_IMPORTANCIA, dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(
    matriz_rf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Não venceu", "Venceu"],
    yticklabels=["Não venceu", "Venceu"],
)
plt.title("Matriz de Confusão Random Forest", fontsize=14, pad=15)
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.tight_layout()
plt.savefig(GRAFICO_MATRIZ_CONFUSAO, dpi=300)
plt.show()

print("Processo finalizado com sucesso.")
print("Arquivos principais gerados:")
for arquivo in [
    ARQUIVO_BASE,
    ARQUIVO_UNIVERSO,
    ARQUIVO_DIAGNOSTICO_ETL,
    ARQUIVO_DIAGNOSTICO_SETOR,
    ARQUIVO_DIAGNOSTICO_SEGMENTO,
    ARQUIVO_TICKERS_REMOVIDOS,
    ARQUIVO_RESULTADOS,
    ARQUIVO_SELECAO_FEATURES,
    ARQUIVO_IMPORTANCIA,
    ARQUIVO_ESTATISTICAS,
    ARQUIVO_MATRIZ_CONFUSAO_RF,
    GRAFICO_CORRELACAO,
    GRAFICO_DESEMPENHO,
    GRAFICO_IMPORTANCIA,
    GRAFICO_MATRIZ_CONFUSAO,
]:
    if arquivo.exists():
        print(f"- {arquivo.resolve()}")
