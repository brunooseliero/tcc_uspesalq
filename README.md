# Machine Learning na Predição de Ações

Este repositório contém o código e os dados de apoio do TCC **Machine Learning na Predição de Ações: A Rentabilidade e a Anomalia da Baixa Volatilidade no Mercado Brasileiro**.

O objetivo do projeto é construir uma base anual de empresas brasileiras listadas na B3 e avaliar quais variáveis explicativas ajudam a prever se uma ação superou o Ibovespa no ano. O pipeline está organizado em duas etapas principais:

- **Etapa 1 - Engenharia de Dados (ETL):** coleta, tratamento e consolidação de preços, fundamentos, variáveis macroeconômicas e informações de governança/setor.
- **Etapa 2 - Machine Learning:** seleção estatística de variáveis, treino dos modelos e comparação de desempenho preditivo.

## Fontes de Dados

O projeto prioriza fontes oficiais e auditáveis:

- **Preços das ações:** arquivos oficiais B3 `COTAHIST`, mantidos localmente em `cotahist/`.
- **Fundamentos contábeis:** demonstrações financeiras padronizadas da CVM `DFP`, mantidas localmente em `cvm_dfp/`.
- **Macroeconomia:** séries públicas do Banco Central do Brasil via SGS.
- **Universo, setor e governança:** dados públicos da B3 e mapeamento consolidado em `universo_empresas_b3_cvm.csv`.
- **Ibovespa:** usado como benchmark anual de comparação.

## Estrutura do Repositório

```text
.
├── script_tcc.py                       # Código-fonte principal em formato editável
├── script_tcc.ipynb                    # Notebook explicativo para execução célula a célula
├── DATABASE_MESTRE_FINAL.csv           # Base final consolidada usada no Machine Learning
├── universo_empresas_b3_cvm.csv        # Universo auditável de empresas B3/CVM
├── resultados_modelos.csv              # Métricas finais dos modelos
├── selecao_features_significancia.csv  # Resultado da seleção estatística de variáveis
├── requirements_tcc.txt                # Dependências Python do projeto
├── cotahist/                           # Cache dos preços oficiais B3 COTAHIST
└── cvm_dfp/                            # Cache dos arquivos CVM DFP
```

## Como Executar

Crie um ambiente virtual e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_tcc.txt
```

Para executar de forma explicativa, abra o notebook:

```bash
jupyter notebook script_tcc.ipynb
```

Também é possível executar o pipeline pelo terminal:

```bash
python script_tcc.py
```

Por padrão, o script executa a Etapa 1 e a Etapa 2. Como os diretórios `cotahist/` e `cvm_dfp/` já estão no repositório, a execução tende a reutilizar os arquivos locais e evita baixar novamente os dados grandes.

Se o objetivo for apenas reproduzir os modelos a partir da base pronta, altere no início do `script_tcc.py`:

```python
EXECUTAR_ETAPA_1 = False
```

Em macOS, caso o XGBoost apresente erro de OpenMP, instale o runtime antes de executar:

```bash
brew install libomp
```

## Saídas Geradas

Durante a execução, o pipeline pode gerar ou atualizar os seguintes arquivos:

- `DATABASE_MESTRE_FINAL.csv`
- `diagnostico_cobertura.csv`
- `diagnostico_cobertura_setor.csv`
- `diagnostico_cobertura_segmento.csv`
- `diagnostico_tickers_removidos.csv`
- `resultados_modelos.csv`
- `selecao_features_significancia.csv`
- `importancia_variaveis.csv`
- `estatisticas_descritivas.csv`
- `matriz_confusao_random_forest.csv`
- `grafico_correlacao.png`
- `grafico_desempenho_modelos.png`
- `grafico_feature_importance.png`
- `grafico_matriz_confusao_random_forest.png`

## Modelos Avaliados

Foram mantidos os quatro modelos definidos na metodologia do trabalho:

- Regressão Logística
- Random Forest
- XGBoost
- MLP Classifier

A variável-alvo é `Target_Venceu`, que indica se a ação superou o Ibovespa no respectivo ano. Antes do treinamento, o pipeline aplica uma etapa de seleção estatística de variáveis para reduzir ruído e melhorar a interpretação dos fatores explicativos.

## Base Final

A base consolidada atual possui:

- **Período:** 2010 a 2025
- **Empresas:** 122 tickers
- **Observações:** 1.402 linhas empresa-ano
- **Colunas:** 58 variáveis

Entre as variáveis explicativas estão medidas de risco, retorno, liquidez, drawdown, beta, rentabilidade, crescimento, alavancagem, governança, setor e macroeconomia.

## Observações Metodológicas

Este projeto tem finalidade acadêmica. Os resultados não constituem recomendação de investimento.

Como a análise usa dados históricos e um universo filtrado por disponibilidade de preço e fundamentos, os resultados devem ser interpretados considerando limitações como viés de sobrevivência, disponibilidade de dados e mudanças de regime de mercado.

O código foi mantido em formato notebook e script para facilitar tanto a leitura acadêmica quanto a reprodutibilidade técnica.
