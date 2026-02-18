# Sistema de Forecast Multi-Produto SKU

Sistema completo de forecasting para multiplos produtos (SKUs) com analise de similaridade entre series temporais, modelos agregados por cluster e rateio proporcional por produto. Dashboard interativo com Flask.

---

## Arquitetura do Sistema

```
sistema_multiplos_forecast/
|
|-- app.py                          # Entry point do dashboard Flask
|-- sistema_forecast.py             # Orquestrador principal (CLI demo)
|-- config.py                       # Configuracoes globais e paleta de cores
|-- requirements.txt                # Dependencias Python
|
|-- data/
|   |-- synthetic_generator.py      # Geracao de dados sinteticos (20 SKUs, 2.5 anos)
|   `-- feature_engineering.py      # Lags, rolling stats, encoding de features
|
|-- models/
|   |-- base_model.py               # Interface abstrata BaseForecaster
|   |-- ml_models.py                # XGBoost e LightGBM
|   |-- arima_model.py              # Auto ARIMA (pmdarima)
|   |-- prophet_model.py            # Prophet (Meta)
|   |-- chronos_model.py            # Chronos-Bolt (Amazon Foundation Model)
|   |-- intermittent_models.py      # Croston SBA (statsforecast)
|   `-- model_registry.py           # Registry para instanciar modelos por nome
|
|-- similarity/
|   |-- clustering.py               # DTW, Pearson, Euclidean + TimeSeriesKMeans
|   `-- aggregation.py              # Agregacao por cluster + rateio proporcional
|
|-- evaluation/
|   `-- metrics.py                  # MAE, RMSE, MAPE, WAPE, Bias
|
|-- pipeline/
|   |-- forecasting_pipeline.py     # Pipeline de forecast individual por SKU
|   `-- cluster_pipeline.py         # Pipeline agregado: cluster -> forecast -> rateio
|
|-- templates/                      # Views HTML (Flask)
`-- static/                         # Assets (CSS/JS)
```

## Funcionalidades

### Dados Sinteticos

Gerador robusto com **20 SKUs** distribuidos em **5 arquetipos de demanda** cobrindo **2.5 anos** de dados diarios:

| Arquetipo | Exemplos | Caracteristicas |
|-----------|----------|-----------------|
| Estavel | Arroz, Feijao, Acucar | Baseline alto, baixo ruido, pouca sazonalidade |
| Sazonal | Protetor Solar, Sorvete, Panetone | Forte sazonalidade alinhada ao verao |
| Tendencia Alta | Whey Protein, Leite Vegetal | Trend linear positivo, crescimento progressivo |
| Tendencia Baixa | DVD Virgem, Filme Fotografico | Trend negativo, produto em declinio |
| Intermitente | Peca Reposicao, Reagente Lab | 50-70% de zeros, demanda esporadica |

**Features incluidas:**
- **Climaticas**: temperatura, precipitacao, umidade (sinteticas baseadas em Ribeirao Preto/SP)
- **Temporais**: dia da semana, mes, ano
- **Estacoes**: verao, outono, inverno, primavera (hemisferio sul)
- **Safra**: fases de soja, milho safrinha e cana-de-acucar (plantio, crescimento, colheita, entressafra)
- **Lags e Rolling**: lag 1/7/14/28 dias + media e desvio movel 7/14/28 dias

### Modelos de Forecast

| Modelo | Biblioteca | Variaveis Exogenas | Demanda Intermitente |
|--------|------------|-------------------|---------------------|
| **XGBoost** | xgboost | Sim | Nao |
| **LightGBM** | lightgbm | Sim | Nao |
| **Auto ARIMA** | pmdarima | Sim | Nao |
| **Prophet** | prophet | Sim (regressores + feriados BR) | Nao |
| **Chronos-Bolt** | chronos-forecasting | Nao (zero-shot) | Nao |
| **Croston SBA** | statsforecast | Nao | Sim |

Todos os modelos implementam a interface `BaseForecaster` com metodos `fit()`, `predict()` e `get_params()`, retornando previsoes pontuais + intervalos de confianca.

### Analise de Similaridade e Clustering

1. **Metricas de distancia**: DTW (Dynamic Time Warping), correlacao de Pearson, distancia Euclidiana
2. **Clustering**: Hierarchical clustering com Ward linkage + selecao automatica de k via silhouette score
3. **Visualizacao**: Heatmap de distancias, projecao MDS 2D, dendrograma hierarquico

### Forecast Agregado com Rateio Proporcional

1. Series similares sao agrupadas em clusters
2. A demanda do cluster e agregada (soma diaria)
3. Modelos sao treinados na serie agregada
4. O forecast e desagregado via **rateio proporcional** (pesos rolling de 28 dias ou estaticos)
5. Cada SKU recebe sua parcela proporcional do forecast agregado

### Dashboard Interativo (5 Paginas)

| Pagina | Descricao |
|--------|-----------|
| **Visao Geral** | KPIs (total SKUs, registros, cobertura, demanda media), heatmap SKU x mes, top 10 SKUs, demanda total agregada, distribuicao por perfil |
| **Exploracao de Dados** | Series de demanda por SKU, dados climaticos (temperatura, chuva, umidade), distribuicoes por estacao/dia/safra/perfil, tabela com download CSV |
| **Similaridade** | Matriz de distancia, silhouette score por k, projecao MDS 2D colorida por cluster, dendrograma, detalhe de series por cluster |
| **Previsao** | Modo individual (por SKU) ou agregado (por cluster), selecao de modelos, grafico de forecast com intervalos de confianca, rateio por SKU, pesos de rateio |
| **Comparacao** | Ranking geral, melhor modelo por metrica e por SKU, radar chart multi-metrica, box plots, graficos agrupados, analise de residuos com ACF |

---

## Como Executar

### Pre-requisitos

- Python 3.10+
- pip

### Instalacao

```bash
git clone https://github.com/viniciusds2020/sistema_multiplos_forecast.git
cd sistema_multiplos_forecast
pip install -r requirements.txt
```

### Executar o Dashboard

```bash
python app.py
```

O dashboard abre em `http://localhost:5000`. Frontend em HTML puro com TailwindCSS + Plotly.js, backend Flask com API REST.

### Executar via CLI (demo)

```bash
python sistema_forecast.py
```

Executa uma demonstracao completa no terminal: geracao de dados, clustering, forecast individual e agregado com metricas.

---

## Dependencias Principais

| Categoria | Bibliotecas |
|-----------|-------------|
| Core | pandas, numpy, scikit-learn |
| Series Temporais | pmdarima, prophet, statsforecast |
| ML | xgboost, lightgbm |
| Foundation Model | chronos-forecasting, torch |
| Clustering | tslearn, scipy |
| Dashboard | flask, plotly, TailwindCSS, Plotly.js |

### Nota sobre Chronos

O Chronos-Bolt requer PyTorch. Para instalacao em CPU (Windows):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install chronos-forecasting
```

Se a instalacao do Chronos falhar, o sistema funciona normalmente com os demais modelos.

---

## Metricas de Avaliacao

| Metrica | Descricao |
|---------|-----------|
| **MAE** | Mean Absolute Error - erro absoluto medio |
| **RMSE** | Root Mean Squared Error - penaliza erros grandes |
| **MAPE** | Mean Absolute Percentage Error - percentual (ignora zeros) |
| **WAPE** | Weighted Absolute Percentage Error - melhor para series intermitentes |
| **Bias** | Vies medio - detecta sobre/sub-previsao sistematica |

---

## Fluxo de Dados

```
[Geracao de Dados Sinteticos]
        │
        ▼
[Feature Engineering]
   (clima, temporal, safra, lags)
        │
        ├──► [Pipeline Individual]
        │         │
        │         ├─ Para cada SKU:
        │         │   fit modelos → predict → avaliar
        │         ▼
        │    [Forecasts por SKU]
        │
        └──► [Pipeline por Cluster]
                  │
                  ├─ [Similaridade] DTW/Pearson → Clustering
                  ├─ [Agregacao] soma demanda por cluster
                  ├─ fit modelos no agregado → predict
                  ├─ [Rateio Proporcional] desagregar por SKU
                  ▼
             [Forecasts Desagregados]
                  │
                  ▼
[Avaliacao & Comparacao]
   MAE, RMSE, MAPE, WAPE por modelo/SKU
        │
        ▼
[Dashboard Interativo]
```

---

## Contribuicao

1. Fork o repositorio
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas alteracoes (`git commit -m 'feat: descricao'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

---

## Licenca

Este projeto esta sob a licenca MIT.
