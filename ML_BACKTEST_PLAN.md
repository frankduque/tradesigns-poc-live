# 🎯 TradeSigns - ML & Backtest Implementation Plan

## 🧠 ARQUITETURA ML + BACKTEST

```
┌─────────────────────────────────────────────────────┐
│         DADOS HISTÓRICOS (2+ anos)                  │
│         - HistData.com (M1/M5/H1)                   │
│         - OANDA Historical API                      │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│    1. DATA PREPARATION                              │
│    - Importação massiva                             │
│    - Limpeza de dados                               │
│    - Feature Engineering (50+ features)             │
│    - Label Creation (profitable/unprofitable)       │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│    2. BACKTEST ENGINE (vectorbt)                    │
│    - Walk-Forward Validation                        │
│    - Multiple Strategies Testing                    │
│    - Parameter Optimization                         │
│    - Performance Metrics                            │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│    3. ML TRAINING                                   │
│    - Random Forest / XGBoost / LightGBM             │
│    - Cross-Validation                               │
│    - Feature Importance Analysis                    │
│    - Hyperparameter Tuning                          │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│    4. MODEL VALIDATION                              │
│    - Out-of-Sample Testing                          │
│    - Live Simulation                                │
│    - Risk Metrics (Sharpe, Sortino, Drawdown)      │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│    5. PRODUCTION INTEGRATION                        │
│    - Model Serving (joblib/pickle)                  │
│    - Live Signal Scoring                            │
│    - Continuous Monitoring                          │
└─────────────────────────────────────────────────────┘
```

---

## 📊 FASE 1: COLETA DE DADOS HISTÓRICOS

### Opção A: HistData.com (GRATUITO - Recomendado)

**Você vai baixar manualmente:**

1. **Acesse**: https://www.histdata.com/download-free-forex-data/
2. **Selecione**:
   - Currency Pair: EUR/USD, GBP/USD
   - Timeframe: 1 Minute Bar (M1)
   - Year: 2022, 2023, 2024
3. **Download**: Arquivos ZIP (ASCII format)

**Estrutura:**
```
downloads/
├── EURUSD_M1_2022.zip
├── EURUSD_M1_2023.zip
├── EURUSD_M1_2024.zip
├── GBPUSD_M1_2022.zip
├── GBPUSD_M1_2023.zip
└── GBPUSD_M1_2024.zip
```

**Formato CSV:**
```
DateTime,Open,High,Low,Close,Volume
20220101 000000,1.13251,1.13262,1.13248,1.13259,0
20220101 000100,1.13259,1.13271,1.13255,1.13268,0
...
```

### Opção B: OANDA API (Automático)

```python
# Script para baixar automaticamente via OANDA
# Limite: 5000 candles por request
# Precisa iterar para pegar anos completos
```

**Qual você prefere? A (manual HistData) ou B (automático OANDA)?**

---

## 🏗️ ESTRUTURA DE ARQUIVOS ML

```
tradesigns-poc-live/
├── data/
│   ├── raw/                    # Dados brutos (CSVs baixados)
│   ├── processed/              # Dados processados
│   └── features/               # Features calculadas
│
├── models/
│   ├── trained/                # Modelos treinados (.pkl)
│   ├── metadata/               # Info sobre treino
│   └── experiments/            # Experimentos
│
├── src/
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Importa histórico
│   │   ├── feature_engineer.py # Cria features
│   │   ├── label_creator.py    # Cria labels (profitable?)
│   │   ├── trainer.py          # Treina modelos
│   │   ├── validator.py        # Walk-forward validation
│   │   └── predictor.py        # Predição em produção
│   │
│   └── backtest/
│       ├── __init__.py
│       ├── engine.py           # Backtest engine
│       ├── vectorbt_runner.py  # vectorbt wrapper
│       ├── optimizer.py        # Otimização de params
│       └── metrics.py          # Métricas avançadas
│
├── notebooks/                  # Jupyter para análise
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtest_results.ipynb
│
└── scripts/
    ├── import_historical.py    # Importa CSVs para DB
    ├── run_backtest.py         # Executa backtest
    └── train_model.py          # Treina modelo ML
```

---

## 🎓 FEATURES ENGINEERING

### Features Técnicas (30+)

**Trend:**
- SMA (10, 20, 30, 50, 200)
- EMA (12, 26)
- SMA_cross (fast > slow?)
- Price vs SMA distance

**Momentum:**
- RSI (14)
- Stochastic (14, 3, 3)
- CCI (20)
- Williams %R
- ROC (Rate of Change)

**Volatility:**
- ATR (14)
- Bollinger Bands Width
- Bollinger %B
- Keltner Channels

**Volume:**
- OBV
- Volume MA Ratio
- MFI (Money Flow Index)

**Price Action:**
- Candle Size
- Upper/Lower Shadow Ratio
- Body/Shadow Ratio
- High/Low Range

**Time-based:**
- Hour of day (0-23)
- Day of week (0-6)
- Is London/NY session

### Features de Contexto (10+)

**Market State:**
- Trend strength (ADX)
- Volatility regime (ATR percentile)
- Volume profile
- Support/Resistance proximity

**Sequential Features:**
- Last N candles pattern
- Consecutive ups/downs
- Recent volatility spike

### Derived Features (10+)

**Ratios:**
- RSI / ATR
- Volume / Volume MA
- Price / Bollinger Middle

**Differences:**
- RSI change (current - previous)
- Price momentum (close - close N periods ago)

---

## 🏷️ LABEL CREATION

### Estratégia de Labels

**Approach 1: Fixed Horizon (Simples)**
```python
def create_labels(df, horizon=10, take_profit=0.002, stop_loss=0.001):
    """
    Label = 1 (BUY) se preço sobe TP% em próximos N candles
    Label = -1 (SELL) se preço cai TP% em próximos N candles
    Label = 0 (HOLD) caso contrário
    """
```

**Approach 2: Realistic Trading (Melhor)**
```python
def create_realistic_labels(df, tp_pct=0.004, sl_pct=0.002, max_duration=60):
    """
    Simula trade real:
    - Entry no close do candle
    - Label = 1 se atingir TP antes de SL/Timeout
    - Label = -1 se atingir SL antes de TP
    - Label = 0 se timeout sem TP/SL
    """
```

**Approach 3: Regression (Avançado)**
```python
def create_regression_target(df, horizon=10):
    """
    Target = % de retorno em N candles
    Modelo prediz retorno esperado (não apenas classificação)
    """
```

---

## 🤖 MODELOS ML

### Opção 1: Random Forest (Baseline)
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=100,
    class_weight='balanced'  # Lidar com desbalanceamento
)
```

**Prós:** Rápido, interpretável, robusto
**Contras:** Menos preciso que gradient boosting

### Opção 2: XGBoost (Recomendado)
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=2  # Ajuste para desbalanceamento
)
```

**Prós:** Alta precisão, rápido, robusto
**Contras:** Mais hyperparameters para tunar

### Opção 3: LightGBM (Mais Rápido)
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31
)
```

**Prós:** Muito rápido, memória eficiente
**Contras:** Pode overfittar mais fácil

### Opção 4: Neural Network / LSTM (Avançado)
```python
# Sequencial, captura padrões temporais
# Mais complexo, requer mais dados
# Próxima iteração
```

---

## 📈 BACKTEST COM VECTORBT

```python
import vectorbt as vbt

# Carregar dados
price = vbt.YFData.download("EURUSD=X", start='2022-01-01').get('Close')

# Gerar sinais do modelo ML
entries = model.predict(features) == 1
exits = model.predict(features) == -1

# Executar backtest
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.0002,
    sl_stop=0.02,
    tp_stop=0.04
)

# Métricas
print(f"Sharpe: {portfolio.sharpe_ratio():.2f}")
print(f"Win Rate: {portfolio.win_rate():.2%}")
print(f"Max Drawdown: {portfolio.max_drawdown():.2%}")
print(f"Total Return: {portfolio.total_return():.2%}")
```

---

## 🎯 WALK-FORWARD VALIDATION

```python
# Evitar overfitting com validação walk-forward
# Treina em período passado, testa em período futuro

Train: [2022-01 to 2022-06] → Test: [2022-07]
Train: [2022-02 to 2022-07] → Test: [2022-08]
Train: [2022-03 to 2022-08] → Test: [2022-09]
...

# Simula produção: sempre prediz o futuro com dados do passado
```

---

## 📋 ROADMAP DE IMPLEMENTAÇÃO

### **PASSO 1: Você Baixa Dados (10 min)**
- [ ] Baixar 3 anos de EURUSD M1 do HistData
- [ ] Salvar em `data/raw/`

### **PASSO 2: Eu Crio Scripts (1 hora)**
- [ ] `scripts/import_historical.py` - Importa CSVs
- [ ] `src/ml/data_loader.py` - Carrega dados
- [ ] `src/ml/feature_engineer.py` - Cria features

### **PASSO 3: Feature Engineering (1 hora)**
- [ ] Calcular 50+ features
- [ ] Criar labels (profitable trades)
- [ ] Salvar dataset processado

### **PASSO 4: Backtest Baseline (30 min)**
- [ ] Rodar SMA Cross em dados históricos
- [ ] Calcular métricas baseline
- [ ] Identificar se vale a pena ML

### **PASSO 5: ML Training (1 hora)**
- [ ] Treinar XGBoost
- [ ] Walk-forward validation
- [ ] Feature importance analysis
- [ ] Salvar modelo treinado

### **PASSO 6: Integração Produção (30 min)**
- [ ] Carregar modelo no signal generator
- [ ] Usar ML score + regras
- [ ] Testar em dados recentes

---

## 🚀 PRÓXIMAS AÇÕES

**VOCÊ AGORA:**
1. Decidir: HistData manual OU OANDA automático?
2. Se manual: Baixar 3 anos de EURUSD M1
3. Colocar ZIPs em `tradesigns-poc-live/data/raw/`

**EU AGORA:**
1. Criar estrutura ML completa
2. Script de importação
3. Feature engineering
4. Backtest + ML pipeline

**Qual fonte de dados você prefere?**
- [ ] A) HistData (manual, 10 min download)
- [ ] B) OANDA API (automático via script)
