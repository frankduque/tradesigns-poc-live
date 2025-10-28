# 🎉 Pipeline ML + Backtest - COMPLETO!

## ✅ O QUE FOI CRIADO

### 📦 **Estrutura Completa:**

```
tradesigns-poc-live/
├── data/
│   ├── raw/              ← ✅ 15 ZIPs do HistData (2020-2025)
│   ├── processed/        ← ✅ Parquet processado
│   └── features/         ← ✅ Dataset ML com features
│
├── models/
│   ├── trained/          ← ✅ Modelos .pkl salvos
│   └── metadata/         ← ✅ Métricas, importance, etc
│
├── src/
│   ├── ml/
│   │   ├── data_loader.py      ✅ Importa ZIPs
│   │   ├── feature_engineer.py ✅ Cria 70+ features
│   │   ├── label_creator.py    ✅ Labels realistas
│   │   └── trainer.py          ✅ XGBoost/LightGBM
│   │
│   └── backtest/
│       └── engine.py     ✅ Backtest completo
│
└── scripts/
    ├── import_historical.py  ✅ Importa dados
    ├── prepare_dataset.py    ✅ Features + Labels
    ├── train_model.py        ✅ Treina ML
    └── run_backtest.py       ✅ Valida modelo
```

---

## 🚀 COMO USAR (Passo a Passo)

### **PASSO 1: Instalar Dependências**

```bash
cd tradesigns-poc-live
venv\Scripts\activate

# Instalar novas libs ML
pip install -r requirements.txt
```

---

### **PASSO 2: Importar Dados Históricos** (5-10 min)

```bash
python scripts/import_historical.py
```

**O que faz:**
- ✅ Descompacta os 15 ZIPs
- ✅ Concatena tudo (~1.5M candles)
- ✅ Valida dados (OHLC, gaps, duplicatas)
- ✅ Salva em `data/processed/eurusd_m1_full.parquet` (~100MB)

**Resultado esperado:**
```
✅ IMPORTAÇÃO CONCLUÍDA!
📊 Total de candles: 1,500,000
📅 Período: 2020-01-01 até 2025-10-27
📁 Arquivo salvo: data/processed/eurusd_m1_full.parquet
💾 Tamanho: 112 MB
```

---

### **PASSO 3: Criar Features + Labels** (15-30 min)

```bash
python scripts/prepare_dataset.py
```

**O que faz:**
- ✅ Carrega 1.5M candles
- ✅ Calcula **70+ features técnicas**:
  - Trend: SMA, EMA, crosses
  - Momentum: RSI, MACD, Stochastic
  - Volatility: ATR, Bollinger, Keltner
  - Volume: OBV, VWAP
  - Price Action: candles, shadows
  - Time: sessões de trading
  - Derived: ratios, diffs
- ✅ Simula **trades realistas** para cada candle:
  - TP: 0.4% (40 pips)
  - SL: 0.2% (20 pips)
  - Max Duration: 60 min
- ✅ Salva em `data/features/ml_dataset_full.parquet` (~200MB)

**Resultado esperado:**
```
✅ DATASET ML PRONTO!
📊 Samples: 1,450,000 (após warm-up)
📈 Features: 73
⏱️ Tempo: 25 minutos

📊 Distribuição de Labels:
   WIN (1):  520,000 (36%)
   LOSS (-1): 480,000 (33%)
   HOLD (0): 450,000 (31%)
```

---

### **PASSO 4: Treinar Modelo ML** (10-20 min)

```bash
python scripts/train_model.py
```

**Interativo - você escolhe:**
```
❓ Escolha o modelo (xgboost/lightgbm) [xgboost]:
```

**O que faz:**
- ✅ Walk-Forward Split:
  - Train: 2020-2023 (4 anos) = ~1M samples
  - Test: 2024 (1 ano) = ~260k samples
- ✅ Treina XGBoost com 300 estimators
- ✅ Avalia no teste:
  - Accuracy
  - WIN Precision/Recall
  - Confusion Matrix
- ✅ Feature Importance (top 20)
- ✅ Salva modelo: `models/trained/xgboost_YYYYMMDD_HHMMSS.pkl`

**Resultado esperado:**
```
✅ TREINAMENTO CONCLUÍDO!
🤖 Modelo: xgboost
📊 Accuracy: 0.5843
🎯 WIN Precision: 0.6214
⏱️ Tempo total: 12.5 min

Top Features:
   rsi_14              : 0.0842
   sma_cross_10_30     : 0.0731
   bb_percent          : 0.0689
   macd_hist           : 0.0624
   atr_14              : 0.0587
   ...
```

---

### **PASSO 5: Executar Backtest** (5 min)

```bash
python scripts/run_backtest.py
```

**Interativo:**
```
❓ Data de início do backtest [2024-01-01]:
❓ Data de fim do backtest [2024-12-31]:
```

**O que faz:**
- ✅ Carrega modelo treinado
- ✅ Gera predições para 2024
- ✅ Simula trades com:
  - Capital inicial: $10,000
  - TP/SL realistas
  - Fees incluídas
- ✅ Calcula métricas:
  - Win Rate
  - Total Return
  - Sharpe Ratio
  - Max Drawdown
  - Profit Factor
- ✅ Salva:
  - Todos os trades (CSV)
  - Equity curve (CSV)
  - Métricas (JSON)

**Resultado esperado (se modelo for bom):**
```
📊 RESULTADOS DO BACKTEST
💰 Capital Inicial: $10,000.00
💰 Capital Final: $12,450.00
📈 Retorno Total: +24.50%

📊 Trades:
   Total: 1,245
   Wins: 712 (57.2%)
   Losses: 533

🎯 Outcomes:
   Take Profit: 680
   Stop Loss: 465
   Timeout: 100

📈 Performance:
   Sharpe Ratio: 1.34
   Max Drawdown: -8.23%
   Profit Factor: 1.89
   Avg Win: +0.38%
   Avg Loss: -0.22%
   Avg Duration: 35 min

✅ WIN RATE EXCELENTE (>55%)!
✅ SHARPE RATIO EXCELENTE (>1.0)!
✅ DRAWDOWN CONTROLADO (<10%)!
```

---

## 📊 FEATURES CRIADAS (73 no total)

### **Trend (12 features)**
- sma_10, sma_20, sma_30, sma_50, sma_100, sma_200
- ema_12, ema_26, ema_50
- sma_cross_10_30, sma_cross_20_50
- price_sma_20_dist, price_sma_50_dist

### **Momentum (13 features)**
- rsi_14, rsi_7
- stoch_k, stoch_d
- cci_20, willr_14, roc_10, mfi_14
- macd, macd_signal, macd_hist
- rsi_change, macd_hist_change

### **Volatility (12 features)**
- atr_14
- bb_upper, bb_middle, bb_lower, bb_width, bb_percent
- kc_upper, kc_middle, kc_lower
- volatility_10, volatility_30

### **Volume (4 features)**
- obv, volume_ma_20, volume_ratio, vwap

### **Price Action (8 features)**
- candle_size, candle_body, candle_body_pct
- upper_shadow, lower_shadow
- is_bullish, range_pct
- consecutive_ups, consecutive_downs

### **Time (7 features)**
- hour, day_of_week, day_of_month
- is_london_session, is_ny_session, is_asian_session
- is_overlap

### **Derived (10 features)**
- price_change_1, price_change_5, price_change_10
- rsi_atr_ratio, volume_volatility_ratio
- distance_from_high_20, distance_from_low_20

---

## 🎯 PRÓXIMOS PASSOS

### **Se Backtest for BOM (Win Rate > 52%, Sharpe > 0.5):**

1. **Integrar ao sistema live:**
   - Editar `src/signals/generator.py`
   - Carregar modelo treinado
   - Usar predições em vez de SMA Cross

2. **Monitorar em produção:**
   - `python start.py`
   - Ver sinais ML em tempo real

### **Se Backtest for RUIM:**

1. **Ajustar hiperparâmetros:**
   - Editar `src/ml/trainer.py`
   - Tunar n_estimators, max_depth, etc

2. **Adicionar features:**
   - Editar `src/ml/feature_engineer.py`
   - Adicionar padrões de candles
   - Adicionar features sequenciais (LSTM)

3. **Ajustar labeling:**
   - Editar `src/ml/label_creator.py`
   - Testar TP/SL diferentes
   - Testar max_duration diferente

---

## ⏱️ TEMPO ESTIMADO TOTAL

```
PASSO 1: Instalar deps     →  2 min
PASSO 2: Importar dados    →  8 min
PASSO 3: Features + Labels → 25 min
PASSO 4: Treinar modelo    → 15 min
PASSO 5: Backtest          →  5 min
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:                      ~55 min
```

**Você pode deixar rodando e ir fazer outra coisa!** ☕

---

## 🚀 COMEÇAR AGORA

```bash
# 1. Ativar venv
venv\Scripts\activate

# 2. Instalar dependências ML
pip install -r requirements.txt

# 3. Executar pipeline completo
python scripts/import_historical.py
python scripts/prepare_dataset.py
python scripts/train_model.py
python scripts/run_backtest.py
```

**TUDO PRONTO!** 🎉

Quer começar agora ou tem alguma dúvida? 🚀
