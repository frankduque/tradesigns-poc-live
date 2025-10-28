# 🚀 Quick Start - Multi-Target Regression

## O QUE FOI IMPLEMENTADO

Sistema de Machine Learning para trading que:

1. **Prevê variação real do preço** (não usa labels arbitrárias)
2. **Múltiplos horizontes**: 5min, 10min, 30min
3. **Threshold dinâmico**: Ajustado por volatilidade (ATR)
4. **TP/SL adaptativos**: Baseados na previsão do modelo

---

## ARQUITETURA

```
Data Pipeline:
  ├─ Dados históricos (2M+ candles EUR/USD M1)
  ├─ Feature Engineering (64 features técnicas)
  └─ Labels de Regressão (return_5m, return_10m, return_30m)

ML Pipeline:
  ├─ Multi-Output Random Forest (200 trees)
  ├─ Train/Test Split (80/20)
  └─ Cross-Validation (3-fold)

Signal Generation:
  ├─ Método Simples: Usa pred_return_10m + ATR
  └─ Método Avançado: Confirma em múltiplos horizontes

Backtest:
  └─ TP/SL dinâmicos baseados na magnitude prevista
```

---

## 🎯 EXECUÇÃO (3 PASSOS)

### 1️⃣ Preparar Dataset de Regressão

```bash
python scripts/prepare_regression_dataset.py
```

**O que faz:**
- Carrega dados históricos processados
- Cria 64 features técnicas
- Gera labels realistas: `return_5m`, `return_10m`, `return_30m`
- Salva: `data/features/ml_dataset_regression.parquet`

**Tempo estimado:** ~2-3 minutos

**Output esperado:**
```
Samples: ~2,100,000
Features: 64
Targets: 3 (return_5m, return_10m, return_30m)
```

---

### 2️⃣ Treinar Modelo

```bash
python scripts/train_model_regression.py
```

**O que faz:**
- Treina Multi-Output Random Forest
- Cross-validation 3-fold
- Análise de feature importance
- Salva: `models/regression_model.joblib`

**Tempo estimado:** ~10-15 minutos

**Output esperado:**
```
return_10m (Test Set):
  RMSE: ~0.05-0.08%
  MAE:  ~0.03-0.05%
  R2:   ~0.15-0.30
```

**Interpretação do R2:**
- R2 ~ 0.20-0.30 é EXCELENTE para Forex (mercado eficiente)
- Explica 20-30% da variância dos returns
- Suficiente para gerar alpha

---

### 3️⃣ Gerar Sinais e Backtest

```bash
python scripts/generate_signals_regression.py
```

**O que faz:**
- Carrega modelo treinado
- Gera sinais com 2 métodos (simples e avançado)
- Executa backtest com TP/SL dinâmicos
- Salva resultados em `data/results/`

**Tempo estimado:** ~3-5 minutos

**Output esperado:**
```
MÉTODO SIMPLES:
  Sinais: ~50,000-100,000
  Win Rate: ~52-55%
  Avg P&L: +0.05-0.10%

MÉTODO AVANÇADO:
  Sinais: ~10,000-30,000 (mais seletivo)
  Win Rate: ~55-60%
  Avg P&L: +0.08-0.15%
```

---

## 📊 ANÁLISE DE RESULTADOS

Após executar os 3 passos, você terá:

### 1. Dataset de Regressão
`data/features/ml_dataset_regression.parquet`
- 2M+ samples com 64 features + 3 targets
- Pronto para re-treinar com novos parâmetros

### 2. Modelo Treinado
`models/regression_model.joblib`
- Multi-Output Random Forest
- Feature importance salva

### 3. Resultados de Backtest
`data/results/`
- `backtest_simple.csv`: Trades executados (método simples)
- `backtest_advanced.csv`: Trades executados (método avançado)
- `signals_and_predictions.parquet`: Todos os sinais + previsões

---

## 🔧 AJUSTE DE PARÂMETROS

### Se Win Rate estiver baixo (<50%):

1. **Aumentar base_threshold** em `generate_signals_regression.py`:
   ```python
   base_threshold=0.08  # De 0.05 para 0.08 (mais conservador)
   ```

2. **Usar método avançado** (já confirma em múltiplos horizontes)

3. **Retreinar com mais dados**:
   - Usar período mais recente (2023-2024 apenas)
   - Mercado pode ter mudado de regime

### Se sinais forem poucos (<10,000):

1. **Diminuir base_threshold**:
   ```python
   base_threshold=0.03  # Mais agressivo
   ```

2. **Reduzir atr_multiplier**:
   ```python
   atr_multiplier=1.5  # De 2.0 para 1.5
   ```

### Se quiser mais features:

Editar `src/ml/feature_engineer.py` e adicionar:
- Order book features (se tiver acesso)
- Sentiment indicators
- Correlação com outros pares
- Features de regime de volatilidade

---

## 📈 PRÓXIMOS PASSOS

### Curto Prazo (POC):
- [ ] Validar performance em período out-of-sample
- [ ] Walk-forward analysis (retreinar a cada mês)
- [ ] Adicionar risk management (max drawdown stop)
- [ ] Dashboard real-time com Streamlit

### Médio Prazo (Produto):
- [ ] Implementar Reinforcement Learning
- [ ] API REST para sinais
- [ ] Multi-pair (EUR/USD, GBP/USD, etc.)
- [ ] Sistema de notificações

### Longo Prazo (Scale):
- [ ] Ensemble de modelos (RF + XGBoost + RL)
- [ ] Auto-ML para otimização contínua
- [ ] Deploy em cloud (AWS/GCP)
- [ ] A/B testing de estratégias

---

## ⚠️ TROUBLESHOOTING

### Erro: "Dataset não encontrado"
```bash
# Você precisa ter executado o import_data.py primeiro
python scripts/import_histdata.py
python scripts/process_data.py
```

### Erro: "Memory Error"
- Reduzir período de dados (usar apenas 2023-2024)
- Usar `chunksize` na leitura do parquet
- Aumentar RAM ou usar cloud

### Modelo não está performando
- Verificar distribuição de labels (deve ser ~normal)
- Checar feature importance (features úteis?)
- Testar períodos diferentes (regime de mercado)
- Validar que não há data leakage

---

## 📚 REFERÊNCIAS

**Código:**
- `src/ml/label_creator_regression.py` - Cria labels realistas
- `src/ml/trainer_regression.py` - Treina modelo
- `src/ml/signal_generator_regression.py` - Gera sinais

**Documentação:**
- `docs/ML_APPROACH_DECISION.md` - Decisão técnica completa
- `docs/ARCHITECTURE_DECISION.md` - Arquitetura do sistema

**Logs:**
- `logs/prepare_regression_dataset.log`
- `logs/train_regression_model.log`
- `logs/generate_signals_regression.log`

---

## ✅ CHECKLIST COMPLETO

```
[ ] 1. Dados históricos importados (import_histdata.py)
[ ] 2. Dados processados (process_data.py)
[ ] 3. Dataset de regressão criado (prepare_regression_dataset.py)
[ ] 4. Modelo treinado (train_model_regression.py)
[ ] 5. Sinais gerados e backtest executado (generate_signals_regression.py)
[ ] 6. Resultados analisados (data/results/)
```

**Tempo total estimado:** ~20-30 minutos

---

## 🎯 MÉTRICAS DE SUCESSO

Para considerar o POC bem-sucedido, esperamos:

✅ **Win Rate > 52%** (acima do aleatório)
✅ **Sharpe Ratio > 1.5** (bom risco/retorno)
✅ **Max Drawdown < 15%** (risco controlado)
✅ **Profit Factor > 1.5** (wins > losses)
✅ **Avg P&L > 0.05%** (5+ pips por trade)

Se atingir essas métricas, o POC está validado! 🚀
