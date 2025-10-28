# 🏗️ DECISÕES DE ARQUITETURA - TradeSigns POC

## Data: 2025-10-28

---

## 1. CONTEXTO

Estamos construindo um **robô trader automatizado** que usa **Machine Learning** para:
- Analisar mercado forex (EUR/USD M1)
- Gerar sinais de BUY/SELL em tempo real
- Gerenciar risco dinamicamente
- Maximizar lucro com gestão inteligente

---

## 2. DECISÃO CRÍTICA: ABORDAGEM DE MACHINE LEARNING

### ❌ **REJEITADO: Classification com Labels Fixos**

**O que é:**
```python
# Labels categóricas fixas:
label = 1   # WIN se atingir +5 pips
label = -1  # LOSS se atingir -5 pips
label = 0   # HOLD se não atingir nada
```

**Por que NÃO é ideal para robô trader:**
1. ❌ **TP/SL fixo** - Não se adapta à volatilidade do mercado
2. ❌ **Decisão binária** - "Compra ou não compra" (simplista demais)
3. ❌ **Sem contexto de magnitude** - Não sabe SE vai subir 2 pips ou 20 pips
4. ❌ **Gestão de risco primitiva** - Sempre usa mesmo TP/SL
5. ❌ **Perda de informação** - Mercado sobe 4.9 pips = HOLD (desperdiçado)

**Quando usar:**
- ✅ POC rápido (validação de conceito)
- ✅ Trading manual com regras fixas
- ✅ Aprendizado inicial de ML

---

### ✅ **ESCOLHIDO: Multi-Output Regression**

**O que é:**
```python
# Labels numéricas (valores reais):
predictions = {
    'future_return_5min': +0.0023,    # Vai subir 2.3 pips em 5min
    'future_return_10min': +0.0061,   # Vai subir 6.1 pips em 10min
    'future_return_15min': +0.0045,   # Vai subir 4.5 pips em 15min
    'future_return_30min': +0.0032,   # Vai subir 3.2 pips em 30min
    'max_gain_30min': +0.0089,        # Máximo: vai atingir 8.9 pips
    'max_loss_30min': -0.0021,        # Mínimo: pode cair até 2.1 pips
    'volatility_30min': 0.0015        # Volatilidade esperada
}
```

**Por que É IDEAL para robô trader:**
1. ✅ **TP/SL dinâmico** - Ajusta baseado na previsão real
2. ✅ **Decisão inteligente** - Sabe QUANTO vai mover e QUANDO
3. ✅ **Gestão de risco adaptativa** - Ajusta posição baseado em volatilidade
4. ✅ **Múltiplos horizontes** - Escolhe melhor timeframe para cada trade
5. ✅ **Confiança estatística** - Pode calcular intervalos de confiança
6. ✅ **Position sizing** - Kelly Criterion baseado em expected return

**Vantagens específicas:**
- 📈 **Trade oportunístico**: Se modelo prevê +15 pips, coloca TP 12 pips
- 🛡️ **Risk management**: Se prevê max_loss -8 pips, coloca SL -6 pips
- ⏱️ **Time optimization**: Se melhor retorno é em 10min, fecha posição lá
- 💰 **Position sizing**: Quanto maior expected_return vs max_loss, maior posição
- 🎯 **Win rate melhor**: Não desperdiça trades de 4.9 pips

---

## 3. ESTRATÉGIA DE IMPLEMENTAÇÃO (3 FASES)

### 📋 **FASE 1: POC Básico com Classification (2-3h)**

**Objetivo:** Validar se ML funciona no conceito

**Implementação:**
```
1. Labels fixos (5 pips TP/SL, 10 min duration)
2. Features: 64 indicadores técnicos
3. Modelo: XGBoost Classifier
4. Backtest simples
5. Métricas: Accuracy, Precision, Recall, F1
```

**Critério de sucesso:**
- ✅ Accuracy > 55% (melhor que random)
- ✅ Precision > 50%
- ✅ Backtest lucro > 0
- ✅ Pipeline funcionando end-to-end

**Entregáveis:**
- `ml_dataset_fixed_labels.parquet`
- `model_classifier.pkl`
- `backtest_report_classifier.json`

---

### 🎯 **FASE 2: Regression Multi-Output (3-4h)**

**Objetivo:** Tornar modelo inteligente e adaptativo

**Implementação:**
```
1. Labels numéricas:
   - future_return_5min
   - future_return_10min
   - future_return_15min
   - future_return_30min
   - max_gain_30min
   - max_loss_30min
   
2. Modelo: XGBoost Regressor (MultiOutputRegressor)
3. TP/SL dinâmico baseado em previsões
4. Backtest com gestão de risco adaptativa
5. Métricas: MAE, RMSE, R², Sharpe Ratio, Max Drawdown
```

**Lógica de decisão:**
```python
def should_trade(predictions, confidence_threshold=0.7):
    expected_gain = predictions['max_gain_30min']
    expected_loss = abs(predictions['max_loss_30min'])
    risk_reward = expected_gain / expected_loss
    
    # Só entra se RR > 2:1
    if risk_reward > 2.0:
        tp = expected_gain * 0.8  # 80% do alvo
        sl = expected_loss * 1.2  # 120% de margem
        
        # Position sizing (Kelly Criterion)
        win_prob = model.predict_proba(data)
        kelly = (win_prob * risk_reward - 1) / risk_reward
        position_size = kelly * 0.25  # 25% do Kelly (conservador)
        
        return True, tp, sl, position_size
    
    return False, None, None, None
```

**Critério de sucesso:**
- ✅ MAE < 10 pips
- ✅ R² > 0.30
- ✅ Sharpe Ratio > 1.5
- ✅ Max Drawdown < 20%
- ✅ Win Rate > 55%

**Entregáveis:**
- `ml_dataset_regression.parquet`
- `model_regressor.pkl`
- `backtest_report_regression.json`
- `risk_management_config.yaml`

---

### 🤖 **FASE 3: Robô Trader Live (2-3h)**

**Objetivo:** Trading real com gestão de risco profissional

**Implementação:**
```
1. Conexão com broker (Oanda/MetaTrader API)
2. Loop de trading em tempo real
3. Risk management:
   - Kelly Criterion position sizing
   - Max positions simultâneas
   - Daily loss limit
   - Correlation filtering
4. Monitoring dashboard
5. Alertas (Telegram/Email)
```

**Componentes:**
- `live_trader.py` - Loop principal
- `risk_manager.py` - Gestão de risco
- `order_executor.py` - Execução de ordens
- `monitor_dashboard.py` - Dashboard Streamlit
- `alert_system.py` - Notificações

**Critério de sucesso:**
- ✅ Executa trades automaticamente
- ✅ Respeita limites de risco
- ✅ Dashboard funcionando
- ✅ Alertas enviados
- ✅ Paper trading 1 semana lucrativo

**Entregáveis:**
- Sistema completo em produção
- Dashboard de monitoring
- Logs de trades
- Relatório de performance

---

## 4. COMPARAÇÃO TÉCNICA

| Aspecto | Classification (Fixo) | Regression (Dinâmico) |
|---------|----------------------|----------------------|
| **Labels** | WIN/LOSS/HOLD | Retornos numéricos |
| **TP/SL** | Fixo (5 pips) | Dinâmico (2-50 pips) |
| **Modelo** | XGBClassifier | XGBRegressor (Multi-output) |
| **Decisão** | Binária | Contínua + Confiança |
| **Risk Management** | Fixo | Adaptativo (Kelly) |
| **Position Sizing** | Fixo | Dinâmico |
| **Timeframe** | Único | Múltiplos (5/10/15/30min) |
| **Métricas** | Accuracy, F1 | MAE, RMSE, R², Sharpe |
| **Win Rate esperado** | 50-55% | 55-65% |
| **Sharpe Ratio esperado** | 0.5-1.0 | 1.5-2.5 |
| **Complexidade** | Baixa ⭐ | Média ⭐⭐⭐ |
| **Produção ready** | ❌ Não | ✅ Sim |

---

## 5. STACK TECNOLÓGICO

### **Machine Learning:**
- `pandas` - Manipulação de dados
- `numpy` - Computação numérica
- `scikit-learn` - Pipeline ML, preprocessing, métricas
- `xgboost` - Modelo principal (gradient boosting)
- `lightgbm` - Modelo alternativo (mais rápido)

### **Feature Engineering:**
- `ta-lib` - Indicadores técnicos
- `pandas-ta` - Indicadores extras

### **Backtesting:**
- Custom engine (controle total)
- `vectorbt` - Backtest vetorizado rápido (opcional)

### **Live Trading:**
- `oandapyV20` - API Oanda
- `MetaTrader5` - API MT5 (alternativa)

### **Monitoring:**
- `streamlit` - Dashboard web
- `plotly` - Gráficos interativos
- `loguru` - Logging avançado

### **Infrastructure:**
- `docker` - Containerização
- `docker-compose` - Orquestração
- `postgres` - Banco de dados
- `redis` - Cache (opcional)

---

## 6. ROADMAP DE DESENVOLVIMENTO

### **Semana 1: POC Core**
- ✅ Dia 1: Import dados históricos (2020-2024)
- ✅ Dia 2: Feature engineering (64 features)
- 🔄 Dia 3: Labels + Training (Classification)
- ⏳ Dia 4: Backtest básico
- ⏳ Dia 5: Migração para Regression

### **Semana 2: Regression & Risk Management**
- ⏳ Dia 1-2: Multi-output regression
- ⏳ Dia 3: TP/SL dinâmico
- ⏳ Dia 4: Kelly Criterion + position sizing
- ⏳ Dia 5: Backtest avançado

### **Semana 3: Live Trading**
- ⏳ Dia 1-2: API broker + order execution
- ⏳ Dia 3: Risk manager
- ⏳ Dia 4: Dashboard + alerts
- ⏳ Dia 5: Paper trading tests

### **Semana 4: Refinamento**
- ⏳ Feature selection
- ⏳ Hyperparameter tuning
- ⏳ Ensemble models
- ⏳ Walk-forward optimization

---

## 7. RISCOS E MITIGAÇÕES

| Risco | Impacto | Probabilidade | Mitigação |
|-------|---------|---------------|-----------|
| **Overfitting** | Alto | Alta | Walk-forward validation, regularização |
| **Data snooping** | Alto | Média | Validação out-of-sample rigorosa |
| **Latência API** | Médio | Média | Uso de VPS próximo ao broker |
| **Slippage** | Médio | Alta | Backtest com slippage realista |
| **Market regime change** | Alto | Baixa | Retreino mensal, regime detection |
| **Bugs em produção** | Alto | Média | Testes extensivos, paper trading |
| **Drawdown excessivo** | Alto | Média | Stop loss diário, position limits |

---

## 8. MÉTRICAS DE SUCESSO

### **POC (FASE 1):**
- [ ] Accuracy > 55%
- [ ] Backtest Profit > 0
- [ ] Sharpe > 0.5
- [ ] Max Drawdown < 30%

### **Regression (FASE 2):**
- [ ] MAE < 10 pips
- [ ] R² > 0.30
- [ ] Sharpe > 1.5
- [ ] Max Drawdown < 20%
- [ ] Win Rate > 55%

### **Live Trading (FASE 3):**
- [ ] 1 semana paper trading lucrativo
- [ ] Sharpe > 1.5
- [ ] Max Drawdown < 15%
- [ ] Win Rate > 55%
- [ ] Profit Factor > 1.5

---

## 9. PRÓXIMOS PASSOS

### **IMEDIATO (hoje):**
1. ✅ Finalizar labels fixos (5 pips)
2. 🔄 Treinar modelo XGBoost Classifier
3. ⏳ Backtest básico
4. ⏳ Validar se ML funciona

### **CURTO PRAZO (esta semana):**
1. ⏳ Implementar regression multi-output
2. ⏳ TP/SL dinâmico
3. ⏳ Risk management
4. ⏳ Backtest comparativo

### **MÉDIO PRAZO (próxima semana):**
1. ⏳ Live trading paper
2. ⏳ Dashboard
3. ⏳ Alertas
4. ⏳ Tests em produção

---

## 10. REFERÊNCIAS

### **Papers:**
- "Machine Learning for Algorithmic Trading" - Stefan Jansen
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Quantitative Trading" - Ernest Chan

### **Técnicas:**
- Walk-Forward Optimization
- Kelly Criterion Position Sizing
- Multi-Output Regression
- Feature Importance Analysis
- Time Series Cross-Validation

### **Benchmarks:**
- Random Walk (baseline)
- Buy & Hold
- Simple Moving Average Crossover
- RSI Mean Reversion

---

## 11. CONCLUSÃO

**Decisão final:** Implementar em **3 fases incrementais**, começando com **Classification simples** para validar o conceito, evoluindo para **Regression multi-output** para produção real.

**Justificativa:** Abordagem pragmática que equilibra:
- ✅ Velocidade de desenvolvimento (POC rápido)
- ✅ Validação incremental (fail fast)
- ✅ Qualidade final (produção-ready)
- ✅ Aprendizado progressivo (complexidade gradual)

**Expectativa de timeline:** 3-4 semanas até robô trader completo em produção.

---

**Documento criado por:** Copilot AI  
**Data:** 2025-10-28  
**Versão:** 1.0  
**Status:** 🟢 Aprovado
