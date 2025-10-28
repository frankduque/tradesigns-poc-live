# 🎯 DECISÃO ESTRATÉGICA: ARQUITETURA ML PARA TRADESIGNS

**Data**: 2025-10-28  
**Status**: ✅ DECISÃO APROVADA  
**Autor**: Time TradeSigns POC

---

## 📊 CONTEXTO DO PROBLEMA

**Objetivo Principal**: Criar um robô trader que gere sinais BUY/SELL automaticamente para pares de moeda (EUR/USD).

**Dados Disponíveis**: 
- 2,106,710 candles (M1 - 2020-2024)
- 64 features técnicas (EMAs, RSI, Bollinger, MACD, etc)
- Labels realistas com TP/SL/fees simulados

**Experimento de Configuração**:
- Testamos 10 configurações diferentes (scalping, intraday, swing)
- **MELHOR RESULTADO**: Scalping Ultra (5 pips TP/SL, 10min)
  - WIN: 11.7% | LOSS: 12.9% | HOLD: 75.4%
  - Score: 54.3/100

---

## ❌ ABORDAGENS DESCARTADAS

### 1. ❌ Classification (BUY/SELL/HOLD)
**Por que não funciona**:
- Perde informações críticas: "quanto vai subir/descer?"
- Não diz "quando sair do trade"
- Treina com labels artificiais (não trades reais)
- Ignora dimensão temporal do mercado

**Problema Fundamental**: Labels HOLD (75-95%) dominam o dataset → modelo aprende a "não fazer nada".

---

### 2. ⚠️ Regression Simples (prever preço futuro)
**Limitações**:
- Prediz apenas 1 variável (preço)
- Não considera risco (TP/SL)
- Não define duração do trade
- Difícil traduzir previsão em ação

**Exemplo**: Modelo prevê preço 1.0850 em 10min
- E daí? Compro ou vendo?
- Qual meu stop loss?
- Quando saio do trade?

---

## ✅ ESTRATÉGIA ESCOLHIDA: PIPELINE EVOLUTIVO

### **FASE 1 (POC - 2 semanas): MULTI-TARGET REGRESSION** 🎯

#### **O que é?**
Prever simultaneamente 4 variáveis críticas:
```python
X (features) → Modelo ML → Y (4 targets):
                           - expected_return (%)
                           - confidence (0-1)
                           - optimal_tp (pips)
                           - optimal_sl (pips)
```

#### **Por que funciona melhor?**
1. **Treina com dados reais**: Usa simulações de trades completos
2. **Quantifica incerteza**: Confidence score = força do sinal
3. **Adaptativo**: TP/SL variam conforme contexto de mercado
4. **Acionável**: Saída direta para decisão de trade

#### **Lógica de Decisão**:
```python
if confidence > 0.65 and expected_return > 0.15%:
    if expected_return > 0:
        SIGNAL = BUY
    else:
        SIGNAL = SELL
    
    TP = optimal_tp
    SL = optimal_sl
else:
    SIGNAL = HOLD
```

#### **Vantagens**:
- ✅ Rápido de implementar (~2 semanas)
- ✅ Interpretável (sabemos o porquê das decisões)
- ✅ Fácil de debugar
- ✅ Usa todo o dataset eficientemente
- ✅ Prova de conceito sólida

#### **Desvantagens**:
- ⚠️ Não otimiza sequências de trades
- ⚠️ Não aprende "timing" automaticamente
- ⚠️ Cada candle é independente

---

### **FASE 2 (PRODUTO - 1-2 meses): REINFORCEMENT LEARNING** 🏆

#### **O que é?**
Agente autônomo que aprende a maximizar lucro através de recompensas.

```
Estado (St) → Agente RL → Ação (At) → Recompensa (Rt)
                ↑__________________________|
                   (aprende otimizar)
```

#### **Como funciona**:
- **Estado**: Features do mercado atual
- **Ações**: BUY, SELL, HOLD, CLOSE_POSITION
- **Recompensa**: PnL acumulado - fees - drawdown penalty
- **Algoritmo**: PPO (Proximal Policy Optimization)

#### **Por que é superior**:
1. **Otimização end-to-end**: Aprende estratégia completa
2. **Gestão de risco automática**: Aprende quando não operar
3. **Sequências de trades**: Entende contexto temporal
4. **Auto-adaptação**: Aprende padrões complexos

#### **Exemplo Real**:
```
Situação: Mercado lateral (baixa volatilidade)
- Multi-target: Gera sinais fracos (muitos HOLD) ✅
- RL: APRENDE a evitar trades + ajustar TP/SL 🏆
```

#### **Desafios**:
- ⚠️ Complexo de implementar
- ⚠️ Lento para treinar (dias/semanas)
- ⚠️ Difícil de debugar
- ⚠️ Pode overfittar (precisa regularização)

---

### **FASE 3 (SCALE - 6+ meses): ENSEMBLE HÍBRIDO** 🚀

#### **O que é?**
Combina múltiplos modelos especializados:

```
┌─────────────────────────────────────────┐
│  Multi-Target Regression (confiança)   │──┐
│  RL Agent (timing otimizado)            │──┤
│  Market Regime Classifier (contexto)    │──┼──> META-MODELO
│  Sentiment Analysis (notícias/social)   │──┤   (decisão final)
│  Order Book Imbalance (microestrutura)  │──┘
└─────────────────────────────────────────┘
```

#### **Componentes**:
1. **Regime Detector**: Detecta tipo de mercado (trending/ranging/volatile)
2. **Especialistas**: 1 modelo por regime
3. **Meta-Learner**: Combina previsões com pesos adaptativos
4. **Risk Manager**: Camada final de validação

#### **Por que é o futuro**:
- ✅ Robustez máxima (diversificação)
- ✅ Aprende quando confiar em cada modelo
- ✅ Incorpora múltiplas fontes de dados
- ✅ Escalável para múltiplos pares

---

## 📋 PASSO A PASSO DE IMPLEMENTAÇÃO

### **FASE 1: MULTI-TARGET REGRESSION (POC - AGORA)**

#### **Semana 1: Preparação de Labels**
```bash
1. ✅ FEITO: Dataset com 64 features (2.1M candles)
2. ✅ FEITO: Teste de 10 configurações TP/SL
3. 🔄 PRÓXIMO: Criar 4 targets realistas
   - expected_return: ROI do trade simulado
   - confidence: Qualidade do setup (volatility, spread, momentum)
   - optimal_tp: TP ajustado por ATR
   - optimal_sl: SL ajustado por suporte/resistência
```

**Scripts**:
```bash
# Ajustar label creator para multi-target
python scripts/create_multitarget_labels.py

# Validar distribuição
python scripts/validate_labels.py --plot
```

---

#### **Semana 2: Treinamento & Validação**
```bash
1. Train/Val/Test split (60/20/20)
2. Treinar XGBoost Multi-output
3. Hyperparameter tuning (Optuna)
4. Backtest completo (2023-2024)
5. Métricas:
   - Return prediction: MAE, R²
   - Confidence calibration: Brier Score
   - TP/SL accuracy: RMSE
   - Backtest: Sharpe, Win Rate, PnL
```

**Scripts**:
```bash
python scripts/train_multitarget.py --config configs/multi_target.yaml
python scripts/backtest_model.py --year 2024 --plot
python scripts/live_simulation.py --duration 7days
```

---

### **FASE 2: REINFORCEMENT LEARNING (PRODUTO - Futuro)**

#### **Mês 1: Setup & Baseline**
1. Ambiente Gym customizado (OpenAI Gym)
2. State/Action/Reward design
3. Baseline com PPO (Stable-Baselines3)
4. Comparação com multi-target

#### **Mês 2: Otimização & Deploy**
1. Curriculum learning (treino progressivo)
2. Ensemble de agentes
3. Backtesting rigoroso
4. Deploy staging

**Recursos Necessários**:
- GPU (treinamento 10-20x mais rápido)
- Compute: AWS/GCP (EC2 p3.2xlarge)
- Tempo: ~100 horas de treinamento

---

### **FASE 3: ENSEMBLE (SCALE - Futuro Distante)**

#### **Trimestre 1: Componentes Individuais**
1. Regime classifier (market state)
2. Sentiment analysis (news/twitter)
3. Order book features
4. Volatility forecasting

#### **Trimestre 2: Meta-Learning**
1. Stacking architecture
2. Adaptive weighting
3. A/B testing em produção
4. Multi-pair expansion

---

## 🎯 DECISÃO FINAL

### **RESPOSTA DIRETA**

Para um robô trader profissional:

#### **Curto prazo (POC - AGORA)**: 
✅ **Multi-target Regression**
- Implementar em 2 semanas
- Prova de conceito funcional
- Base sólida para evoluir

#### **Médio prazo (Produto - Q2 2025)**: 
🏆 **Reinforcement Learning**
- Máxima performance
- Complexidade gerenciável
- Diferencial competitivo

#### **Longo prazo (Scale - Q3-Q4 2025)**: 
🚀 **Ensemble Híbrido**
- Robustez máxima
- Multi-pair, multi-strategy
- Estado da arte

---

## 📊 PIPELINE COMPLETO

```
HOJE (Semana 1-2)
├─ Multi-target labels
├─ XGBoost training
├─ Backtest 2024
└─ Live simulation 7 dias

Q2 2025 (Se POC sucesso)
├─ RL environment
├─ PPO training
├─ Ensemble RL agents
└─ Staging deploy

Q3-Q4 2025 (Produto maduro)
├─ Regime detection
├─ Sentiment analysis
├─ Meta-learning
└─ Production scale
```

---

## ✅ PRÓXIMOS PASSOS IMEDIATOS

1. **Ajustar `label_creator.py`**:
   - Adicionar cálculo de confidence
   - Adicionar optimal_tp/optimal_sl adaptativos
   - Exportar 4 targets

2. **Criar `train_multitarget.py`**:
   - XGBoost MultiOutputRegressor
   - Cross-validation temporal
   - Feature importance analysis

3. **Criar `backtest_ml.py`**:
   - Simular trades com outputs do modelo
   - Calcular métricas realistas
   - Comparar com buy-and-hold

4. **Dashboard Streamlit**:
   - Visualizar previsões vs real
   - Confidence distribution
   - Equity curve

---

**Pronto para começar? 🚀**

Execute:
```bash
python scripts/create_multitarget_labels.py
```
