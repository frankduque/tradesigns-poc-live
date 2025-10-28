# 🤖 Decisão de Abordagem de Machine Learning

## 📋 CONTEXTO

Após análise dos resultados da otimização de labels:

```
RANK   CONFIGURACAO                      TP    SL    DUR   WIN%    LOSS%   HOLD%   SCORE
==========================================================================================
1      Scalping Ultra (5pip/10min)      5     5     10    11.7    12.9    75.4    54.3
2      Scalping Agressivo (8pip/15min)  8     8     15    8.4     9.1     82.4    50.3
3      Scalping Balanced (10pip/20min)  10    10    20    7.7     8.2     84.1    49.4
```

**Observações:**
- Melhor configuração: TP=5 pips, SL=5 pips, Duration=10min
- WIN: 11.7% | LOSS: 12.9% | HOLD: 75.4%
- Score: 54.3/100

**Problema Identificado:**
As labels fixas (TP/SL/Duration) são escolhas arbitrárias que NÃO refletem o comportamento real do mercado.

---

## 🎯 ABORDAGENS COMPARADAS

### ❌ ABORDAGEM 1: Classification com Labels Fixas (ATUAL)

**Como funciona:**
1. Define TP=5 pips, SL=5 pips, Duration=10min
2. Simula trades: se bater TP → WIN (1), se bater SL → LOSS (-1), senão → HOLD (0)
3. Treina modelo para prever {-1, 0, 1}

**Problemas:**
- **Arbitrário**: Por que 5 pips? Por que 10 minutos?
- **Inflexível**: Mercado não respeita nossas regras
- **Desbalanceado**: 75% HOLD, difícil de treinar
- **Otimização manual**: Testar 10 combinações não é eficiente
- **Pré-decisão**: Estamos DECIDINDO o resultado antes do modelo aprender

**Resultado:**
- Win Rate baixo (11.7%)
- Muitos holds (75%)
- Score médio (54/100)

---

### ✅ ABORDAGEM 2: Regression para Previsão de Movimento (RECOMENDADA)

**Como funciona:**
1. **Label = Variação real do preço** em N minutos futuros
   - Exemplo: `label = (close[t+10] - close[t]) / close[t] * 100`
   - Em vez de {-1, 0, 1}, temos valores contínuos: [-0.05, +0.08, +0.02, ...]

2. **Modelo prevê movimento esperado**: `predicted_change = model.predict(features)`
   
3. **Decisão baseada no output**:
   - Se `predicted_change > +threshold`: **BUY**
   - Se `predicted_change < -threshold`: **SELL**
   - Senão: **HOLD**

4. **Threshold dinâmico**: Ajustado pela volatilidade (ATR)
   - Mercado volátil (ATR alto): threshold maior
   - Mercado calmo (ATR baixo): threshold menor

**Vantagens:**
- ✅ **Aprende do mercado real**: Não impomos regras
- ✅ **Captura magnitude**: Sabe se é movimento pequeno (+0.02%) ou grande (+0.15%)
- ✅ **Flexível**: Threshold adaptativo por contexto
- ✅ **Dados balanceados**: Distribuição gaussiana natural
- ✅ **Confiança**: Pode medir incerteza da previsão

**Desvantagens:**
- Precisa de mais dados (mas temos 2M+ candles)
- Treinamento um pouco mais lento (mas aceitável)

---

### ⚡ ABORDAGEM 3: Multi-Target Regression (AVANÇADA)

**Como funciona:**
1. Prevê MÚLTIPLOS horizontes simultaneamente:
   - `y1 = variação em 5 minutos`
   - `y2 = variação em 10 minutos`
   - `y3 = variação em 30 minutos`
   - `y4 = variação máxima até 30min (upside potential)`
   - `y5 = variação mínima até 30min (downside risk)`

2. **Decisão sofisticada**:
   ```python
   if (y1 > 0.05% and y2 > 0.08% and y4 > 0.15% and y5 > -0.05%):
       signal = BUY  # Upside alto, downside protegido
   ```

**Vantagens:**
- ✅ **Captura dinâmica completa**: Não só direção, mas timing e magnitude
- ✅ **Risk-aware**: Sabe quando movimento é arriscado
- ✅ **Otimização de saída**: Pode prever melhor momento de take profit

**Desvantagens:**
- Complexo de implementar
- Requer mais capacidade computacional
- Pode overfit se não tiver dados suficientes

---

### 🧠 ABORDAGEM 4: Deep Learning (LSTM/Transformers) (FUTURA)

**Como funciona:**
- LSTM ou Transformer para capturar padrões temporais complexos
- Input: Sequência de candles (ex: últimos 100 candles)
- Output: Previsão de movimento

**Quando usar:**
- **Após validar abordagem 2 ou 3**
- Se tiver milhões de dados
- Se performance com ML tradicional platear

**Vantagens:**
- ✅ Captura padrões não-lineares complexos
- ✅ Aprende features automaticamente

**Desvantagens:**
- ❌ Black box (difícil explicar)
- ❌ Requer muito mais dados e GPU
- ❌ Propenso a overfit
- ❌ Overkill para Forex (eficiência de mercado)

---

## 🎯 DECISÃO FINAL

### **ESCOLHIDA: ABORDAGEM 2 - Regression para Previsão de Movimento**

**Justificativa:**
1. **Mais realista**: Aprende do comportamento real do mercado
2. **Balanceamento natural**: Distribuição de returns é normal (gaussiana)
3. **Flexível**: Threshold adaptativo por volatilidade
4. **Interpretável**: Sabemos o que o modelo está prevendo
5. **Escalável**: Pode evoluir para abordagem 3 depois
6. **Comprovado**: Usado por quants em fundos reais

**Referências:**
- "Machine Learning for Algorithmic Trading" - Stefan Jansen
- Papers de quant finance usam regression para forecast de returns
- Renaissance Technologies (hedge fund) usa modelos similares

---

## 📋 IMPLEMENTAÇÃO

### Passo 1: Criar Labels de Regression

```python
# src/ml/label_creator_regression.py

def create_regression_labels(df: pd.DataFrame, horizons: list = [5, 10, 30]) -> pd.DataFrame:
    """
    Cria labels de regressão = variação % do preço em N minutos
    
    Args:
        df: DataFrame com OHLCV
        horizons: Lista de horizontes em minutos [5, 10, 30]
    
    Returns:
        DataFrame com colunas: return_5m, return_10m, return_30m
    """
    df = df.copy()
    
    for h in horizons:
        # Variação percentual
        df[f'return_{h}m'] = (df['close'].shift(-h) - df['close']) / df['close'] * 100
        
        # Variação máxima (upside potential)
        df[f'max_return_{h}m'] = (
            df['high'].rolling(h).max().shift(-h) - df['close']
        ) / df['close'] * 100
        
        # Variação mínima (downside risk)
        df[f'min_return_{h}m'] = (
            df['low'].rolling(h).min().shift(-h) - df['close']
        ) / df['close'] * 100
    
    # Remove linhas sem label (final do dataset)
    max_horizon = max(horizons)
    df = df.iloc[:-max_horizon]
    
    return df
```

### Passo 2: Treinar Modelo de Regression

```python
# src/ml/trainer_regression.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

class RegressionTrainer:
    def __init__(self):
        # Modelo para múltiplos targets
        self.model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=100,
                min_samples_leaf=50,
                n_jobs=-1,
                random_state=42
            )
        )
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Treina modelo para prever múltiplos horizontes
        
        Args:
            X: Features (64 colunas)
            y: Targets [return_5m, return_10m, return_30m] (3 colunas)
        """
        self.model.fit(X, y)
        
        # Feature importance
        importances = self.model.estimators_[0].feature_importances_
        
        return importances
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna previsões: [return_5m, return_10m, return_30m]
        """
        return self.model.predict(X)
```

### Passo 3: Signal Generator com Threshold Dinâmico

```python
# src/ml/signal_generator_regression.py

class RegressionSignalGenerator:
    def __init__(self, model, base_threshold: float = 0.05):
        self.model = model
        self.base_threshold = base_threshold  # 0.05% = 5 pips
    
    def generate_signals(self, df: pd.DataFrame, features: np.ndarray) -> pd.Series:
        """
        Gera sinais baseado em previsão de movimento
        
        Returns:
            Series com {-1, 0, 1}
        """
        # Previsão
        predictions = self.model.predict(features)  # [return_5m, return_10m, return_30m]
        
        # Usar previsão de 10min (médio prazo)
        predicted_return = predictions[:, 1]  # return_10m
        
        # Threshold adaptativo baseado em ATR
        atr = df['atr'].values
        atr_normalized = atr / df['close'].values  # ATR como % do preço
        
        # Threshold = base * (1 + atr_normalized)
        # Quando ATR alto → threshold maior (mais conservador)
        dynamic_threshold = self.base_threshold * (1 + atr_normalized * 2)
        
        # Gerar sinais
        signals = np.zeros(len(predicted_return))
        signals[predicted_return > dynamic_threshold] = 1   # BUY
        signals[predicted_return < -dynamic_threshold] = -1 # SELL
        
        return pd.Series(signals, index=df.index)
    
    def generate_signals_advanced(self, df: pd.DataFrame, features: np.ndarray) -> pd.Series:
        """
        Versão avançada: considera upside/downside
        """
        predictions = self.model.predict(features)
        
        # Desempacotar previsões
        return_5m = predictions[:, 0]
        return_10m = predictions[:, 1]
        return_30m = predictions[:, 2]
        
        # Se modelo previu upside/downside (multi-target regression)
        # max_return_30m = predictions[:, 3]
        # min_return_30m = predictions[:, 4]
        
        signals = np.zeros(len(return_10m))
        
        # BUY: movimento positivo consistente + upside > downside
        buy_condition = (
            (return_5m > 0.03) &  # 3 pips em 5min
            (return_10m > 0.05) &  # 5 pips em 10min
            (return_30m > 0.08)    # 8 pips em 30min
        )
        
        # SELL: movimento negativo consistente
        sell_condition = (
            (return_5m < -0.03) &
            (return_10m < -0.05) &
            (return_30m < -0.08)
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return pd.Series(signals, index=df.index)
```

### Passo 4: Backtest com Take Profit Dinâmico

```python
# src/ml/backtester_dynamic.py

class DynamicBacktester:
    def backtest_with_predictions(self, df: pd.DataFrame, signals: pd.Series, predictions: np.ndarray):
        """
        Backtest onde TP/SL são baseados na PREVISÃO do modelo
        
        Args:
            predictions: [return_5m, return_10m, return_30m]
        """
        trades = []
        
        for i in range(len(signals)):
            if signals.iloc[i] == 0:
                continue
            
            signal_type = 'BUY' if signals.iloc[i] == 1 else 'SELL'
            entry_price = df['close'].iloc[i]
            predicted_move = predictions[i, 1]  # return_10m
            
            # TP/SL baseados na PREVISÃO
            if signal_type == 'BUY':
                take_profit = entry_price * (1 + abs(predicted_move) / 100)
                stop_loss = entry_price * (1 - abs(predicted_move) / 100 * 0.5)  # SL = metade do TP
            else:
                take_profit = entry_price * (1 - abs(predicted_move) / 100)
                stop_loss = entry_price * (1 + abs(predicted_move) / 100 * 0.5)
            
            # Simular trade
            outcome = self.simulate_trade(
                df.iloc[i:i+30],  # Próximos 30 candles
                entry_price,
                take_profit,
                stop_loss,
                signal_type
            )
            
            trades.append(outcome)
        
        return self.calculate_metrics(trades)
```

---

## 📊 COMPARAÇÃO DE RESULTADOS ESPERADOS

### Classification com Labels Fixas (Atual)
```
Win Rate: 11.7%
Signals/day: ~50 (muitos holds)
Sharpe: ???
Max Drawdown: ???
```

### Regression Esperado (Meta)
```
Win Rate: 50-55% (mais realista)
Signals/day: 20-30 (mais seletivo)
Sharpe: > 1.5
Max Drawdown: < 15%
Avg Return per Trade: 0.08-0.12% (8-12 pips)
```

**Por quê esperamos melhora?**
- Modelo aprende magnitude do movimento
- Threshold dinâmico reduz sinais em mercado volátil
- TP/SL adaptativos ao contexto
- Labels menos arbitrárias

---

## 🚀 PLANO DE IMPLEMENTAÇÃO

### Sprint 1: Preparação (2-3 dias)
- [ ] Criar `label_creator_regression.py`
- [ ] Gerar labels de regression para dataset
- [ ] Análise exploratória dos returns
- [ ] Verificar distribuição (deve ser ~normal)

### Sprint 2: Treinamento (2-3 dias)
- [ ] Implementar `trainer_regression.py`
- [ ] Treinar modelo de regression
- [ ] Validação cross-validation
- [ ] Análise de feature importance

### Sprint 3: Signal Generation (2 dias)
- [ ] `signal_generator_regression.py` com threshold dinâmico
- [ ] Testar geração de sinais
- [ ] Ajustar thresholds

### Sprint 4: Backtest (3-4 dias)
- [ ] Backtest com TP/SL dinâmicos
- [ ] Comparar com abordagem classificação
- [ ] Walk-forward validation
- [ ] Análise de métricas

### Sprint 5: Otimização (2-3 dias)
- [ ] Grid search para base_threshold
- [ ] Testar múltiplos horizontes
- [ ] Feature engineering adicional se necessário

**Total: 11-15 dias**

---

## 🔬 PRÓXIMOS EXPERIMENTOS

### Após Regression Funcionar:
1. **Multi-Target Regression** (upside/downside)
2. **Ensemble de Modelos** (RF + XGBoost + LightGBM)
3. **Features de Volatility Regime** (low/medium/high vol)
4. **Reinforcement Learning** (agente aprende política ótima)
5. **Deep Learning** (LSTM se tiver GPU)

### Métricas de Sucesso:
- Win Rate > 52%
- Sharpe > 1.5
- Max DD < 15%
- Profit Factor > 1.5
- Calmar Ratio > 1.0

---

## 📚 REFERÊNCIAS

1. **Livros:**
   - "Advances in Financial Machine Learning" - Marcos López de Prado
   - "Machine Learning for Algorithmic Trading" - Stefan Jansen

2. **Papers:**
   - "The Cross-Section of Expected Stock Returns" (Fama-French)
   - "Do We Need Hundreds of Classifiers to Solve Real World Classification Problems?" (Fernández-Delgado)

3. **Blogs/Artigos:**
   - QuantStart: Regression for Trading
   - Machine Learning Mastery: Multi-Output Regression

---

## ✅ CONCLUSÃO

**Regression >> Classification para nosso caso porque:**

1. ✅ Labels aprendem do mercado real (não arbitrárias)
2. ✅ Captura magnitude do movimento
3. ✅ Threshold dinâmico por volatilidade
4. ✅ Balanceamento natural dos dados
5. ✅ Flexível e escalável

**Próximo passo:** Começar Sprint 1 - implementar labels de regression.

**Pergunta chave resolvida:** Sim, existe abordagem melhor que regression tradicional (multi-target), mas começamos com single-target regression que já é superior à classification.

---

## 🎯 PIPELINE DE EVOLUÇÃO DO SISTEMA

### FASE 1 - POC (Atual - 2-3 semanas)
**Abordagem:** Multi-target Regression
- **Prevê:** price_change, volatility, trend_strength
- **Modelo:** XGBoost/LightGBM
- **Deploy:** Python script local
- **Objetivo:** Validar conceito e métricas
- **Métricas alvo:** Win Rate > 52%, Sharpe > 1.5

### FASE 2 - Produto (Médio Prazo - 1-2 meses)
**Abordagem:** Reinforcement Learning
- **Agente:** Aprende estratégia ótima de trading
- **Ambiente:** Simulação realista de mercado com slippage/spreads
- **Deploy:** API REST + Redis para cache
- **Objetivo:** Sistema comercial robusto e adaptável
- **Features:** Dashboard web, notificações, múltiplos pares

### FASE 3 - Scale (Longo Prazo - 3-6 meses)
**Abordagem:** Ensemble Híbrido
- **Combina:** Multi-target + RL + modelos especializados por regime
- **Orquestração:** Kubernetes para escalabilidade
- **Deploy:** Cloud distribuído (AWS/GCP)
- **Objetivo:** Performance máxima e escalabilidade para milhares de usuários
- **Features:** Auto-ML, A/B testing, personalização por usuário
