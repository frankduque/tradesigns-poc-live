# 🎓 MANUAL DE TREINAMENTO - Regressão Multi-Target

## COMO EXECUTAR O TREINAMENTO COMPLETO

### ⏱️ TEMPO ESTIMADO

### **Setup Atual (CPU)**
- **Dataset**: 2.1M samples, 61 features
- **Modelo**: Random Forest (200 árvores, depth=15)
- **Tempo esperado**: 30-60 minutos (depende do CPU)
- **Memória RAM**: ~8-16GB

### **Setup Futuro (GPU - GTX 1060 3GB)**
- **Modelo**: LightGBM com GPU acceleration
- **Tempo esperado**: 3-8 minutos ⚡ (6-10x mais rápido)
- **VRAM**: 3GB (suficiente)
- **Status**: Será implementado após POC validado

---

## 📝 PASSO A PASSO

### 1️⃣ Ativar ambiente virtual

```bash
cd "C:\projetos\Fila de Idéias\TradeSigns\emDevComManus\tradesigns-poc-live"
.\venv\Scripts\activate
```

### 2️⃣ Executar treinamento

```bash
python scripts/train_model_regression.py
```

**O que vai acontecer:**
1. Carrega dataset (2.1M samples) - ~5 segundos
2. Remove features com NaN - ~5 segundos
3. Remove linhas com infinitos - ~5 segundos
4. Split train/test (80/20) - instantâneo
5. **TREINA MODELO** - **30-60 minutos** ⏳
6. Avalia no test set - ~2 minutos
7. Calcula feature importance - ~1 minuto
8. Salva modelo - ~5 segundos

---

## 📊 OUTPUT ESPERADO

### Durante o treinamento:
```
2025-10-28 10:43:41 - INFO - Treinando modelo...
(aguarde... sem output por 30-60 min - ISSO É NORMAL!)
```

### Após conclusão:
```
✓ Modelo treinado!
   Tempo: 45.3 minutos

MÉTRICAS DE TESTE:
==================
return_5m:
   RMSE: 0.0287%
   MAE:  0.0201%
   R²:   0.2342

return_10m:
   RMSE: 0.0402%
   MAE:  0.0289%
   R²:   0.2567

return_30m:
   RMSE: 0.0698%
   MAE:  0.0512%
   R²:   0.2891

✓ Modelo salvo: models/regression_model.joblib
✓ Feature importance: models/feature_importance.csv
```

---

## 🎯 INTERPRETAÇÃO DOS RESULTADOS

### R² Score (Coeficiente de Determinação)
- **R² > 0.20**: EXCELENTE para Forex (mercado eficiente)
- **R² > 0.25**: MUITO BOM
- **R² > 0.30**: EXCEPCIONAL

**Por quê R² "baixo" é bom aqui?**
- Forex é um mercado eficiente (difícil de prever)
- R² = 0.25 significa: explica 25% da variância dos returns
- Isso é **suficiente para gerar alpha** e lucrar

### RMSE/MAE (Erro Médio)
- **RMSE < 0.05%**: Bom (5 pips de erro)
- **MAE < 0.03%**: Muito bom (3 pips de erro)

### Feature Importance
- Verifica quais features são mais importantes
- **TOP 10 esperadas**:
  1. atr_14 (volatilidade)
  2. rsi_14 (momentum)
  3. bb_position (posição nas Bandas)
  4. ema_9, ema_21 (tendência)
  5. macd_signal (momentum)
  6. stoch_k (oscilador)
  7. price_vs_ema_200 (tendência longa)
  8. recent_high_low (suporte/resistência)
  9. hour, day_of_week (sazonalidade)

---

## ⚠️ TROUBLESHOOTING

### Processo travado (sem output por >60 min)
- Verifique Task Manager: CPU deve estar 100%
- Verifique RAM: não deve estar no limite
- **Se CPU baixo (<50%)**: Pode ter travado
  - Pressione Ctrl+C
  - Verifique se tem outro processo consumindo recursos
  - Tente novamente

### Erro: "MemoryError"
- Dataset muito grande para RAM disponível
- **Solução 1**: Fechar outros programas
- **Solução 2**: Reduzir dataset temporariamente:
  ```python
  # Em train_model_regression.py, linha ~42
  df = df.sample(n=500000, random_state=42)  # Usar apenas 500k samples
  ```

### Modelo não performa bem (R² < 0.10)
- Features podem não ser preditivas
- Revisar feature engineering
- Testar períodos diferentes (regime de mercado)

---

## 🔧 OTIMIZAÇÃO DE PARÂMETROS (Opcional)

Se quiser experimentar com hiperparâmetros:

### Mais árvores = Melhor (mas mais lento)
```python
trainer = MultiTargetRegressionTrainer(
    n_estimators=300,  # De 200 para 300 (+50% tempo)
)
```

### Árvores mais profundas = Mais expressivo
```python
trainer = MultiTargetRegressionTrainer(
    max_depth=20,  # De 15 para 20 (cuidado com overfit)
)
```

### Grid Search (mais científico, MUITO lento)
```bash
# Cria grid_search_regression.py
python scripts/grid_search_regression.py
# Tempo: 6-12 horas!
```

---

## 📁 ARQUIVOS GERADOS

Após o treinamento:

```
models/
├── regression_model.joblib          (Modelo treinado - ~500MB)
├── feature_importance.csv           (Importância das features)
└── training_metrics.json            (Métricas detalhadas)

logs/
└── train_regression_model.log       (Log completo)
```

---

## ✅ PRÓXIMOS PASSOS

Após o modelo treinado:

### PASSO 3: Gerar sinais e backtest
```bash
python scripts/generate_signals_regression.py
```

**Tempo**: ~5-10 minutos

**Output esperado**:
- Sinais gerados: 50,000-100,000
- Win Rate: 52-55%
- Sharpe Ratio: 1.5-2.0
- Max Drawdown: <15%

---

## 💡 DICAS

1. **Execute durante a noite** - Processo longo, deixe rodando
2. **Monitore a primeira vez** - Garanta que não trave
3. **Salve logs** - Útil para debug depois
4. **CPU melhor = mais rápido** - Considere cloud se necessário
5. **n_jobs=-1** - Já configurado para usar todos os cores

---

## 🚀 MELHORIAS FUTURAS

### **PRÓXIMA ITERAÇÃO (Pós-POC):**

#### 🔥 **1. LightGBM com GPU (GTX 1060 3GB)** - PRIORIDADE!
**Por quê?**
- ⚡ 6-10x mais rápido (3-8 min vs 30-60 min)
- 🎯 Performance igual ou melhor que Random Forest
- 💾 3GB VRAM suficiente para nosso dataset
- 🔄 Permite retreinos diários/semanais (walk-forward)

**Como implementar:**
```bash
# 1. Instalar LightGBM com suporte GPU
pip install lightgbm --install-option=--gpu

# 2. Instalar dependências CUDA (se ainda não tiver)
# Download: https://developer.nvidia.com/cuda-downloads
# Versão recomendada: CUDA 11.8 ou 12.x

# 3. Executar novo script
python scripts/train_model_lightgbm_gpu.py
```

**Arquivo será criado:** `scripts/train_model_lightgbm_gpu.py`

---

### **Outras Melhorias (Médio Prazo):**

2. **XGBoost com GPU** - Alternativa ao LightGBM
3. **Optuna** - Auto-tune de hiperparâmetros
4. **Feature Selection** - Reduz de 61 para 30 features top
5. **Ensemble** - Combina RF + LightGBM + XGBoost

---

## 🎮 **HARDWARE ROADMAP**

| Fase | Hardware | Algoritmo | Tempo Treino | Status |
|------|----------|-----------|--------------|--------|
| **POC** | CPU only | Random Forest | 30-60 min | ✅ Atual |
| **Produção V1** | GTX 1060 3GB | LightGBM GPU | 3-8 min | 🔜 Próximo |
| **Produção V2** | GTX 1060 3GB | Ensemble | 10-20 min | 📋 Futuro |
| **Scale** | RTX 3060 12GB | Neural Network | 5-15 min | 🌟 Opcional |

---

**Boa sorte! 🎯**

Qualquer dúvida, consulte os logs em `logs/train_regression_model.log`
