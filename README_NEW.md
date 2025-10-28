# 🚀 TradeSigns - Ensemble ML Trading Bot

Sistema de trading automatizado usando **Machine Learning Ensemble** com paralelização GPU + Multi-CPU.

## 🎯 OBJETIVO

Bot de trading lucrativo com:
- **150+ features** técnicas avançadas
- **50+ modelos** combinados em ensemble
- **6 horizontes** de previsão (5m, 15m, 30m, 1h, 4h, 1d)
- **Risk-aware** (prevê upside E downside)
- **Meta de lucro**: 8-15% ao mês | Sharpe > 1.5

## 📊 STATUS ATUAL

**Fase**: Preparação para Ensemble Training  
**Progresso**: 35% completo  

✅ Setup + GPU configurada  
✅ 2.1M candles históricos (2020-2025)  
✅ Baseline testado (R² negativo)  
✅ Scripts avançados criados  
🔴 **PRÓXIMO**: Preparar dataset ensemble (150+ features)

## 🚀 QUICK START

### 1. Setup (10 min)
```bash
git clone https://github.com/frankduque/tradesigns-poc-live.git
cd tradesigns-poc-live
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Preparar Dataset Ensemble (30-60 min)
```bash
python scripts\prepare_ensemble_dataset.py
```

### 3. Treinar 500 Modelos em Paralelo (3-4h GPU)
```bash
python scripts\train_ensemble_parallel.py
```

### 4. Backtest (5 min)
```bash
python scripts\backtest_ensemble.py --model data/models/ensemble_XXX.pkl
```

### 5. Deploy (se backtest for bom)
```bash
python scripts\generate_signals_ensemble.py --model data/models/ensemble_XXX.pkl
streamlit run dashboard\ensemble_monitor.py
```

## 📖 DOCUMENTAÇÃO

📄 **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Status completo + próximos passos  
📄 **[docs/](docs/)** - Guias técnicos detalhados

## 🎮 HARDWARE

**GPU**: NVIDIA GTX 1060 3GB (CUDA 12.8) ✅  
**Paralelismo**: 4 workers + GPU = 500 modelos em 3-4h

| Task | CPU | GPU + Paralelo |
|------|-----|----------------|
| Feature engineering | 30 min | 5 min |
| Treinar 500 modelos | 3 dias | 3-4 horas ⚡ |
| Backtest | 10 min | 10 min |

## 🏆 DIFERENCIAL

- ✅ **50 modelos** (vs 1 da maioria)
- ✅ **6 horizontes** (scalping + day trade + swing)
- ✅ **Risk-aware** (prevê upside/downside)
- ✅ **Retreino diário** (GPU rápida = sempre atualizado)
- ✅ **Ensemble stacking** (inteligência coletiva)

## ⚠️ NOTAS

- Dados históricos (~2GB) NÃO estão no Git
- GPU necessária para treino rápido (CPU funciona mas demora 50x mais)
- Win Rate esperado: 55-60% (nada é garantido em trading!)

**Status**: 🟡 Em desenvolvimento ativo
