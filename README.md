# 🎯 TradeSigns POC - Multi-Target Regression

Sistema de ML para geração de sinais de trading (EUR/USD) usando regressão multi-target.

## 🚀 QUICK START

\\\ash
# Setup
git clone https://github.com/SEU_USUARIO/tradesigns-poc-live.git
cd tradesigns-poc-live
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Treinar (GPU)
python scripts/train_model_lightgbm_gpu.py  # 5-8 min ⚡

# Gerar sinais
python scripts/generate_signals_regression.py
\\\

## 📊 STATUS

- ✅ Dataset preparado (2.1M samples, 61 features)
- ✅ Labels de regressão criadas
- 🎮 Migração para GPU em progresso

## 📖 DOCUMENTAÇÃO

- \docs/REGRESSION_QUICKSTART.md\ - Start em 5 min
- \docs/GPU_SETUP_GUIDE.md\ - Configurar GPU
- \docs/MIGRATION_GUIDE.md\ - Migrar entre PCs

## ⚠️ IMPORTANTE

**Dados NÃO estão no Git** (~2GB)
Transferir de outro PC ou processar do zero

## 🎮 GPU

| Hardware | Tempo |
|----------|-------|
| CPU | 30-60 min |
| GTX 1060 3GB | 5-8 min ⚡ |

**Status**: 🟢 POC em desenvolvimento
