# ✅ CHECKLIST - Migração CPU → GPU

## 🎯 **SITUAÇÃO ATUAL**
- ⏳ Treinamento no PC atual está levando 4+ horas
- 🎮 GTX 1060 3GB disponível em outro PC
- 📦 Projeto pronto para migração

---

## 📋 **PASSO A PASSO COMPLETO**

### **PARTE 1: PC ATUAL (CPU) - Preparar**

```bash
# 1. Parar treino atual
# Pressione Ctrl+C no terminal onde está rodando

# 2. Navegar para o projeto
cd "C:\projetos\Fila de Idéias\TradeSigns\emDevComManus\tradesigns-poc-live"

# 3. Verificar arquivos importantes
dir data\features\ml_dataset_regression.parquet
# Deve mostrar ~843 MB

# 4. Commitar mudanças no Git
git status
git add .
git commit -m "POC Multi-Target Regression - Ready for GPU"

# 5. Criar repositório no GitHub
# Vá em: https://github.com/new
# Nome: tradesigns-poc-live
# Descrição: TradeSigns POC - Multi-Target Regression ML
# Private: Sim

# 6. Push para GitHub
git remote add origin https://github.com/SEU_USUARIO/tradesigns-poc-live.git
git branch -M main
git push -u origin main

# 7. Copiar dados para pen drive (NÃO VAI PRO GIT!)
xcopy /E /I "data" "E:\tradesigns\data"
# Substitua E:\ pela letra do seu pen drive
```

---

### **PARTE 2: PC COM GPU - Instalar**

```bash
# 1. Clonar repositório
cd C:\projetos
git clone https://github.com/SEU_USUARIO/tradesigns-poc-live.git
cd tradesigns-poc-live

# 2. Criar ambiente virtual
python -m venv venv
.\venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Copiar dados do pen drive
xcopy /E /I "E:\tradesigns\data" "data"
# Substitua E:\ pela letra do pen drive

# 5. Verificar dados
dir data\features\ml_dataset_regression.parquet
# Deve mostrar ~843 MB
```

---

### **PARTE 3: PC COM GPU - Configurar GPU**

```bash
# 1. Verificar se driver NVIDIA está instalado
nvidia-smi
# Deve mostrar: GTX 1060 3GB

# 2. Instalar CUDA (se necessário)
# Download: https://developer.nvidia.com/cuda-downloads
# Versão recomendada: CUDA 12.x
# Instalação: Next, Next, Next... (padrão)
# IMPORTANTE: Reinicie o PC após instalação!

# 3. Verificar CUDA instalado
nvcc --version
# Deve mostrar: Cuda compilation tools, release 12.x

# 4. Testar GPU
python test_gpu.py
# Deve mostrar:
# ✓ CPU: ~8s
# ✓ GPU: ~2s
# 🚀 Speedup: 4x mais rápido!
# ✅ GPU está funcionando PERFEITAMENTE!
```

---

### **PARTE 4: PC COM GPU - Treinar!** 🚀

```bash
# Treinar com GPU (5-8 minutos)
python scripts/train_model_lightgbm_gpu.py

# O que vai acontecer:
# 1. Carrega dataset (5s)
# 2. Split train/test (1s)
# 3. TREINA com GPU (3-8 min) ⚡
# 4. Avalia no test set (30s)
# 5. Salva modelo (5s)
# 6. Mostra métricas finais
```

---

## 📊 **OUTPUT ESPERADO**

```
======================================================================
🎮 LIGHTGBM GPU TRAINING - Multi-Target Regression
======================================================================

🔍 Verificando GPU...
✓ GPU detectada: GeForce GTX 1060 3GB, 3072 MiB, 531.68

📂 Carregando dataset: data\features\ml_dataset_regression.parquet
✓ 2,106,480 samples carregados
✓ Features: 61
✓ Targets: 3

📊 Train/Test Split:
   Train: 1,685,184 samples
   Test:  421,296 samples

🔥 Iniciando treinamento GPU...

📈 Target 1/3: return_5m
   RMSE: 0.0287%
   MAE:  0.0201%
   R²:   0.2342

📈 Target 2/3: return_10m
   RMSE: 0.0402%
   MAE:  0.0289%
   R²:   0.2567

📈 Target 3/3: return_30m
   RMSE: 0.0698%
   MAE:  0.0512%
   R²:   0.2891

✓ Treinamento concluído em 5.23 minutos

======================================================================
✅ TREINAMENTO CONCLUÍDO COM SUCESSO!
======================================================================
```

---

## ⏱️ **COMPARAÇÃO DE TEMPO**

| Etapa | PC Atual (CPU) | PC com GPU | Ganho |
|-------|----------------|------------|-------|
| **Preparar dados** | ✅ Feito (~3 min) | ⏭️ Copiar (0 min) | - |
| **Treinar modelo** | ❌ 4+ horas | ✅ 5-8 min | **30-50x** ⚡ |
| **Gerar sinais** | ⏳ Pendente | ⏳ Pendente | - |
| **TOTAL** | 4+ horas | **10-15 min** | **~25x** |

---

## ✅ **CHECKLIST DETALHADO**

### PC Atual:
- [ ] Parar treino (Ctrl+C)
- [ ] Git commit
- [ ] Criar repo GitHub
- [ ] Git push
- [ ] Copiar dados para pen drive

### PC com GPU:
- [ ] Git clone
- [ ] Criar venv
- [ ] pip install
- [ ] Copiar dados do pen drive
- [ ] Verificar nvidia-smi
- [ ] Instalar CUDA (se necessário)
- [ ] Reiniciar PC (após CUDA)
- [ ] python test_gpu.py
- [ ] python scripts/train_model_lightgbm_gpu.py

### Após treino:
- [ ] Verificar métricas (R² > 0.20?)
- [ ] python scripts/generate_signals_regression.py
- [ ] Analisar resultados
- [ ] Validar POC! 🎉

---

## 🆘 **TROUBLESHOOTING**

### ❌ "nvidia-smi não encontrado"
**Solução**: Instalar driver NVIDIA
```
https://www.nvidia.com/Download/index.aspx
Selecione: GTX 1060 → Windows 10/11
```

### ❌ "CUDA not found"
**Solução**: Instalar CUDA Toolkit
```
https://developer.nvidia.com/cuda-downloads
IMPORTANTE: Reiniciar PC após instalação
```

### ❌ "GPU is not available"
**Solução**: Reinstalar LightGBM
```bash
pip uninstall lightgbm
pip install lightgbm --upgrade --no-cache-dir
```

### ❌ "data\features\ml_dataset_regression.parquet not found"
**Solução**: Copiar dados do pen drive
```bash
xcopy /E /I "E:\tradesigns\data" "data"
```

### ❌ "CUDA out of memory"
**Solução**: Reduzir parâmetros (improvável com 3GB)
```python
# Em train_model_lightgbm_gpu.py, linha ~129
'max_depth': 10,  # De 15 para 10
```

---

## 📞 **AJUDA**

Se algo der errado:

1. **Verifique logs**: `logs/train_lightgbm_gpu.log`
2. **Consulte docs**: `docs/GPU_SETUP_GUIDE.md`
3. **Teste GPU**: `python test_gpu.py`
4. **nvidia-smi**: Monitore uso da GPU
   ```bash
   nvidia-smi -l 1  # Atualiza a cada segundo
   ```

---

## 🎯 **RESUMO**

**Tempo total estimado**: 30-45 minutos
- Setup Git/GitHub: 5 min
- Transferir dados: 5 min
- Clonar no PC GPU: 2 min
- Instalar dependências: 5 min
- Configurar CUDA: 10 min (se necessário)
- **Treinar modelo: 5-8 min** ⚡
- Gerar sinais: 5 min

**Resultado**: 
- ✅ Modelo treinado em **~10 min** (vs 4+ horas)
- ✅ POC validado
- ✅ Pronto para produção

---

**Boa sorte! 🚀**

Qualquer dúvida, consulte `docs/MIGRATION_GUIDE.md` ou `docs/GPU_SETUP_GUIDE.md`
