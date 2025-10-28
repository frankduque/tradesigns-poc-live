# 🔄 GUIA DE MIGRAÇÃO - CPU → GPU

## 📋 SITUAÇÃO ATUAL

- ⏳ Treinamento no PC atual (CPU) está levando 4+ horas
- 🎮 GPU (GTX 1060 3GB) está em outro PC
- 🔄 Precisamos migrar o projeto para o PC com GPU

---

## 🚀 PASSO A PASSO - MIGRAÇÃO

### **1️⃣ Preparar repositório (PC atual)**

```bash
# Navegar para a raiz do projeto
cd "C:\projetos\Fila de Idéias\TradeSigns\emDevComManus\tradesigns-poc-live"

# Inicializar Git (se ainda não tiver)
git init

# Criar .gitignore
# (arquivo já criado abaixo)

# Add e commit
git add .
git commit -m "POC Multi-Target Regression - Ready for GPU training"

# Conectar ao GitHub
git remote add origin https://github.com/frankduque/tradesigns-poc-live.git
git branch -M main
git push -u origin main
```

---

### **2️⃣ Clonar no PC com GPU**

```bash
# No PC com GTX 1060 3GB
cd "C:\projetos"  # Ou onde preferir
git clone https://github.com/frankduque/tradesigns-poc-live.git
cd tradesigns-poc-live
```

---

### **3️⃣ Setup ambiente Python (PC com GPU)**

```bash
# Criar ambiente virtual
python -m venv venv
.\venv\Scripts\activate

# Instalar dependências básicas
pip install -r requirements.txt

# Instalar LightGBM com GPU
pip install lightgbm --upgrade
```

---

### **4️⃣ Configurar GPU (PC com GPU)**

Seguir: `docs/GPU_SETUP_GUIDE.md`

**Resumo:**
1. Verificar driver NVIDIA: `nvidia-smi`
2. Instalar CUDA 12.x (se necessário)
3. Testar GPU: `python test_gpu.py`

---

### **5️⃣ Transferir dados (crítico!)**

**IMPORTANTE**: Dados não vão pro GitHub (são muito grandes)

**Opção A: Pen Drive / HD Externo** (recomendado)
```bash
# PC atual - copiar para pen drive
xcopy /E /I "data\" "E:\tradesigns-data\data\"

# PC com GPU - colar
xcopy /E /I "E:\tradesigns-data\data\" "C:\projetos\tradesigns-poc-live\data\"
```

**Opção B: Rede local / Google Drive**
```bash
# Compactar dados
tar -czf tradesigns_data.tar.gz data/

# Transferir por rede local ou upload
# Descompactar no PC com GPU
tar -xzf tradesigns_data.tar.gz
```

**Opção C: Re-processar dados** (mais demorado)
```bash
# No PC com GPU, reprocessar do zero
python scripts/import_histdata.py
python scripts/process_data.py
python scripts/prepare_regression_dataset.py
```

---

### **6️⃣ Treinar com GPU!** 🚀

```bash
# No PC com GPU
python scripts/train_model_lightgbm_gpu.py
```

**Tempo esperado**: 5-8 minutos ⚡

---

## 📁 ARQUIVOS QUE VÃO PRO GITHUB

### ✅ **Incluir no Git:**
```
src/                           # Código-fonte
scripts/                       # Scripts de treino
docs/                          # Documentação
requirements.txt               # Dependências
README.md                      # Instruções
.gitignore                     # Ignorar arquivos grandes
```

### ❌ **NÃO incluir (muito grandes):**
```
data/                          # Datasets (~1-2GB)
models/                        # Modelos treinados (~500MB)
logs/                          # Logs
venv/                          # Ambiente virtual
__pycache__/                   # Cache Python
*.parquet                      # Arquivos de dados
*.joblib                       # Modelos salvos
```

---

## 📄 CRIAR .gitignore

Criar arquivo `.gitignore` na raiz:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data files (muito grandes)
data/raw/
data/processed/
data/features/
data/results/
*.parquet
*.csv
*.zip

# Models (muito grandes)
models/
*.joblib
*.pkl
*.h5
*.pth

# Logs
logs/
*.log

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Temp
tmp/
temp/
```

---

## 📦 CRIAR requirements.txt

```bash
# No PC atual, gerar requirements
pip freeze > requirements.txt
```

Ou criar manualmente:

```txt
# requirements.txt
pandas==2.2.0
numpy==1.26.3
scikit-learn==1.4.0
lightgbm==4.3.0
joblib==1.3.2
tqdm==4.66.1
pyarrow==15.0.0
```

---

## ✅ CHECKLIST DE MIGRAÇÃO

```
[ ] 1. Parar treino atual (Ctrl+C)
[ ] 2. Criar .gitignore
[ ] 3. Criar requirements.txt
[ ] 4. Git init + commit
[ ] 5. Criar repo no GitHub
[ ] 6. Push para GitHub
[ ] 7. Clonar no PC com GPU
[ ] 8. Setup ambiente Python
[ ] 9. Transferir dados (pen drive/rede)
[ ] 10. Configurar GPU (CUDA + drivers)
[ ] 11. Testar GPU (test_gpu.py)
[ ] 12. Treinar com GPU! 🚀
```

---

## 🎯 RESULTADO ESPERADO

**PC atual (CPU):**
- Random Forest: 4+ horas ⏳

**PC com GPU (GTX 1060 3GB):**
- LightGBM GPU: 5-8 minutos ⚡
- **Speedup: ~30-50x mais rápido!**

---

## 💡 DICAS

1. **Dados são o mais importante**: Priorize transferir `data/features/ml_dataset_regression.parquet`
2. **Teste pequeno primeiro**: Use subset dos dados para testar setup
3. **Monitore GPU**: Use `nvidia-smi -l 1` durante treino
4. **Salve modelos**: Depois do treino, faça backup dos modelos
5. **Documente mudanças**: Anote qualquer ajuste necessário

---

## 🆘 SE ALGO DER ERRADO

### GPU não detectada
```bash
nvidia-smi  # Deve mostrar GTX 1060
nvcc --version  # Deve mostrar CUDA version
```

### LightGBM não usa GPU
```bash
pip uninstall lightgbm
pip install lightgbm --upgrade --no-cache-dir
```

### Dados corrompidos
```bash
# Re-gerar dataset
python scripts/prepare_regression_dataset.py
```

---

**Boa sorte com a migração! 🚀**

Depois me conta quanto tempo levou o treino na GPU! Aposto que vai ser <10 minutos! 🎮
