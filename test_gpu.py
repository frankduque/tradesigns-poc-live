"""
Script de teste rápido para verificar se GPU está funcionando
"""
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_regression
import time

print("="*70)
print("🔍 TESTE DE GPU - TradeSigns POC")
print("="*70)

# Verificar versão
print(f"\n📦 LightGBM version: {lgb.__version__}")

# Dataset de teste
print("\n📊 Criando dataset de teste (100k samples, 50 features)...")
X, y = make_regression(n_samples=100000, n_features=50, random_state=42)

# Testar CPU
print("\n⏱️  TESTE 1: Treinando com CPU...")
start = time.time()
try:
    model_cpu = lgb.LGBMRegressor(
        device='cpu',
        n_estimators=100,
        verbose=-1
    )
    model_cpu.fit(X, y)
    cpu_time = time.time() - start
    print(f"✓ CPU: {cpu_time:.2f}s")
except Exception as e:
    print(f"❌ Erro CPU: {e}")
    cpu_time = None

# Testar GPU
print("\n⏱️  TESTE 2: Treinando com GPU...")
start = time.time()
try:
    model_gpu = lgb.LGBMRegressor(
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        n_estimators=100,
        verbose=-1
    )
    model_gpu.fit(X, y)
    gpu_time = time.time() - start
    print(f"✓ GPU: {gpu_time:.2f}s")
except Exception as e:
    print(f"\n❌ Erro ao usar GPU: {e}")
    print("\n🔍 Possíveis causas:")
    print("   1. CUDA não instalado corretamente")
    print("   2. LightGBM compilado sem suporte GPU")
    print("   3. Driver NVIDIA desatualizado")
    print("\n📖 Consulte: docs/GPU_SETUP_GUIDE.md")
    gpu_time = None

# Resultado
print("\n" + "="*70)
print("📊 RESULTADOS")
print("="*70)

if cpu_time and gpu_time:
    speedup = cpu_time / gpu_time
    print(f"\n⚡ CPU:     {cpu_time:.2f}s")
    print(f"⚡ GPU:     {gpu_time:.2f}s")
    print(f"🚀 Speedup: {speedup:.2f}x mais rápido!")
    
    if speedup > 2.0:
        print("\n✅ GPU está funcionando PERFEITAMENTE!")
        print("   Pronto para treinar o modelo completo!")
    elif speedup > 1.2:
        print("\n✅ GPU funcionando, mas com speedup modesto")
        print("   (Normal para datasets pequenos)")
    else:
        print("\n⚠️  GPU não está acelerando como esperado")
        print("   Verifique configurações CUDA")
elif cpu_time:
    print(f"\n⚡ CPU: {cpu_time:.2f}s")
    print("\n⚠️  GPU não disponível - usando apenas CPU")
else:
    print("\n❌ Ambos falharam - verifique instalação")

print("\n" + "="*70)

# Verificar NVIDIA
print("\n🔍 Verificando NVIDIA GPU...")
try:
    import subprocess
    nvidia_smi = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
        stderr=subprocess.DEVNULL
    )
    gpu_info = nvidia_smi.decode('utf-8').strip()
    print(f"✓ GPU detectada: {gpu_info}")
except Exception as e:
    print(f"⚠️  nvidia-smi não disponível")
    print(f"   Certifique-se de que os drivers NVIDIA estão instalados")

print("\n✅ Teste concluído!")
print("="*70)
