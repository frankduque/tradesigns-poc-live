"""
Script: Treinar Modelo Multi-Target Regression com LightGBM + GPU

HARDWARE NECESSÁRIO:
- NVIDIA GPU (GTX 1060 3GB ou superior)
- CUDA Toolkit instalado
- LightGBM com suporte GPU

VANTAGENS:
- 6-10x mais rápido que Random Forest CPU
- Melhor performance (geralmente)
- Menos memória VRAM necessária
- Ideal para retreinos frequentes
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import time

try:
    import lightgbm as lgb
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("ERRO: LightGBM não instalado!")
    print("Instale com: pip install lightgbm --install-option=--gpu")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_lightgbm_gpu.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> tuple:
    """
    Carrega dataset e separa features/targets
    
    Returns:
        (X, y, feature_names, target_names)
    """
    logger.info(f"\n📂 Carregando dataset: {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    logger.info(f"✓ {len(df):,} samples carregados")
    
    # Targets de regressão
    target_cols = ['return_5m', 'return_10m', 'return_30m']
    
    # Features = todas exceto targets e OHLCV básicas
    exclude_cols = target_cols + [
        'timestamp',
        'open', 'high', 'low', 'close', 'volume',
        'max_return_5m', 'max_return_10m', 'max_return_30m',
        'min_return_5m', 'min_return_10m', 'min_return_30m'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remover features com muitos NaN (>50%)
    nan_pct = df[feature_cols].isnull().mean()
    bad_features = nan_pct[nan_pct > 0.5].index.tolist()
    
    if bad_features:
        logger.warning(f"⚠️  Removendo {len(bad_features)} features com >50% NaN: {bad_features}")
        feature_cols = [col for col in feature_cols if col not in bad_features]
    
    logger.info(f"✓ Features: {len(feature_cols)}")
    logger.info(f"✓ Targets: {len(target_cols)}")
    
    # Remover linhas com NaN e infinitos
    df_clean = df[feature_cols + target_cols].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    
    n_removed = len(df) - len(df_clean)
    if n_removed > 0:
        logger.warning(f"⚠️  Removidas {n_removed:,} linhas com NaN/infinitos ({n_removed/len(df)*100:.1f}%)")
    
    X = df_clean[feature_cols].values
    y = df_clean[target_cols].values
    
    logger.info(f"\n📊 Dataset final:")
    logger.info(f"   X shape: {X.shape}")
    logger.info(f"   y shape: {y.shape}")
    
    return X, y, feature_cols, target_cols


def train_lightgbm_gpu(X, y, feature_names, target_names):
    """
    Treina modelo LightGBM com GPU acceleration
    """
    logger.info("\n" + "="*70)
    logger.info("🚀 TREINAMENTO - LightGBM GPU")
    logger.info("="*70)
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    logger.info(f"\n📊 Train/Test Split:")
    logger.info(f"   Train: {len(X_train):,} samples")
    logger.info(f"   Test:  {len(X_test):,} samples")
    
    # Parâmetros otimizados para GPU
    params = {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'n_estimators': 200,
        'max_depth': 15,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    logger.info(f"\n⚙️  Parâmetros:")
    logger.info(f"   Device: GPU")
    logger.info(f"   N_estimators: {params['n_estimators']}")
    logger.info(f"   Max_depth: {params['max_depth']}")
    logger.info(f"   Learning_rate: {params['learning_rate']}")
    
    # Treinar um modelo para cada target
    models = {}
    metrics = {}
    
    logger.info(f"\n🔥 Iniciando treinamento GPU...\n")
    start_time = time.time()
    
    for i, target in enumerate(target_names):
        logger.info(f"📈 Target {i+1}/3: {target}")
        
        model = lgb.LGBMRegressor(**params)
        
        # Treinar
        model.fit(
            X_train, y_train[:, i],
            eval_set=[(X_test, y_test[:, i])],
            eval_metric='rmse'
        )
        
        # Prever
        y_pred = model.predict(X_test)
        
        # Métricas
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred))
        mae = mean_absolute_error(y_test[:, i], y_pred)
        r2 = r2_score(y_test[:, i], y_pred)
        
        models[target] = model
        metrics[target] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        logger.info(f"   RMSE: {rmse:.4f}%")
        logger.info(f"   MAE:  {mae:.4f}%")
        logger.info(f"   R²:   {r2:.4f}\n")
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Treinamento concluído em {elapsed/60:.2f} minutos")
    
    return models, metrics, feature_names


def save_models(models, metrics, feature_names):
    """Salva modelos e métricas"""
    logger.info(f"\n💾 Salvando modelos...")
    
    # Criar diretório
    Path("models").mkdir(exist_ok=True)
    
    # Salvar modelos
    for target, model in models.items():
        model_path = Path(f"models/lightgbm_gpu_{target}.txt")
        model.booster_.save_model(str(model_path))
        logger.info(f"   ✓ {model_path}")
    
    # Salvar métricas
    metrics_df = pd.DataFrame(metrics).T
    metrics_path = Path("models/lightgbm_gpu_metrics.csv")
    metrics_df.to_csv(metrics_path)
    logger.info(f"   ✓ {metrics_path}")
    
    # Salvar feature importance
    importance_dict = {}
    for target, model in models.items():
        importance_dict[target] = model.feature_importances_
    
    importance_df = pd.DataFrame(
        importance_dict,
        index=feature_names
    ).sort_values(by='return_10m', ascending=False)
    
    importance_path = Path("models/lightgbm_gpu_feature_importance.csv")
    importance_df.to_csv(importance_path)
    logger.info(f"   ✓ {importance_path}")
    
    logger.info(f"\n📊 Top 10 features mais importantes:")
    for i, (feat, val) in enumerate(importance_df['return_10m'].head(10).items(), 1):
        logger.info(f"   {i:2d}. {feat:30s} {val:8.1f}")


def main():
    """Main training pipeline"""
    try:
        logger.info("="*70)
        logger.info("🎮 LIGHTGBM GPU TRAINING - Multi-Target Regression")
        logger.info("="*70)
        
        # Verificar se GPU está disponível
        logger.info("\n🔍 Verificando GPU...")
        try:
            import subprocess
            nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])
            gpu_info = nvidia_smi.decode('utf-8').strip()
            logger.info(f"✓ GPU detectada: {gpu_info}")
        except Exception as e:
            logger.warning(f"⚠️  Não foi possível detectar GPU via nvidia-smi")
            logger.warning(f"   Continuando mesmo assim... (LightGBM vai tentar usar GPU)")
        
        # Carregar dataset
        dataset_path = Path("data/features/ml_dataset_regression.parquet")
        if not dataset_path.exists():
            logger.error(f"❌ Dataset não encontrado: {dataset_path}")
            logger.error("   Execute primeiro: python scripts/prepare_regression_dataset.py")
            return False
        
        X, y, feature_names, target_names = load_dataset(dataset_path)
        
        # Treinar modelos
        models, metrics, feature_names = train_lightgbm_gpu(X, y, feature_names, target_names)
        
        # Salvar
        save_models(models, metrics, feature_names)
        
        # Resumo final
        logger.info("\n" + "="*70)
        logger.info("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("="*70)
        logger.info("\n📊 Métricas Finais:")
        for target, m in metrics.items():
            logger.info(f"\n{target}:")
            logger.info(f"   RMSE: {m['rmse']:.4f}%")
            logger.info(f"   MAE:  {m['mae']:.4f}%")
            logger.info(f"   R²:   {m['r2']:.4f}")
        
        logger.info("\n🎯 Próximo passo:")
        logger.info("   python scripts/generate_signals_regression.py --model lightgbm")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ ERRO no treinamento: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
