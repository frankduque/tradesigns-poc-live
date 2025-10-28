"""
Script: Treinar Modelo Multi-Target Regression

Treina modelo para prever variação de preço em múltiplos horizontes.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
import pandas as pd
import numpy as np
from src.ml.trainer_regression import MultiTargetRegressionTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_regression_model.log'),
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
    logger.info(f"\nCarregando dataset: {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    logger.info(f" {len(df):,} samples carregados")
    
    # Targets de regressão
    target_cols = ['return_5m', 'return_10m', 'return_30m']
    
    # Features = todas exceto targets e OHLCV básicas
    exclude_cols = target_cols + [
        'timestamp',  # timestamp não é feature numérica
        'open', 'high', 'low', 'close', 'volume',
        'max_return_5m', 'max_return_10m', 'max_return_30m',
        'min_return_5m', 'min_return_10m', 'min_return_30m'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remover features com muitos NaN (>50%)
    nan_pct = df[feature_cols].isnull().mean()
    bad_features = nan_pct[nan_pct > 0.5].index.tolist()
    
    if bad_features:
        logger.warning(f"   Removendo {len(bad_features)} features com >50% NaN: {bad_features}")
        feature_cols = [col for col in feature_cols if col not in bad_features]
    
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Targets: {len(target_cols)}")
    
    # Remover linhas com NaN remanescentes
    df_clean = df[feature_cols + target_cols].dropna()
    n_removed = len(df) - len(df_clean)
    
    if n_removed > 0:
        logger.warning(f"   Removidas {n_removed:,} linhas com NaN ({n_removed/len(df)*100:.1f}%)")
    
    # Remover infinitos e valores extremos
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    n_removed_inf = len(df) - n_removed - len(df_clean)
    
    if n_removed_inf > 0:
        logger.warning(f"   Removidas {n_removed_inf:,} linhas com infinitos ({n_removed_inf/len(df)*100:.1f}%)")
    
    X = df_clean[feature_cols].values
    y = df_clean[target_cols].values
    
    logger.info(f"\nDataset final:")
    logger.info(f"   X shape: {X.shape}")
    logger.info(f"   y shape: {y.shape}")
    
    return X, y, feature_cols, target_cols


def main():
    """Treinar modelo de regressão"""
    try:
        logger.info("="*70)
        logger.info("TREINAMENTO - MULTI-TARGET REGRESSION")
        logger.info("="*70)
        
        # ===== PASSO 1: Carregar dataset =====
        dataset_path = Path("data/features/ml_dataset_regression.parquet")
        
        if not dataset_path.exists():
            logger.error(f"Dataset não encontrado: {dataset_path}")
            logger.error("Execute primeiro: python scripts/prepare_regression_dataset.py")
            return False
        
        X, y, feature_names, target_names = load_dataset(dataset_path)
        
        # ===== PASSO 2: Treinar modelo =====
        logger.info("\nPASSO 2: Treinando modelo...")
        
        trainer = MultiTargetRegressionTrainer(
            n_estimators=200,
            max_depth=15,
            min_samples_split=100,
            min_samples_leaf=50,
            random_state=42
        )
        
        metrics = trainer.train(
            X=X,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            test_size=0.2,
            validation=True
        )
        
        # ===== PASSO 3: Salvar modelo =====
        logger.info("\nPASSO 3: Salvando modelo...")
        model_path = Path("models/regression_model.joblib")
        trainer.save(model_path)
        
        # ===== RESUMO FINAL =====
        logger.info("\n" + "="*70)
        logger.info("MODELO TREINADO COM SUCESSO!")
        logger.info("="*70)
        logger.info(f"Modelo: {model_path}")
        logger.info(f"Tempo de treinamento: {metrics['train_time']:.1f}s")
        
        logger.info("\nPerformance (Test Set):")
        for target in target_names:
            test_metrics = metrics['test'][target]
            logger.info(f"\n{target}:")
            logger.info(f"   RMSE: {test_metrics['rmse']:.4f}%")
            logger.info(f"   MAE:  {test_metrics['mae']:.4f}%")
            logger.info(f"   R2:   {test_metrics['r2']:.3f}")
        
        logger.info("\nCross-Validation (3-fold):")
        for target in target_names:
            cv = metrics['cv_scores'][target]
            logger.info(f"   {target}: RMSE = {cv['rmse_mean']:.4f} (+/- {cv['rmse_std']:.4f})")
        
        # Análise de feature importance
        logger.info("\nTop 15 Features Mais Importantes:")
        importance_df = metrics['feature_importance'].head(15)
        for idx, row in importance_df.iterrows():
            logger.info(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        logger.info("\nProximo passo:")
        logger.info("   python scripts/generate_signals_regression.py")
        
        return True
    
    except Exception as e:
        logger.error(f"ERRO no treinamento: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
