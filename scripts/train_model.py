"""
Script de Treino de Modelo ML
Treina XGBoost/LightGBM com Walk-Forward Validation
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import logging
import pandas as pd
from datetime import datetime

from src.ml.trainer import MLTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/train_model.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Treina modelo ML"""
    
    logger.info("=" * 70)
    logger.info(" TREINAMENTO DE MODELO ML")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    # 1. Carregar dataset preparado
    logger.info("\n Carregando dataset ML...")
    
    dataset_file = Path("data/features/ml_dataset_full.parquet")
    
    if not dataset_file.exists():
        logger.error(f" Dataset no encontrado: {dataset_file}")
        logger.error("   Execute: python scripts/prepare_dataset.py")
        return False
    
    df = pd.read_parquet(dataset_file)
    logger.info(f" Dataset carregado: {len(df):,} samples")
    logger.info(f"   Perodo: {df['timestamp'].min()} at {df['timestamp'].max()}")
    
    # 2. Escolher modelo
    logger.info("\n Configurando modelo...")
    
    model_type = input("\n Escolha o modelo (xgboost/lightgbm) [xgboost]: ").strip().lower()
    if not model_type:
        model_type = 'xgboost'
    
    if model_type not in ['xgboost', 'lightgbm']:
        logger.error(f" Modelo invlido: {model_type}")
        return False
    
    trainer = MLTrainer(model_type=model_type)
    
    # 3. Preparar dados (Walk-Forward Split)
    logger.info("\n Preparando dados (Walk-Forward)...")
    
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df,
        train_start='2020-01-01',
        train_end='2023-12-31',    # 4 anos de treino
        test_start='2024-01-01',
        test_end='2024-12-31'      # 1 ano de teste
    )
    
    # 4. Treinar modelo
    logger.info("\n Iniciando treinamento...")
    logger.info(f"   Isso pode demorar alguns minutos...")
    
    if model_type == 'xgboost':
        trainer.train_xgboost(X_train, y_train)
    else:
        trainer.train_lightgbm(X_train, y_train)
    
    # 5. Avaliar
    logger.info("\n Avaliando modelo...")
    metrics = trainer.evaluate(X_test, y_test)
    
    # 6. Feature Importance
    logger.info("\n Analisando feature importance...")
    importance_df = trainer.get_feature_importance(top_n=20)
    
    # Salvar importance
    importance_file = Path("models/metadata/feature_importance.csv")
    importance_file.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_file, index=False)
    logger.info(f" Feature importance salvo: {importance_file}")
    
    # 7. Salvar modelo
    logger.info("\n Salvando modelo treinado...")
    model_path = trainer.save_model()
    
    # 8. Salvar metadata
    metadata = {
        'model_type': model_type,
        'training_date': datetime.now().isoformat(),
        'train_period': '2020-01-01 to 2023-12-31',
        'test_period': '2024-01-01 to 2024-12-31',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(trainer.feature_names),
        'metrics': {k: float(v) if not isinstance(v, (list, dict)) else str(v) 
                   for k, v in metrics.items() if k != 'confusion_matrix'}
    }
    
    import json
    metadata_file = Path("models/metadata/last_training.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f" Metadata salvo: {metadata_file}")
    
    # Resumo final
    elapsed = datetime.now() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info(" TREINAMENTO CONCLUDO!")
    logger.info("=" * 70)
    logger.info(f" Modelo: {model_type}")
    logger.info(f" Salvo em: {model_path}")
    logger.info(f" Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f" WIN Precision: {metrics['win_precision']:.4f}")
    logger.info(f" Tempo total: {elapsed.total_seconds()/60:.1f} min")
    
    logger.info(f"\n Prximos passos:")
    logger.info(f"   1. python scripts/run_backtest.py (testar em dados histricos)")
    logger.info(f"   2. Integrar modelo ao sistema live (src/signals/generator.py)")
    
    # Perguntar se quer fazer backtest agora
    response = input("\n Executar backtest agora? (y/N): ")
    if response.lower() == 'y':
        logger.info("\n Iniciando backtest...")
        import subprocess
        subprocess.run([sys.executable, "scripts/run_backtest.py"])
    
    return True


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)
