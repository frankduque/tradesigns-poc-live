"""
Script: Preparar Dataset para Multi-Target Regression

Cria labels realistas baseadas em variação real do preço.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
from src.ml.data_loader import DataLoader
from src.ml.feature_engineer import FeatureEngineer
from src.ml.label_creator_regression import RegressionLabelCreator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prepare_regression_dataset.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Preparar dataset de regressão"""
    try:
        logger.info("="*70)
        logger.info("PREPARACAO DE DATASET - MULTI-TARGET REGRESSION")
        logger.info("="*70)
        
        # ===== PASSO 1: Carregar dados =====
        logger.info("\nPASSO 1: Carregando dados historicos...")
        loader = DataLoader()
        df = loader.load_processed()
        
        if df is None or df.empty:
            logger.error("ERRO: Dados não carregados!")
            return False
        
        logger.info(f" {len(df):,} candles carregados")
        logger.info(f"   Periodo: {df.index[0]} ate {df.index[-1]}")
        
        # ===== PASSO 2: Criar features técnicas =====
        logger.info("\nPASSO 2: Criando features tecnicas...")
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        if df_features is None or df_features.empty:
            logger.error("ERRO: Features não criadas!")
            return False
        
        logger.info(f" {len(df_features.columns)} features criadas")
        
        # ===== PASSO 3: Criar labels de regressão =====
        logger.info("\nPASSO 3: Criando labels de regressao...")
        label_creator = RegressionLabelCreator(horizons=[5, 10, 30])
        df_with_labels = label_creator.create_labels(df_features)
        
        if df_with_labels is None or df_with_labels.empty:
            logger.error("ERRO: Labels não criadas!")
            return False
        
        # ===== PASSO 4: Salvar dataset =====
        logger.info("\nPASSO 4: Salvando dataset final...")
        output_path = Path("data/features/ml_dataset_regression.parquet")
        label_creator.save_labels(df_with_labels, output_path)
        
        # ===== RESUMO FINAL =====
        logger.info("\n" + "="*70)
        logger.info("DATASET DE REGRESSAO PRONTO!")
        logger.info("="*70)
        logger.info(f"Arquivo: {output_path}")
        logger.info(f"Tamanho: {output_path.stat().st_size / (1024**2):.2f} MB")
        logger.info(f"Samples: {len(df_with_labels):,}")
        logger.info(f"Features: {len(df_features.columns)}")
        logger.info(f"Targets: {len(label_creator.get_label_columns())}")
        
        # Distribuição dos labels
        logger.info("\nDistribuicao dos Targets:")
        for col in ['return_5m', 'return_10m', 'return_30m']:
            if col in df_with_labels.columns:
                data = df_with_labels[col].dropna()
                logger.info(f"\n{col}:")
                logger.info(f"   Mean:   {data.mean():.4f}%")
                logger.info(f"   Std:    {data.std():.4f}%")
                logger.info(f"   Min:    {data.min():.4f}%")
                logger.info(f"   Max:    {data.max():.4f}%")
                logger.info(f"   Median: {data.median():.4f}%")
        
        logger.info("\nProximo passo:")
        logger.info("   python scripts/train_model_regression.py")
        
        return True
    
    except Exception as e:
        logger.error(f"ERRO no pipeline: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
