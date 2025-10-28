"""
Script Completo de Preparao de Dados
Importa  Features  Labels  Salva
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import logging
import pandas as pd
from datetime import datetime

from src.ml.data_loader import HistDataLoader
from src.ml.feature_engineer import FeatureEngineer
from src.ml.label_creator import LabelCreator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/prepare_dataset.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Pipeline completo de preparacao"""
    
    logger.info("=" * 70)
    logger.info("PREPARACAO DE DATASET ML")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    # 1. Carregar dados brutos
    logger.info("\nPASSO 1: Carregando dados historicos...")
    loader = HistDataLoader()
    df = loader.load_processed()
    
    if df is None:
        logger.error("Dados nao encontrados!")
        logger.error("   Execute: python scripts/import_historical.py")
        return False
    
    logger.info(f"OK - {len(df):,} candles carregados")
    
    # 2. Criar features
    logger.info("\nPASSO 2: Criando features tecnicas...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    logger.info(f"OK - {len(engineer.get_feature_names())} features criadas")
    
    # 3. Criar labels
    logger.info("\nPASSO 3: Criando labels (simulando trades)...")
    label_creator = LabelCreator(
        take_profit_pct=0.0008,   # 8 pips (0.08%) - scalping agressivo
        stop_loss_pct=0.0008,     # 8 pips (0.08%) - risk/reward 1:1
        max_duration_candles=15,  # 15 minutos - trades rapidos
        fee_pct=0.0001            # 1 pip fee (0.01%)
    )
    
    df_labeled = label_creator.create_realistic_labels(df_features)
    
    # 4. Salvar dataset final
    logger.info("\nPASSO 4: Salvando dataset completo...")
    
    output_file = Path("data/features/ml_dataset_full.parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df_labeled.to_parquet(output_file, compression='gzip', index=False)
    
    file_size = output_file.stat().st_size / (1024 * 1024)
    
    # Resumo final
    elapsed = datetime.now() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("DATASET ML PRONTO!")
    logger.info("=" * 70)
    logger.info(f"Arquivo: {output_file}")
    logger.info(f"Tamanho: {file_size:.2f} MB")
    logger.info(f"Samples: {len(df_labeled):,}")
    logger.info(f"Features: {len(engineer.get_feature_names())}")
    logger.info(f"Tempo: {elapsed.total_seconds():.1f}s")
    
    # Estatisticas
    logger.info(f"\nDistribuicao de Labels:")
    label_counts = df_labeled['label'].value_counts()
    total_valid = label_counts.sum()
    logger.info(f"   WIN (1):  {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total_valid*100:.1f}%)")
    logger.info(f"   LOSS (-1): {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/total_valid*100:.1f}%)")
    logger.info(f"   HOLD (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total_valid*100:.1f}%)")
    
    logger.info(f"\nPeriodo dos dados:")
    logger.info(f"   Inicio: {df_labeled['timestamp'].min()}")
    logger.info(f"   Fim: {df_labeled['timestamp'].max()}")
    
    logger.info(f"\nProximo passo:")
    logger.info(f"   python scripts/train_model.py")
    
    return True


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)
