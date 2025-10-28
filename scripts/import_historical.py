"""
Script de Importao de Dados Histricos
Processa ZIPs do HistData e salva em formato otimizado
"""
import sys
from pathlib import Path

# Adicionar raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import logging
from src.ml.data_loader import HistDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/import_historical.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Importa dados histricos do HistData"""
    
    logger.info("=" * 70)
    logger.info(" IMPORTAO DE DADOS HISTRICOS")
    logger.info("=" * 70)
    
    # Criar loader
    loader = HistDataLoader()
    
    # Verificar se j existe processado
    existing = loader.load_processed()
    
    if existing is not None:
        logger.warning(" Dados j processados encontrados!")
        logger.info(f"   {len(existing):,} candles carregados")
        logger.info(f"   Perodo: {existing['timestamp'].min()} at {existing['timestamp'].max()}")
        
        response = input("\n Reprocessar dados? (y/N): ")
        if response.lower() != 'y':
            logger.info(" Usando dados existentes")
            return existing
    
    try:
        # Processar ZIPs
        logger.info("\n Processando arquivos ZIP...")
        df = loader.load_all_data(pair="EURUSD")
        
        # Validar
        logger.info("\n Validando dados...")
        issues = loader.validate_data(df)
        
        if issues['invalid_ohlc'] > 0 or issues['missing_values']:
            logger.warning(" Problemas encontrados nos dados!")
            response = input(" Continuar mesmo assim? (y/N): ")
            if response.lower() != 'y':
                logger.error(" Importao cancelada")
                return None
        
        # Salvar
        logger.info("\n Salvando dados processados...")
        output_path = loader.save_processed(df)
        
        logger.info("\n" + "=" * 70)
        logger.info(" IMPORTAO CONCLUDA!")
        logger.info("=" * 70)
        logger.info(f" Total de candles: {len(df):,}")
        logger.info(f" Perodo: {df['timestamp'].min()} at {df['timestamp'].max()}")
        logger.info(f" Arquivo salvo: {output_path}")
        logger.info(f" Tamanho: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Estatsticas
        logger.info(f"\n Estatsticas:")
        logger.info(f"   Preo mdio: {df['close'].mean():.5f}")
        logger.info(f"   Preo mn: {df['close'].min():.5f}")
        logger.info(f"   Preo mx: {df['close'].max():.5f}")
        logger.info(f"   Volatilidade: {df['close'].pct_change().std()*100:.3f}%")
        
        logger.info(f"\n Prximos passos:")
        logger.info(f"   1. python scripts/create_features.py")
        logger.info(f"   2. python scripts/train_model.py")
        
        return df
        
    except Exception as e:
        logger.error(f" Erro durante importao: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Criar pasta de logs
    Path("logs").mkdir(exist_ok=True)
    
    # Executar
    result = main()
    
    if result is not None:
        sys.exit(0)
    else:
        sys.exit(1)
