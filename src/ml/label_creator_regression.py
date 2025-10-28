"""
Multi-Target Regression Label Creator

Cria labels realistas baseadas em variação real do preço,
não em regras arbitrárias de TP/SL.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RegressionLabelCreator:
    """
    Cria labels de regressão para ML:
    - return_5m, return_10m, return_30m (variação % do preço)
    - max_return_Xm (upside potential)
    - min_return_Xm (downside risk)
    """
    
    def __init__(self, horizons: list = [5, 10, 30]):
        """
        Args:
            horizons: Lista de horizontes em minutos [5, 10, 30]
        """
        self.horizons = horizons
        logger.info("Regression Label Creator inicializado")
        logger.info(f"   Horizontes: {horizons} minutos")
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria labels de regressão = variação % do preço em N minutos
        
        Args:
            df: DataFrame com OHLCV
        
        Returns:
            DataFrame com labels: return_Xm, max_return_Xm, min_return_Xm
        """
        logger.info(" Criando labels de regressão...")
        
        df = df.copy()
        
        for h in self.horizons:
            logger.info(f"   Horizonte {h}m...")
            
            # 1. RETURN: Variação percentual do close
            # Quanto o preço variou em h minutos?
            df[f'return_{h}m'] = (
                (df['close'].shift(-h) - df['close']) / df['close'] * 100
            )
            
            # 2. MAX RETURN: Máximo upside possível em h minutos
            # Qual foi o maior ganho possível nesse período?
            future_highs = df['high'].rolling(h).max().shift(-h)
            df[f'max_return_{h}m'] = (
                (future_highs - df['close']) / df['close'] * 100
            )
            
            # 3. MIN RETURN: Máximo downside (risco) em h minutos
            # Qual foi a maior queda possível nesse período?
            future_lows = df['low'].rolling(h).min().shift(-h)
            df[f'min_return_{h}m'] = (
                (future_lows - df['close']) / df['close'] * 100
            )
        
        # Remove linhas sem label (final do dataset)
        max_horizon = max(self.horizons)
        n_before = len(df)
        df = df.iloc[:-max_horizon]
        n_after = len(df)
        
        logger.info(f" Labels criados!")
        logger.info(f"   Removidas {n_before - n_after:,} linhas (sem futuro)")
        logger.info(f"   Dataset final: {n_after:,} samples")
        
        # Estatísticas dos labels
        self._log_label_statistics(df)
        
        return df
    
    def _log_label_statistics(self, df: pd.DataFrame):
        """Log estatísticas das labels criadas"""
        logger.info("\nEstatisticas dos Labels:")
        
        for h in self.horizons:
            return_col = f'return_{h}m'
            max_col = f'max_return_{h}m'
            min_col = f'min_return_{h}m'
            
            if return_col in df.columns:
                returns = df[return_col].dropna()
                max_returns = df[max_col].dropna()
                min_returns = df[min_col].dropna()
                
                logger.info(f"\n{h} minutos:")
                logger.info(f"   Return mean: {returns.mean():.4f}%")
                logger.info(f"   Return std:  {returns.std():.4f}%")
                logger.info(f"   Return min:  {returns.min():.4f}%")
                logger.info(f"   Return max:  {returns.max():.4f}%")
                logger.info(f"   Upside avg:  {max_returns.mean():.4f}%")
                logger.info(f"   Downside avg: {min_returns.mean():.4f}%")
                
                # Contagem de movimentos significativos
                n_up = (returns > 0.05).sum()
                n_down = (returns < -0.05).sum()
                n_flat = ((returns >= -0.05) & (returns <= 0.05)).sum()
                total = len(returns)
                
                logger.info(f"   UP (>0.05%):   {n_up:,} ({n_up/total*100:.1f}%)")
                logger.info(f"   DOWN (<-0.05%): {n_down:,} ({n_down/total*100:.1f}%)")
                logger.info(f"   FLAT:          {n_flat:,} ({n_flat/total*100:.1f}%)")
    
    def get_label_columns(self) -> list:
        """Retorna lista de colunas de labels criadas"""
        cols = []
        for h in self.horizons:
            cols.extend([
                f'return_{h}m',
                f'max_return_{h}m',
                f'min_return_{h}m'
            ])
        return cols
    
    def save_labels(self, df: pd.DataFrame, output_path: Path):
        """Salva dataset com labels"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, compression='snappy', index=True)
        
        file_size = output_path.stat().st_size / (1024 ** 2)
        logger.info(f"\nDataset salvo:")
        logger.info(f"   Path: {output_path}")
        logger.info(f"   Size: {file_size:.2f} MB")
        logger.info(f"   Rows: {len(df):,}")
        logger.info(f"   Columns: {len(df.columns)}")


if __name__ == "__main__":
    # Teste básico
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.ml.data_loader import DataLoader
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("TESTE: Regression Label Creator")
    print("="*70)
    
    # Carregar dados processados
    loader = DataLoader()
    df = loader.load_processed()
    
    print(f"\nDados carregados: {len(df):,} candles")
    
    # Criar labels
    creator = RegressionLabelCreator(horizons=[5, 10, 30])
    df_with_labels = creator.create_labels(df)
    
    print(f"\nDataset com labels: {len(df_with_labels):,} samples")
    print(f"Colunas de labels: {creator.get_label_columns()}")
    
    # Salvar
    output_path = Path("data/features/ml_dataset_regression.parquet")
    creator.save_labels(df_with_labels, output_path)
    
    print("\nTESTE CONCLUÍDO!")
