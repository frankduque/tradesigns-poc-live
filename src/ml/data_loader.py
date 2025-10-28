"""
Data Loader - Importa dados histricos do HistData
"""
import pandas as pd
import zipfile
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HistDataLoader:
    """Carrega e processa dados do HistData.com"""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_data(self, pair: str = "EURUSD") -> pd.DataFrame:
        """
        Carrega todos os ZIPs do par especificado
        
        Returns:
            DataFrame com colunas: [timestamp, open, high, low, close, volume]
        """
        logger.info(f" Buscando arquivos {pair} em {self.raw_data_dir}")
        
        # Encontrar todos os ZIPs do par
        zip_files = sorted(self.raw_data_dir.glob(f"*{pair}*.zip"))
        
        if not zip_files:
            raise FileNotFoundError(f"Nenhum arquivo ZIP encontrado para {pair}")
        
        logger.info(f" Encontrados {len(zip_files)} arquivos:")
        for zf in zip_files:
            logger.info(f"   - {zf.name}")
        
        # Processar cada arquivo
        all_data = []
        
        for zip_file in tqdm(zip_files, desc="Processando ZIPs"):
            try:
                df = self._process_zip(zip_file)
                all_data.append(df)
                logger.info(f" {zip_file.name}: {len(df):,} candles")
            except Exception as e:
                logger.error(f" Erro em {zip_file.name}: {e}")
        
        # Concatenar tudo
        logger.info(" Concatenando dados...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Ordenar por timestamp
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Remover duplicatas
        initial_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'])
        removed = initial_len - len(combined_df)
        
        if removed > 0:
            logger.warning(f" Removidas {removed:,} linhas duplicadas")
        
        logger.info(f" Dataset final: {len(combined_df):,} candles")
        logger.info(f"   Perodo: {combined_df['timestamp'].min()} at {combined_df['timestamp'].max()}")
        
        return combined_df
    
    def _process_zip(self, zip_path: Path) -> pd.DataFrame:
        """Processa um arquivo ZIP do HistData"""
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # HistData sempre tem 1 CSV dentro do ZIP
            csv_name = zip_ref.namelist()[0]
            
            with zip_ref.open(csv_name) as csv_file:
                # Formato HistData: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
                df = pd.read_csv(
                    csv_file,
                    sep=';',  # HistData usa ponto-e-vrgula
                    names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                    parse_dates=False,
                    dtype={
                        'open': float,
                        'high': float,
                        'low': float,
                        'close': float,
                        'volume': int
                    }
                )
        
        # Converter datetime (formato: "20200101 000000")
        df['timestamp'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
        df = df.drop('datetime', axis=1)
        
        # Reordenar colunas
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def save_processed(self, df: pd.DataFrame, filename: str = "eurusd_m1_full.parquet"):
        """Salva dados processados em formato Parquet (comprimido e rpido)"""
        output_path = self.processed_dir / filename
        
        logger.info(f" Salvando dados processados em {output_path}")
        df.to_parquet(output_path, compression='gzip', index=False)
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f" Salvo: {file_size:.2f} MB")
        
        return output_path
    
    def load_processed(self, filename: str = "eurusd_m1_full.parquet") -> Optional[pd.DataFrame]:
        """Carrega dados j processados (mais rpido)"""
        file_path = self.processed_dir / filename
        
        if not file_path.exists():
            return None
        
        logger.info(f" Carregando dados processados de {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f" Carregados {len(df):,} candles")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """Valida qualidade dos dados"""
        issues = {
            'missing_values': df.isnull().sum().to_dict(),
            'invalid_ohlc': 0,
            'gaps': 0,
            'duplicates': 0
        }
        
        # Verificar OHLC vlido (High >= Low, etc)
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        issues['invalid_ohlc'] = invalid_ohlc.sum()
        
        # Verificar gaps (mais de 5 minutos entre candles)
        time_diff = df['timestamp'].diff()
        gaps = (time_diff > pd.Timedelta(minutes=5)).sum()
        issues['gaps'] = gaps
        
        # Verificar duplicatas
        issues['duplicates'] = df.duplicated(subset=['timestamp']).sum()
        
        # Log
        logger.info(" Validao de dados:")
        logger.info(f"   Missing values: {sum(issues['missing_values'].values())}")
        logger.info(f"   Invalid OHLC: {issues['invalid_ohlc']}")
        logger.info(f"   Gaps (>5min): {issues['gaps']}")
        logger.info(f"   Duplicatas: {issues['duplicates']}")
        
        return issues


def main():
    """Funo principal para testar o loader"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    loader = HistDataLoader()
    
    # Tentar carregar dados j processados
    df = loader.load_processed()
    
    if df is None:
        # Se no existir, processar ZIPs
        logger.info(" Dados processados no encontrados. Processando ZIPs...")
        df = loader.load_all_data(pair="EURUSD")
        
        # Validar
        loader.validate_data(df)
        
        # Salvar
        loader.save_processed(df)
    
    # Exibir resumo
    logger.info("\n" + "="*60)
    logger.info(" RESUMO DOS DADOS")
    logger.info("="*60)
    logger.info(f"Total de candles: {len(df):,}")
    logger.info(f"Perodo: {df['timestamp'].min()} at {df['timestamp'].max()}")
    logger.info(f"Anos cobertos: {df['timestamp'].dt.year.nunique()}")
    logger.info(f"\nPrimeiras linhas:")
    logger.info(df.head())
    logger.info(f"\nltimas linhas:")
    logger.info(df.tail())
    logger.info(f"\nEstatsticas:")
    logger.info(df.describe())



# Alias para compatibilidade
DataLoader = HistDataLoader


if __name__ == "__main__":
    main()
