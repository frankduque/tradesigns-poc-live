"""
Feature Engineering - Cria features para ML
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import List

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Cria features tcnicas para machine learning"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria todas as features de uma vez
        
        Args:
            df: DataFrame com [timestamp, open, high, low, close, volume]
        
        Returns:
            DataFrame com 50+ features adicionadas
        """
        logger.info(" Criando features...")
        
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. Features de Tendncia
        logger.info("    Trend features...")
        df = self._add_trend_features(df)
        
        # 2. Features de Momentum
        logger.info("    Momentum features...")
        df = self._add_momentum_features(df)
        
        # 3. Features de Volatilidade
        logger.info("    Volatility features...")
        df = self._add_volatility_features(df)
        
        # 4. Features de Volume
        logger.info("    Volume features...")
        df = self._add_volume_features(df)
        
        # 5. Features de Price Action
        logger.info("    Price action features...")
        df = self._add_price_action_features(df)
        
        # 6. Features Temporais
        logger.info("    Time-based features...")
        df = self._add_time_features(df)
        
        # 7. Features Derivadas
        logger.info("    Derived features...")
        df = self._add_derived_features(df)
        
        # Remover primeiras 200 linhas (warm-up dos indicadores)
        # Não usar dropna() pois remove linhas válidas com NaN esparso
        WARMUP_ROWS = 200
        initial_len = len(df)
        df = df.iloc[WARMUP_ROWS:].reset_index(drop=True)
        
        # Preencher NaNs restantes com forward fill
        df = df.ffill().bfill()
        
        removed = initial_len - len(df)
        logger.info(f"    Removidas {removed} linhas (warm-up)")
        logger.info(f" Total de features: {len(df.columns) - 6}")  # -6 colunas base
        
        self.feature_names = [col for col in df.columns if col not in 
                             ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de tendncia (SMAs, EMAs, etc)"""
        
        # SMAs
        for period in [10, 20, 30, 50, 100, 200]:
            df[f'sma_{period}'] = ta.sma(df['close'], length=period)
        
        # EMAs
        for period in [12, 26, 50]:
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)
        
        # SMA Crosses
        df['sma_cross_10_30'] = (df['sma_10'] > df['sma_30']).astype(int)
        df['sma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # Distncia do preo s SMAs (normalizado pelo ATR)
        df['price_sma_20_dist'] = (df['close'] - df['sma_20']) / df['close']
        df['price_sma_50_dist'] = (df['close'] - df['sma_50']) / df['close']
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de momentum (RSI, Stochastic, etc)"""
        
        # RSI
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['rsi_7'] = ta.rsi(df['close'], length=7)
        
        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # CCI
        df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        
        # Williams %R
        df['willr_14'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # ROC (Rate of Change)
        df['roc_10'] = ta.roc(df['close'], length=10)
        
        # MFI (Money Flow Index)
        df['mfi_14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de volatilidade (ATR, Bollinger, etc)"""
        
        # ATR
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bbands['BBU_20_2.0_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Keltner Channels
        kc = ta.kc(df['high'], df['low'], df['close'], length=20)
        df['kc_upper'] = kc['KCUe_20_2']
        df['kc_middle'] = kc['KCBe_20_2']
        df['kc_lower'] = kc['KCLe_20_2']
        
        # Volatilidade histrica
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        df['volatility_30'] = df['close'].pct_change().rolling(30).std()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de volume"""
        
        # OBV
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # Volume MA
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # VWAP removido - causava problemas com index
        # Substituir por volume weighted price simples
        df['vwap'] = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        return df
    
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de price action (candles)"""
        
        # Tamanho do candle
        df['candle_size'] = df['high'] - df['low']
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_body_pct'] = df['candle_body'] / df['candle_size']
        
        # Sombras
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Tipo de candle
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        # Range (high-low) normalizado
        df['range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Consecutivos
        df['consecutive_ups'] = (df['close'] > df['open']).astype(int).rolling(5).sum()
        df['consecutive_downs'] = (df['close'] < df['open']).astype(int).rolling(5).sum()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features baseadas em tempo"""
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Sesses de trading
        df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        
        # Overlap (Londres + NY = alta liquidez)
        df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features derivadas (ratios, diffs, etc)"""
        
        # Momentum changes
        df['rsi_change'] = df['rsi_14'].diff()
        df['macd_hist_change'] = df['macd_hist'].diff()
        
        # Price momentum
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        
        # Ratios
        df['rsi_atr_ratio'] = df['rsi_14'] / (df['atr_14'] * 1000)  # Normalizado
        df['volume_volatility_ratio'] = df['volume_ratio'] / df['volatility_10']
        
        # Distance from extremes
        df['distance_from_high_20'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['distance_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nomes das features"""
        return self.feature_names


def main():
    """Teste do feature engineer"""
    logging.basicConfig(level=logging.INFO)
    
    from src.ml.data_loader import HistDataLoader
    
    # Carregar dados
    loader = HistDataLoader()
    df = loader.load_processed()
    
    if df is None:
        logger.error("Dados no encontrados. Execute data_loader.py primeiro")
        return
    
    # Criar features (usar s 100k candles para teste)
    logger.info(" Testando com 100k candles...")
    df_sample = df.head(100000)
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_sample)
    
    logger.info(f"\n Features criadas:")
    logger.info(f"   Total: {len(engineer.get_feature_names())}")
    logger.info(f"   Dataset shape: {df_features.shape}")
    logger.info(f"\nPrimeiras features:")
    for feat in engineer.get_feature_names()[:10]:
        logger.info(f"   - {feat}")


if __name__ == "__main__":
    main()
