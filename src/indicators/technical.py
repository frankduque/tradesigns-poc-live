"""
Indicadores Tcnicos usando pandas-ta
"""
import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calcula indicadores tcnicos"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame com colunas ['open', 'high', 'low', 'close']
                e timestamp como index
        """
        self.df = df.copy()
        
        # Garantir nomes em minsculo
        self.df.columns = [col.lower() for col in self.df.columns]
    
    def calculate_all(self) -> pd.DataFrame:
        """Calcula todos os indicadores de uma vez"""
        df = self.df.copy()
        
        try:
            # Trend Indicators
            df['sma_10'] = ta.sma(df['close'], length=10)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_30'] = ta.sma(df['close'], length=30)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['ema_12'] = ta.ema(df['close'], length=12)
            df['ema_26'] = ta.ema(df['close'], length=26)
            
            # Momentum Indicators
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            
            # Volatility Indicators
            bbands = ta.bbands(df['close'], length=20, std=2)
            if bbands is not None:
                df['bb_upper'] = bbands['BBU_20_2.0']
                df['bb_middle'] = bbands['BBM_20_2.0']
                df['bb_lower'] = bbands['BBL_20_2.0']
            
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd is not None:
                df['macd'] = macd['MACD_12_26_9']
                df['macd_signal'] = macd['MACDs_12_26_9']
                df['macd_hist'] = macd['MACDh_12_26_9']
            
            logger.debug(f" Indicadores calculados: {len(df)} candles")
            
        except Exception as e:
            logger.error(f" Erro ao calcular indicadores: {e}")
        
        return df
    
    def calculate_sma_only(self, periods: list = [10, 20, 30, 50]) -> pd.DataFrame:
        """Calcula apenas SMAs (mais rpido para estratgia simples)"""
        df = self.df.copy()
        
        for period in periods:
            df[f'sma_{period}'] = ta.sma(df['close'], length=period)
        
        return df
    
    @staticmethod
    def has_crossover(fast: pd.Series, slow: pd.Series, index: int) -> bool:
        """
        Detecta crossover (fast cruzou acima do slow)
        
        Returns:
            True se houve crossover no ndice especificado
        """
        if index < 1:
            return False
        
        current_fast = fast.iloc[index]
        current_slow = slow.iloc[index]
        prev_fast = fast.iloc[index - 1]
        prev_slow = slow.iloc[index - 1]
        
        # Crossover: estava abaixo e agora est acima
        return prev_fast <= prev_slow and current_fast > current_slow
    
    @staticmethod
    def has_crossunder(fast: pd.Series, slow: pd.Series, index: int) -> bool:
        """
        Detecta crossunder (fast cruzou abaixo do slow)
        
        Returns:
            True se houve crossunder no ndice especificado
        """
        if index < 1:
            return False
        
        current_fast = fast.iloc[index]
        current_slow = slow.iloc[index]
        prev_fast = fast.iloc[index - 1]
        prev_slow = slow.iloc[index - 1]
        
        # Crossunder: estava acima e agora est abaixo
        return prev_fast >= prev_slow and current_fast < current_slow
