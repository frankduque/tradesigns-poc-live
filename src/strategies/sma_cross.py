"""
Estratgia: SMA Crossover
Sinal de compra quando SMA rpida cruza acima da SMA lenta
Sinal de venda quando SMA rpida cruza abaixo da SMA lenta
"""
import pandas as pd
import logging
from src.strategies.base import BaseStrategy
from src.indicators.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class SMACrossStrategy(BaseStrategy):
    """Estratgia de cruzamento de mdias mveis simples"""
    
    name = "SMA_Cross"
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        """
        Args:
            fast_period: Perodo da SMA rpida (default: 10)
            slow_period: Perodo da SMA lenta (default: 30)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"SMA_Cross_{fast_period}_{slow_period}"
        
        logger.info(f"Estratgia inicializada: {self.name}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera sinais baseados no cruzamento das SMAs
        
        Returns:
            DataFrame com coluna 'signal' (1=BUY, -1=SELL, 0=NONE)
        """
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Verificar se indicadores existem
        fast_col = f'sma_{self.fast_period}'
        slow_col = f'sma_{self.slow_period}'
        
        if fast_col not in df.columns or slow_col not in df.columns:
            logger.warning(f"SMAs no encontradas no DataFrame. Calculando...")
            indicators = TechnicalIndicators(df)
            df = indicators.calculate_sma_only([self.fast_period, self.slow_period])
        
        sma_fast = df[fast_col]
        sma_slow = df[slow_col]
        
        # Detectar crossovers
        for i in range(1, len(df)):
            # BUY: SMA rpida cruza acima da lenta
            if TechnicalIndicators.has_crossover(sma_fast, sma_slow, i):
                signals.iloc[i, 0] = 1  # signal = 1
                logger.debug(f" BUY signal @ {df.index[i]}")
            
            # SELL: SMA rpida cruza abaixo da lenta
            elif TechnicalIndicators.has_crossunder(sma_fast, sma_slow, i):
                signals.iloc[i, 0] = -1  # signal = -1
                logger.debug(f" SELL signal @ {df.index[i]}")
        
        return signals
    
    def calculate_score(self, df: pd.DataFrame, idx: int) -> float:
        """
        Calcula score de qualidade do sinal
        
        Fatores considerados:
        - Distncia entre as SMAs (fora da tendncia)
        - Volume (se disponvel)
        - Posio do RSI
        """
        score = 0.5  # Base score
        
        fast_col = f'sma_{self.fast_period}'
        slow_col = f'sma_{self.slow_period}'
        
        try:
            sma_fast = df[fast_col].iloc[idx]
            sma_slow = df[slow_col].iloc[idx]
            
            # 1. Distncia entre SMAs (max +0.3)
            if 'atr' in df.columns:
                atr = df['atr'].iloc[idx]
                if atr > 0:
                    distance = abs(sma_fast - sma_slow)
                    normalized_distance = min(distance / atr, 2) / 2
                    score += 0.3 * normalized_distance
            else:
                # Sem ATR, usar distncia percentual
                distance_pct = abs(sma_fast - sma_slow) / sma_slow
                score += min(0.3, distance_pct * 100)
            
            # 2. RSI no extremo (max +0.2)
            if 'rsi_14' in df.columns:
                rsi = df['rsi_14'].iloc[idx]
                # RSI entre 40-60  ideal (mercado no extremo)
                if 40 <= rsi <= 60:
                    score += 0.2
                elif 30 <= rsi <= 70:
                    score += 0.1
        
        except (KeyError, IndexError) as e:
            logger.warning(f"Erro ao calcular score: {e}")
        
        return min(max(score, 0.0), 1.0)  # Limitar entre 0 e 1
