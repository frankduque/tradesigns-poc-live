"""
Signal Generator - Gera sinais quando novos candles so recebidos
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd

from src.database.repositories import price_repo, signal_repo
from src.indicators.technical import TechnicalIndicators
from src.strategies.sma_cross import SMACrossStrategy

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Gera sinais de trading em tempo real"""
    
    def __init__(self):
        self.strategies = [
            SMACrossStrategy(fast_period=10, slow_period=30),
        ]
        self.lookback_candles = 200  # Janela para calcular indicadores
        self.min_candles_required = 50  # Mnimo necessrio
        
        logger.info(f"Signal Generator inicializado com {len(self.strategies)} estratgias")
    
    async def generate_for_candle(self, candle: dict):
        """
        Gera sinais quando um novo candle  recebido
        
        Args:
            candle: Dict com dados do candle {pair, timestamp, open, high, low, close}
        """
        pair = candle['pair']
        
        try:
            # 1. Buscar histrico recente do banco
            df = price_repo.get_recent_candles(pair, limit=self.lookback_candles)
            
            if df.empty or len(df) < self.min_candles_required:
                logger.warning(
                    f" Dados insuficientes para {pair}: {len(df)} candles "
                    f"(mnimo: {self.min_candles_required})"
                )
                return
            
            # 2. Calcular indicadores
            indicators = TechnicalIndicators(df)
            df = indicators.calculate_all()
            
            # 3. Gerar sinais de cada estratgia
            for strategy in self.strategies:
                signals = strategy.generate_signals(df)
                
                # Verificar sinal no ltimo candle
                last_idx = len(signals) - 1
                last_signal = signals.iloc[last_idx]['signal']
                
                if last_signal != 0:  # BUY (1) ou SELL (-1)
                    signal_type = 'BUY' if last_signal == 1 else 'SELL'
                    
                    # 4. Calcular score de qualidade
                    score = strategy.calculate_score(df, last_idx)
                    
                    # 5. Filtrar sinais fracos
                    if score < 0.5:
                        logger.info(
                            f" Sinal rejeitado: {signal_type} {pair} "
                            f"(score: {score:.2f} < 0.50)"
                        )
                        continue
                    
                    # 6. Salvar sinal no banco
                    signal_data = {
                        'pair': pair,
                        'timestamp': candle['timestamp'],
                        'signal_type': signal_type,
                        'strategy': strategy.name,
                        'entry_price': candle['close'],
                        'score': score,
                        'indicators': self._extract_indicators(df, last_idx)
                    }
                    
                    signal_id = signal_repo.save_signal(signal_data)
                    
                    if signal_id:
                        logger.info(
                            f" SINAL GERADO #{signal_id}: {signal_type} {pair} @ {candle['close']:.5f} "
                            f"(Score: {score:.2f}) [{strategy.name}]"
                        )
        
        except Exception as e:
            logger.error(f" Erro ao gerar sinal para {pair}: {e}", exc_info=True)
    
    def _extract_indicators(self, df: pd.DataFrame, idx: int) -> Dict:
        """Extrai valores dos indicadores para salvar com o sinal"""
        indicators = {}
        
        indicator_cols = ['sma_10', 'sma_30', 'rsi_14', 'atr']
        
        for col in indicator_cols:
            if col in df.columns:
                value = df[col].iloc[idx]
                if pd.notna(value):
                    indicators[col] = float(value)
        
        return indicators


# Instncia global
signal_generator = SignalGenerator()


async def generate_signal_for_candle(candle: dict):
    """Funo helper para ser chamada pelo streamer"""
    await signal_generator.generate_for_candle(candle)
