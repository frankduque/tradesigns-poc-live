"""
Classe base para estratgias
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class BaseStrategy(ABC):
    """Classe abstrata para estratgias de trading"""
    
    name: str = "BaseStrategy"
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera sinais de trading baseado nos dados
        
        Args:
            df: DataFrame com preos e indicadores
        
        Returns:
            DataFrame com coluna 'signal':
                1 = BUY
               -1 = SELL
                0 = No signal
        """
        pass
    
    def calculate_score(self, df: pd.DataFrame, idx: int) -> float:
        """
        Calcula score de qualidade do sinal (0.0 a 1.0)
        
        Args:
            df: DataFrame com preos e indicadores
            idx: ndice do sinal
        
        Returns:
            Score entre 0.0 e 1.0
        """
        # Implementao padro - pode ser sobrescrita
        return 0.5
    
    def validate_signal(self, df: pd.DataFrame, idx: int, signal_type: str) -> bool:
        """
        Valida se o sinal  vlido
        
        Args:
            df: DataFrame com preos e indicadores
            idx: ndice do sinal
            signal_type: 'BUY' ou 'SELL'
        
        Returns:
            True se vlido
        """
        # Implementao padro - pode ser sobrescrita
        return True
