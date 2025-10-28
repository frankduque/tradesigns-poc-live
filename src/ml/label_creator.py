"""
Label Creator - Cria labels para machine learning
Classifica se um trade seria profitable ou no
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class LabelCreator:
    """Cria labels realistas para trading"""
    
    def __init__(
        self,
        take_profit_pct: float = 0.004,  # 0.4% = 40 pips
        stop_loss_pct: float = 0.002,    # 0.2% = 20 pips
        max_duration_candles: int = 60,  # 60 minutos = 1 hora
        fee_pct: float = 0.0002          # 0.02% de fee
    ):
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_duration_candles = max_duration_candles
        self.fee_pct = fee_pct
        
        logger.info(f"Label Creator inicializado:")
        logger.info(f"   Take Profit: {take_profit_pct*100:.2f}%")
        logger.info(f"   Stop Loss: {stop_loss_pct*100:.2f}%")
        logger.info(f"   Max Duration: {max_duration_candles} candles")
        logger.info(f"   Fee: {fee_pct*100:.3f}%")
    
    def create_realistic_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria labels SIMPLIFICADOS E RAPIDOS
        
        Ao invés de simular cada trade, usa aproximação:
        - Label = 1 (WIN) se próximo high > entry + TP
        - Label = -1 (LOSS) se próximo low < entry - SL  
        - Label = 0 (HOLD) caso contrário
        
        Returns:
            DataFrame com coluna 'label' adicionada
        """
        logger.info(" Criando labels realistas...")
        
        df = df.copy()
        df = df.reset_index(drop=True)
        
        # Calcular thresholds
        df['tp_target'] = df['close'] * (1 + self.take_profit_pct + self.fee_pct)
        df['sl_target'] = df['close'] * (1 - self.stop_loss_pct - self.fee_pct)
        
        # Olhar próximos N candles (max_duration)
        window = self.max_duration_candles
        
        # Calcular max high e min low nos próximos N candles
        df['future_high'] = df['high'].rolling(window=window, min_periods=1).max().shift(-window)
        df['future_low'] = df['low'].rolling(window=window, min_periods=1).min().shift(-window)
        
        # Criar labels
        df['label'] = 0  # Default: HOLD
        
        # WIN: atingiu TP
        df.loc[df['future_high'] >= df['tp_target'], 'label'] = 1
        
        # LOSS: atingiu SL
        df.loc[df['future_low'] <= df['sl_target'], 'label'] = -1
        
        # Se atingiu ambos, o que vier primeiro ganha (simplificação: prioridade para SL)
        both_hit = (df['future_high'] >= df['tp_target']) & (df['future_low'] <= df['sl_target'])
        df.loc[both_hit, 'label'] = -1  # Conservador: assume SL primeiro
        
        # Remover colunas temporárias
        df = df.drop(['tp_target', 'sl_target', 'future_high', 'future_low'], axis=1)
        
        # Remover últimas linhas (sem dados futuros)
        df = df.iloc[:-window]
        
        # Estatisticas
        label_counts = df['label'].value_counts()
        total_labels = len(df)
        logger.info(f"\nLabels criados:")
        if total_labels > 0:
            logger.info(f"   WIN (1):  {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total_labels*100:.1f}%)")
            logger.info(f"   LOSS (-1): {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/total_labels*100:.1f}%)")
            logger.info(f"   HOLD (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total_labels*100:.1f}%)")
        else:
            logger.warning("Nenhum label criado!")
        
        return df
        
        df['label'] = labels
        df['label_pnl'] = pnl_values
        df['label_duration'] = durations
        df['label_outcome'] = outcomes
        
        # Estatisticas
        label_counts = pd.Series(labels).value_counts()
        total_labels = len(labels)
        logger.info(f"\nLabels criados:")
        if total_labels > 0:
            logger.info(f"   WIN (1):  {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total_labels*100:.1f}%)")
            logger.info(f"   LOSS (-1): {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/total_labels*100:.1f}%)")
            logger.info(f"   HOLD (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total_labels*100:.1f}%)")
        else:
            logger.warning("Nenhum label criado!")
        
        return df
    
    def _simulate_trade(
        self, 
        df: pd.DataFrame, 
        entry_idx: int, 
        entry_price: float
    ) -> Tuple[int, float, int, str]:
        """
        Simula um trade BUY
        
        Returns:
            (label, pnl, duration, outcome)
        """
        tp_price = entry_price * (1 + self.take_profit_pct)
        sl_price = entry_price * (1 - self.stop_loss_pct)
        
        # Percorrer candles seguintes
        for duration in range(1, self.max_duration_candles + 1):
            idx = entry_idx + duration
            
            if idx >= len(df):
                break
            
            high = df.loc[idx, 'high']
            low = df.loc[idx, 'low']
            
            # Verificar se atingiu SL primeiro (usando low)
            if low <= sl_price:
                pnl = -self.stop_loss_pct - self.fee_pct
                return (-1, pnl, duration, 'SL')
            
            # Verificar se atingiu TP (usando high)
            if high >= tp_price:
                pnl = self.take_profit_pct - self.fee_pct
                return (1, pnl, duration, 'TP')
        
        # Timeout: calcular P&L no ltimo candle
        exit_idx = min(entry_idx + self.max_duration_candles, len(df) - 1)
        exit_price = df.loc[exit_idx, 'close']
        pnl = (exit_price - entry_price) / entry_price - self.fee_pct
        
        # Classificar timeout baseado no P&L
        if pnl > 0.001:  # Ganhou mais que 0.1%
            return (1, pnl, self.max_duration_candles, 'TIMEOUT_WIN')
        elif pnl < -0.001:  # Perdeu mais que 0.1%
            return (-1, pnl, self.max_duration_candles, 'TIMEOUT_LOSS')
        else:
            return (0, pnl, self.max_duration_candles, 'TIMEOUT_NEUTRAL')
    
    def create_regression_target(self, df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
        """
        Alternativa: Cria target de regresso (predizer retorno futuro)
        
        Args:
            horizon: Quantos candles  frente olhar
        
        Returns:
            DataFrame com coluna 'target_return' (% de retorno)
        """
        logger.info(f" Criando target de regresso (horizon={horizon})...")
        
        df = df.copy()
        
        # Retorno N candles  frente
        df['target_return'] = df['close'].pct_change(horizon).shift(-horizon)
        
        # Remover ltimos candles sem target
        valid_targets = df['target_return'].notna().sum()
        logger.info(f" Targets criados: {valid_targets:,}")
        logger.info(f"   Mean return: {df['target_return'].mean()*100:.4f}%")
        logger.info(f"   Std return: {df['target_return'].std()*100:.4f}%")
        
        return df
    
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balanceia dataset (mesmo nmero de WIN/LOSS)
        til para evitar vis do modelo
        """
        logger.info(" Balanceando dataset...")
        
        df_valid = df[df['label'].notna()].copy()
        
        wins = df_valid[df_valid['label'] == 1]
        losses = df_valid[df_valid['label'] == -1]
        holds = df_valid[df_valid['label'] == 0]
        
        logger.info(f"   Antes: WIN={len(wins):,}, LOSS={len(losses):,}, HOLD={len(holds):,}")
        
        # Pegar o menor count
        min_count = min(len(wins), len(losses))
        
        # Undersample
        wins_balanced = wins.sample(n=min_count, random_state=42)
        losses_balanced = losses.sample(n=min_count, random_state=42)
        
        # HOLD: pegar 20% do tamanho
        hold_count = int(min_count * 0.2)
        holds_balanced = holds.sample(n=min(hold_count, len(holds)), random_state=42)
        
        # Concatenar
        df_balanced = pd.concat([wins_balanced, losses_balanced, holds_balanced])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"   Depois: WIN={len(wins_balanced):,}, LOSS={len(losses_balanced):,}, HOLD={len(holds_balanced):,}")
        logger.info(f"   Total: {len(df_balanced):,} samples")
        
        return df_balanced


def main():
    """Teste do label creator"""
    logging.basicConfig(level=logging.INFO)
    
    from src.ml.data_loader import HistDataLoader
    from src.ml.feature_engineer import FeatureEngineer
    
    # Carregar dados
    loader = HistDataLoader()
    df = loader.load_processed()
    
    if df is None:
        logger.error("Dados no encontrados")
        return
    
    # Usar amostra pequena para teste
    logger.info(" Testando com 10k candles...")
    df_sample = df.head(10000)
    
    # Criar features primeiro (labels precisam de high/low)
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_sample)
    
    # Criar labels
    label_creator = LabelCreator(
        take_profit_pct=0.004,  # 40 pips
        stop_loss_pct=0.002,    # 20 pips
        max_duration_candles=60  # 1 hora
    )
    
    df_labeled = label_creator.create_realistic_labels(df_features)
    
    logger.info(f"\n Dataset final:")
    logger.info(f"   Shape: {df_labeled.shape}")
    logger.info(f"   Colunas: {df_labeled.columns.tolist()}")
    logger.info(f"\nDistribuio de labels:")
    logger.info(df_labeled['label'].value_counts())
    logger.info(f"\nDistribuio de outcomes:")
    logger.info(df_labeled['label_outcome'].value_counts())


if __name__ == "__main__":
    main()
