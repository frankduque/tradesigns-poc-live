"""
Regression Signal Generator

Gera sinais de trading baseado em previsões de regressão,
com threshold dinâmico baseado em volatilidade (ATR).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


class RegressionSignalGenerator:
    """
    Gera sinais de trading baseado em previsões de movimento futuro.
    
    Sinais:
    - BUY (1): Quando modelo prevê movimento positivo > threshold
    - SELL (-1): Quando modelo prevê movimento negativo < -threshold
    - HOLD (0): Caso contrário
    """
    
    def __init__(
        self,
        model,
        base_threshold: float = 0.05,
        atr_multiplier: float = 2.0,
        use_confidence: bool = True
    ):
        """
        Args:
            model: Modelo treinado (MultiTargetRegressionTrainer)
            base_threshold: Threshold base em % (0.05 = 5 pips)
            atr_multiplier: Multiplica ATR para ajustar threshold
            use_confidence: Se True, considera múltiplos horizontes
        """
        self.model = model
        self.base_threshold = base_threshold
        self.atr_multiplier = atr_multiplier
        self.use_confidence = use_confidence
        
        logger.info("Regression Signal Generator inicializado")
        logger.info(f"   Base Threshold: {base_threshold:.2f}%")
        logger.info(f"   ATR Multiplier: {atr_multiplier}")
        logger.info(f"   Use Confidence: {use_confidence}")
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        features: np.ndarray,
        method: str = 'simple'
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Gera sinais de trading
        
        Args:
            df: DataFrame com OHLCV e ATR
            features: Features para previsão (N x 64)
            method: 'simple' ou 'advanced'
        
        Returns:
            (signals, predictions_df)
            - signals: Series com {-1, 0, 1}
            - predictions_df: DataFrame com previsões detalhadas
        """
        logger.info(f"\nGerando sinais (method={method})...")
        
        # Fazer previsões
        predictions = self.model.predict(features)  # (N x 3)
        
        # Criar DataFrame com previsões
        predictions_df = pd.DataFrame(
            predictions,
            columns=['pred_return_5m', 'pred_return_10m', 'pred_return_30m'],
            index=df.index[:len(predictions)]
        )
        
        if method == 'simple':
            signals = self._generate_simple_signals(df, predictions_df)
        else:
            signals = self._generate_advanced_signals(df, predictions_df)
        
        # Estatísticas
        self._log_signal_statistics(signals)
        
        return signals, predictions_df
    
    def _generate_simple_signals(
        self,
        df: pd.DataFrame,
        predictions_df: pd.DataFrame
    ) -> pd.Series:
        """
        Método simples: usa previsão de 10min com threshold dinâmico
        """
        # Usar previsão de 10min (horizonte médio)
        predicted_return = predictions_df['pred_return_10m'].values
        
        # Threshold dinâmico baseado em ATR
        if 'atr' in df.columns:
            atr = df['atr'].iloc[:len(predicted_return)].values
            close = df['close'].iloc[:len(predicted_return)].values
            atr_normalized = atr / close  # ATR como % do preço
            
            # Threshold aumenta com volatilidade
            dynamic_threshold = self.base_threshold * (1 + atr_normalized * self.atr_multiplier)
        else:
            logger.warning("ATR não encontrado, usando threshold fixo")
            dynamic_threshold = np.full(len(predicted_return), self.base_threshold)
        
        # Gerar sinais
        signals = np.zeros(len(predicted_return))
        signals[predicted_return > dynamic_threshold] = 1   # BUY
        signals[predicted_return < -dynamic_threshold] = -1  # SELL
        
        return pd.Series(signals, index=predictions_df.index, dtype=int)
    
    def _generate_advanced_signals(
        self,
        df: pd.DataFrame,
        predictions_df: pd.DataFrame
    ) -> pd.Series:
        """
        Método avançado: considera múltiplos horizontes para confirmar tendência
        """
        pred_5m = predictions_df['pred_return_5m'].values
        pred_10m = predictions_df['pred_return_10m'].values
        pred_30m = predictions_df['pred_return_30m'].values
        
        signals = np.zeros(len(pred_10m))
        
        # Thresholds diferentes por horizonte
        th_5m = 0.03   # 3 pips em 5min
        th_10m = 0.05  # 5 pips em 10min
        th_30m = 0.08  # 8 pips em 30min
        
        # BUY: movimento positivo consistente em todos os horizontes
        buy_condition = (
            (pred_5m > th_5m) &
            (pred_10m > th_10m) &
            (pred_30m > th_30m)
        )
        
        # SELL: movimento negativo consistente
        sell_condition = (
            (pred_5m < -th_5m) &
            (pred_10m < -th_10m) &
            (pred_30m < -th_30m)
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return pd.Series(signals, index=predictions_df.index, dtype=int)
    
    def _log_signal_statistics(self, signals: pd.Series):
        """Log estatísticas dos sinais gerados"""
        total = len(signals)
        n_buy = (signals == 1).sum()
        n_sell = (signals == -1).sum()
        n_hold = (signals == 0).sum()
        
        logger.info("\nDistribuicao de Sinais:")
        logger.info(f"   BUY:  {n_buy:,} ({n_buy/total*100:.1f}%)")
        logger.info(f"   SELL: {n_sell:,} ({n_sell/total*100:.1f}%)")
        logger.info(f"   HOLD: {n_hold:,} ({n_hold/total*100:.1f}%)")
        logger.info(f"   Total signals: {n_buy + n_sell:,}")
    
    def backtest_signals(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        predictions_df: pd.DataFrame,
        tp_multiplier: float = 1.0,
        sl_multiplier: float = 0.5,
        max_duration: int = 30
    ) -> pd.DataFrame:
        """
        Backtest com TP/SL dinâmicos baseados na previsão
        
        Args:
            df: DataFrame com OHLCV
            signals: Sinais gerados
            predictions_df: Previsões do modelo
            tp_multiplier: Multiplicador para TP (1.0 = usar previsão exata)
            sl_multiplier: Multiplicador para SL (0.5 = metade da previsão)
            max_duration: Duração máxima em candles
        
        Returns:
            DataFrame com trades executados
        """
        logger.info("\nExecutando backtest com TP/SL dinamicos...")
        
        trades = []
        
        for i in range(len(signals)):
            if signals.iloc[i] == 0:
                continue
            
            signal_type = 'BUY' if signals.iloc[i] == 1 else 'SELL'
            entry_time = signals.index[i]
            entry_price = df.loc[entry_time, 'close']
            
            # Usar previsão de 10min para calcular TP/SL
            predicted_move = abs(predictions_df.loc[entry_time, 'pred_return_10m'])
            
            # TP/SL dinâmicos
            if signal_type == 'BUY':
                take_profit = entry_price * (1 + predicted_move / 100 * tp_multiplier)
                stop_loss = entry_price * (1 - predicted_move / 100 * sl_multiplier)
            else:
                take_profit = entry_price * (1 - predicted_move / 100 * tp_multiplier)
                stop_loss = entry_price * (1 + predicted_move / 100 * sl_multiplier)
            
            # Simular trade
            outcome = self._simulate_trade(
                df.iloc[i:i+max_duration+1],
                entry_price,
                take_profit,
                stop_loss,
                signal_type
            )
            
            outcome['entry_time'] = entry_time
            outcome['predicted_move'] = predicted_move
            trades.append(outcome)
        
        trades_df = pd.DataFrame(trades)
        
        # Calcular métricas
        if len(trades_df) > 0:
            self._log_backtest_metrics(trades_df)
        
        return trades_df
    
    def _simulate_trade(
        self,
        df_window: pd.DataFrame,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        signal_type: str
    ) -> dict:
        """Simula execução de um trade"""
        for i, (timestamp, row) in enumerate(df_window.iterrows()):
            if i == 0:
                continue  # Pular barra de entrada
            
            high = row['high']
            low = row['low']
            
            if signal_type == 'BUY':
                # Checar TP
                if high >= take_profit:
                    return {
                        'outcome': 'WIN',
                        'exit_price': take_profit,
                        'pnl_pct': (take_profit - entry_price) / entry_price * 100,
                        'duration': i,
                        'signal_type': signal_type
                    }
                # Checar SL
                if low <= stop_loss:
                    return {
                        'outcome': 'LOSS',
                        'exit_price': stop_loss,
                        'pnl_pct': (stop_loss - entry_price) / entry_price * 100,
                        'duration': i,
                        'signal_type': signal_type
                    }
            else:  # SELL
                # Checar TP
                if low <= take_profit:
                    return {
                        'outcome': 'WIN',
                        'exit_price': take_profit,
                        'pnl_pct': (entry_price - take_profit) / entry_price * 100,
                        'duration': i,
                        'signal_type': signal_type
                    }
                # Checar SL
                if high >= stop_loss:
                    return {
                        'outcome': 'LOSS',
                        'exit_price': stop_loss,
                        'pnl_pct': (entry_price - stop_loss) / entry_price * 100,
                        'duration': i,
                        'signal_type': signal_type
                    }
        
        # Timeout
        exit_price = df_window.iloc[-1]['close']
        if signal_type == 'BUY':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        return {
            'outcome': 'TIMEOUT',
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'duration': len(df_window) - 1,
            'signal_type': signal_type
        }
    
    def _log_backtest_metrics(self, trades_df: pd.DataFrame):
        """Log métricas do backtest"""
        total_trades = len(trades_df)
        wins = (trades_df['outcome'] == 'WIN').sum()
        losses = (trades_df['outcome'] == 'LOSS').sum()
        timeouts = (trades_df['outcome'] == 'TIMEOUT').sum()
        
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        avg_pnl = trades_df['pnl_pct'].mean()
        avg_win = trades_df[trades_df['outcome'] == 'WIN']['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['outcome'] == 'LOSS']['pnl_pct'].mean() if losses > 0 else 0
        
        logger.info("\nResultados do Backtest:")
        logger.info(f"   Total Trades: {total_trades:,}")
        logger.info(f"   Wins:         {wins:,} ({win_rate:.1f}%)")
        logger.info(f"   Losses:       {losses:,} ({losses/total_trades*100:.1f}%)")
        logger.info(f"   Timeouts:     {timeouts:,} ({timeouts/total_trades*100:.1f}%)")
        logger.info(f"   Avg P&L:      {avg_pnl:.4f}%")
        logger.info(f"   Avg Win:      {avg_win:.4f}%")
        logger.info(f"   Avg Loss:     {avg_loss:.4f}%")
        
        if wins > 0 and losses > 0:
            profit_factor = abs(avg_win * wins / (avg_loss * losses))
            logger.info(f"   Profit Factor: {profit_factor:.2f}")


if __name__ == "__main__":
    print("\nRegression Signal Generator ready!")
    print("Use generate_signals_regression.py para gerar sinais.")
