"""
Backtest Engine - Testa estratgias em dados histricos
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Representa um trade executado"""
    entry_time: pd.Timestamp
    entry_price: float
    signal_type: str  # BUY ou SELL
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl_pct: float = None
    outcome: str = None  # TP, SL, TIMEOUT
    duration_minutes: int = None


class BacktestEngine:
    """Engine para backtesting de estratgias"""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        take_profit_pct: float = 0.004,
        stop_loss_pct: float = 0.002,
        max_duration_minutes: int = 60,
        fee_pct: float = 0.0002
    ):
        self.initial_capital = initial_capital
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_duration_minutes = max_duration_minutes
        self.fee_pct = fee_pct
        
        self.trades: List[Trade] = []
        self.equity_curve = []
        
        logger.info(f"Backtest Engine inicializado:")
        logger.info(f"   Capital: ${initial_capital:,.2f}")
        logger.info(f"   TP: {take_profit_pct*100:.2f}%")
        logger.info(f"   SL: {stop_loss_pct*100:.2f}%")
    
    def run(
        self, 
        df: pd.DataFrame, 
        signals: pd.Series
    ) -> Dict:
        """
        Executa backtest
        
        Args:
            df: DataFrame com preos (timestamp, open, high, low, close)
            signals: Series com sinais (1=BUY, -1=SELL, 0=HOLD)
        
        Returns:
            Dict com mtricas de performance
        """
        logger.info(" Executando backtest...")
        
        self.trades = []
        self.equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        
        df = df.reset_index(drop=True)
        signals = signals.reset_index(drop=True)
        
        i = 0
        while i < len(df) - self.max_duration_minutes:
            signal = signals.iloc[i]
            
            if signal == 1:  # BUY signal
                trade = self._execute_trade(df, i, 'BUY')
                
                if trade.outcome:
                    # Atualizar capital
                    pnl = current_capital * trade.pnl_pct
                    current_capital += pnl
                    self.equity_curve.append(current_capital)
                    
                    self.trades.append(trade)
                    
                    # Avanar para depois do trade
                    i = df[df['timestamp'] == trade.exit_time].index[0] + 1
                    continue
            
            i += 1
        
        # Calcular mtricas
        metrics = self._calculate_metrics()
        
        logger.info(f" Backtest concludo: {len(self.trades)} trades executados")
        
        return metrics
    
    def _execute_trade(self, df: pd.DataFrame, entry_idx: int, signal_type: str) -> Trade:
        """Executa um trade e retorna resultado"""
        
        entry_row = df.iloc[entry_idx]
        trade = Trade(
            entry_time=entry_row['timestamp'],
            entry_price=entry_row['close'],
            signal_type=signal_type
        )
        
        if signal_type == 'BUY':
            tp_price = trade.entry_price * (1 + self.take_profit_pct)
            sl_price = trade.entry_price * (1 - self.stop_loss_pct)
        else:  # SELL
            tp_price = trade.entry_price * (1 - self.take_profit_pct)
            sl_price = trade.entry_price * (1 + self.stop_loss_pct)
        
        # Simular durao do trade
        for duration in range(1, self.max_duration_minutes + 1):
            idx = entry_idx + duration
            
            if idx >= len(df):
                break
            
            candle = df.iloc[idx]
            
            # Verificar SL/TP
            if signal_type == 'BUY':
                if candle['low'] <= sl_price:
                    # Stop Loss hit
                    trade.exit_time = candle['timestamp']
                    trade.exit_price = sl_price
                    trade.pnl_pct = -self.stop_loss_pct - self.fee_pct
                    trade.outcome = 'SL'
                    trade.duration_minutes = duration
                    return trade
                
                elif candle['high'] >= tp_price:
                    # Take Profit hit
                    trade.exit_time = candle['timestamp']
                    trade.exit_price = tp_price
                    trade.pnl_pct = self.take_profit_pct - self.fee_pct
                    trade.outcome = 'TP'
                    trade.duration_minutes = duration
                    return trade
            
            else:  # SELL
                if candle['high'] >= sl_price:
                    trade.exit_time = candle['timestamp']
                    trade.exit_price = sl_price
                    trade.pnl_pct = -self.stop_loss_pct - self.fee_pct
                    trade.outcome = 'SL'
                    trade.duration_minutes = duration
                    return trade
                
                elif candle['low'] <= tp_price:
                    trade.exit_time = candle['timestamp']
                    trade.exit_price = tp_price
                    trade.pnl_pct = self.take_profit_pct - self.fee_pct
                    trade.outcome = 'TP'
                    trade.duration_minutes = duration
                    return trade
        
        # Timeout
        exit_idx = min(entry_idx + self.max_duration_minutes, len(df) - 1)
        exit_candle = df.iloc[exit_idx]
        
        if signal_type == 'BUY':
            pnl_pct = (exit_candle['close'] - trade.entry_price) / trade.entry_price - self.fee_pct
        else:
            pnl_pct = (trade.entry_price - exit_candle['close']) / trade.entry_price - self.fee_pct
        
        trade.exit_time = exit_candle['timestamp']
        trade.exit_price = exit_candle['close']
        trade.pnl_pct = pnl_pct
        trade.outcome = 'TIMEOUT'
        trade.duration_minutes = self.max_duration_minutes
        
        return trade
    
    def _calculate_metrics(self) -> Dict:
        """Calcula mtricas de performance"""
        
        if not self.trades:
            return {}
        
        wins = [t for t in self.trades if t.pnl_pct > 0]
        losses = [t for t in self.trades if t.pnl_pct < 0]
        
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (simplified)
        returns = pd.Series([t.pnl_pct for t in self.trades])
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        metrics = {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) if self.trades else 0,
            'total_return': total_return,
            'final_capital': self.equity_curve[-1],
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'avg_win': np.mean([t.pnl_pct for t in wins]) if wins else 0,
            'avg_loss': np.mean([t.pnl_pct for t in losses]) if losses else 0,
            'profit_factor': abs(sum(t.pnl_pct for t in wins) / sum(t.pnl_pct for t in losses)) if losses else 0,
            'avg_duration': np.mean([t.duration_minutes for t in self.trades]),
            'tp_count': len([t for t in self.trades if t.outcome == 'TP']),
            'sl_count': len([t for t in self.trades if t.outcome == 'SL']),
            'timeout_count': len([t for t in self.trades if t.outcome == 'TIMEOUT'])
        }
        
        return metrics
    
    def get_trades_df(self) -> pd.DataFrame:
        """Retorna trades como DataFrame"""
        return pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'signal_type': t.signal_type,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl_pct': t.pnl_pct,
                'outcome': t.outcome,
                'duration_minutes': t.duration_minutes
            }
            for t in self.trades
        ])
    
    def print_summary(self, metrics: Dict):
        """Imprime resumo do backtest"""
        logger.info("\n" + "=" * 70)
        logger.info(" RESULTADOS DO BACKTEST")
        logger.info("=" * 70)
        logger.info(f" Capital Inicial: ${self.initial_capital:,.2f}")
        logger.info(f" Capital Final: ${metrics['final_capital']:,.2f}")
        logger.info(f" Retorno Total: {metrics['total_return']*100:+.2f}%")
        logger.info(f"")
        logger.info(f" Trades:")
        logger.info(f"   Total: {metrics['total_trades']}")
        logger.info(f"   Wins: {metrics['wins']} ({metrics['win_rate']*100:.1f}%)")
        logger.info(f"   Losses: {metrics['losses']}")
        logger.info(f"")
        logger.info(f" Outcomes:")
        logger.info(f"   Take Profit: {metrics['tp_count']}")
        logger.info(f"   Stop Loss: {metrics['sl_count']}")
        logger.info(f"   Timeout: {metrics['timeout_count']}")
        logger.info(f"")
        logger.info(f" Performance:")
        logger.info(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"   Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"   Avg Win: {metrics['avg_win']*100:+.2f}%")
        logger.info(f"   Avg Loss: {metrics['avg_loss']*100:.2f}%")
        logger.info(f"   Avg Duration: {metrics['avg_duration']:.0f} min")
