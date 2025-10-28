"""
Performance Tracker - Monitora sinais abertos e calcula P&L
"""
import asyncio
import logging
from datetime import datetime, timedelta
from src.config import STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_TRADE_DURATION_MINUTES
from src.database.repositories import signal_repo, price_repo

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Monitora e fecha sinais baseado em SL/TP/Timeout"""
    
    def __init__(self):
        self.stop_loss_pct = STOP_LOSS_PCT
        self.take_profit_pct = TAKE_PROFIT_PCT
        self.max_duration_minutes = MAX_TRADE_DURATION_MINUTES
        self.check_interval = 5  # Verificar a cada 5 segundos
        self.running = False
        
        logger.info(
            f"Performance Tracker inicializado: "
            f"SL={self.stop_loss_pct*100:.1f}% | "
            f"TP={self.take_profit_pct*100:.1f}% | "
            f"Max Duration={self.max_duration_minutes}min"
        )
    
    async def start(self):
        """Inicia o loop de monitoramento"""
        self.running = True
        logger.info(" Performance Tracker iniciado")
        
        try:
            while self.running:
                await self.check_open_signals()
                await asyncio.sleep(self.check_interval)
        except Exception as e:
            logger.error(f" Erro no tracker: {e}", exc_info=True)
        finally:
            self.running = False
            logger.info(" Performance Tracker parado")
    
    async def check_open_signals(self):
        """Verifica todos os sinais abertos"""
        try:
            open_signals = signal_repo.get_open_signals()
            
            if not open_signals:
                logger.debug("Nenhum sinal aberto")
                return
            
            logger.debug(f" Monitorando {len(open_signals)} sinais abertos")
            
            for signal in open_signals:
                await self.check_signal_exit(signal)
        
        except Exception as e:
            logger.error(f" Erro ao verificar sinais: {e}")
    
    async def check_signal_exit(self, signal: dict):
        """Verifica se um sinal deve ser fechado"""
        try:
            # Obter preo atual
            current_price = price_repo.get_current_price(signal['pair'])
            
            if current_price is None:
                logger.warning(f" Preo no disponvel para {signal['pair']}")
                return
            
            entry_price = signal['entry_price']
            signal_type = signal['signal_type']
            
            # Calcular P&L
            if signal_type == 'BUY':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SELL
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Verificar condies de sada
            should_exit = False
            outcome = None
            reason = None
            
            # 1. Stop Loss
            if pnl_pct <= -self.stop_loss_pct:
                should_exit = True
                outcome = 'LOSS'
                reason = 'Stop Loss'
            
            # 2. Take Profit
            elif pnl_pct >= self.take_profit_pct:
                should_exit = True
                outcome = 'WIN'
                reason = 'Take Profit'
            
            # 3. Timeout
            else:
                duration = (datetime.now() - signal['timestamp']).total_seconds() / 60
                if duration >= self.max_duration_minutes:
                    should_exit = True
                    outcome = 'TIMEOUT'
                    reason = 'Max Duration'
            
            # Fechar sinal se necessrio
            if should_exit:
                signal_repo.close_signal(
                    signal_id=signal['id'],
                    exit_price=current_price,
                    pnl_pct=pnl_pct,
                    outcome=outcome
                )
                
                # Log do fechamento
                emoji = "" if outcome == 'WIN' else "" if outcome == 'LOSS' else ""
                logger.info(
                    f"{emoji} Sinal #{signal['id']} fechado: "
                    f"{signal['signal_type']} {signal['pair']} | "
                    f"Entry: {entry_price:.5f}  Exit: {current_price:.5f} | "
                    f"P&L: {pnl_pct*100:+.2f}% | "
                    f"Outcome: {outcome} ({reason})"
                )
        
        except Exception as e:
            logger.error(f" Erro ao verificar sinal #{signal['id']}: {e}")
    
    def stop(self):
        """Para o tracker"""
        self.running = False


# Instncia global
performance_tracker = PerformanceTracker()


async def main():
    """Funo principal para rodar o tracker standalone"""
    logger.info("=" * 60)
    logger.info(" TradeSigns Performance Tracker")
    logger.info("=" * 60)
    
    try:
        await performance_tracker.start()
    except KeyboardInterrupt:
        logger.info("\n Interrompido pelo usurio")
        performance_tracker.stop()
    except Exception as e:
        logger.error(f" Erro fatal: {e}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
