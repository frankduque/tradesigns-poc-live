"""
Database repositories - Data Access Layer
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import pandas as pd
from sqlalchemy import text, desc
from src.database.connection import get_db_sync

class PriceRepository:
    """Repository para preos"""
    
    @staticmethod
    def save_candle(candle: dict):
        """Salva um candle completo no banco"""
        db = get_db_sync()
        try:
            query = text("""
                INSERT INTO live_prices (pair, timestamp, open, high, low, close, ticks)
                VALUES (:pair, :timestamp, :open, :high, :low, :close, :ticks)
                ON CONFLICT (pair, timestamp) DO UPDATE SET
                    high = GREATEST(live_prices.high, EXCLUDED.high),
                    low = LEAST(live_prices.low, EXCLUDED.low),
                    close = EXCLUDED.close,
                    ticks = live_prices.ticks + EXCLUDED.ticks
            """)
            
            db.execute(query, {
                'pair': candle['pair'],
                'timestamp': candle['timestamp'],
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'ticks': candle.get('ticks', 1)
            })
            db.commit()
        except Exception as e:
            db.rollback()
            print(f" Erro ao salvar candle: {e}")
        finally:
            db.close()
    
    @staticmethod
    def get_recent_candles(pair: str, limit: int = 200) -> pd.DataFrame:
        """Busca candles recentes para clculo de indicadores"""
        db = get_db_sync()
        try:
            query = text("""
                SELECT timestamp, open, high, low, close
                FROM live_prices
                WHERE pair = :pair
                ORDER BY timestamp DESC
                LIMIT :limit
            """)
            
            result = db.execute(query, {'pair': pair, 'limit': limit})
            rows = result.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').set_index('timestamp')
            
            return df
        finally:
            db.close()
    
    @staticmethod
    def get_current_price(pair: str) -> Optional[float]:
        """Retorna o preo atual (ltimo close)"""
        db = get_db_sync()
        try:
            query = text("""
                SELECT close FROM live_prices
                WHERE pair = :pair
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            result = db.execute(query, {'pair': pair})
            row = result.fetchone()
            
            return float(row[0]) if row else None
        finally:
            db.close()


class SignalRepository:
    """Repository para sinais"""
    
    @staticmethod
    def save_signal(signal_data: dict):
        """Salva um novo sinal"""
        db = get_db_sync()
        try:
            query = text("""
                INSERT INTO signals 
                (pair, timestamp, signal_type, strategy, entry_price, score, indicators, status)
                VALUES 
                (:pair, :timestamp, :signal_type, :strategy, :entry_price, :score, :indicators::jsonb, 'OPEN')
                RETURNING id
            """)
            
            import json
            result = db.execute(query, {
                'pair': signal_data['pair'],
                'timestamp': signal_data['timestamp'],
                'signal_type': signal_data['signal_type'],
                'strategy': signal_data['strategy'],
                'entry_price': signal_data['entry_price'],
                'score': signal_data['score'],
                'indicators': json.dumps(signal_data.get('indicators', {}))
            })
            
            signal_id = result.fetchone()[0]
            db.commit()
            
            print(f" Sinal salvo: ID={signal_id} {signal_data['signal_type']} {signal_data['pair']}")
            return signal_id
            
        except Exception as e:
            db.rollback()
            print(f" Erro ao salvar sinal: {e}")
            return None
        finally:
            db.close()
    
    @staticmethod
    def get_open_signals() -> List[Dict]:
        """Retorna todos os sinais abertos"""
        db = get_db_sync()
        try:
            query = text("""
                SELECT id, pair, timestamp, signal_type, strategy, entry_price, score
                FROM signals
                WHERE status = 'OPEN'
                ORDER BY timestamp DESC
            """)
            
            result = db.execute(query)
            rows = result.fetchall()
            
            return [
                {
                    'id': row[0],
                    'pair': row[1],
                    'timestamp': row[2],
                    'signal_type': row[3],
                    'strategy': row[4],
                    'entry_price': float(row[5]),
                    'score': float(row[6]) if row[6] else 0
                }
                for row in rows
            ]
        finally:
            db.close()
    
    @staticmethod
    def close_signal(signal_id: int, exit_price: float, pnl_pct: float, outcome: str):
        """Fecha um sinal com resultado"""
        db = get_db_sync()
        try:
            query = text("""
                UPDATE signals SET
                    status = 'CLOSED',
                    exit_price = :exit_price,
                    exit_timestamp = NOW(),
                    pnl_pct = :pnl_pct,
                    outcome = :outcome
                WHERE id = :signal_id
            """)
            
            db.execute(query, {
                'signal_id': signal_id,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'outcome': outcome
            })
            db.commit()
        except Exception as e:
            db.rollback()
            print(f" Erro ao fechar sinal {signal_id}: {e}")
        finally:
            db.close()
    
    @staticmethod
    def get_signals_last_24h() -> List[Dict]:
        """Retorna sinais das ltimas 24h"""
        db = get_db_sync()
        try:
            query = text("""
                SELECT id, pair, timestamp, signal_type, strategy, score, outcome, pnl_pct
                FROM signals
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                ORDER BY timestamp DESC
            """)
            
            result = db.execute(query)
            rows = result.fetchall()
            
            return [
                {
                    'id': row[0],
                    'pair': row[1],
                    'timestamp': row[2],
                    'signal_type': row[3],
                    'strategy': row[4],
                    'score': float(row[5]) if row[5] else 0,
                    'outcome': row[6],
                    'pnl_pct': float(row[7]) if row[7] else 0
                }
                for row in rows
            ]
        finally:
            db.close()
    
    @staticmethod
    def get_performance_stats_24h() -> Optional[Dict]:
        """Retorna estatsticas de performance das ltimas 24h"""
        db = get_db_sync()
        try:
            query = text("SELECT * FROM stats_24h")
            result = db.execute(query)
            row = result.fetchone()
            
            if not row:
                return None
            
            return {
                'total_signals': row[0] or 0,
                'wins': row[1] or 0,
                'losses': row[2] or 0,
                'timeouts': row[3] or 0,
                'win_rate': float(row[4]) if row[4] else 0.0,
                'total_pnl': float(row[5]) if row[5] else 0.0,
                'avg_pnl': float(row[6]) if row[6] else 0.0
            }
        finally:
            db.close()


# Instncias globais
price_repo = PriceRepository()
signal_repo = SignalRepository()
