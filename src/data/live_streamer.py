"""
Live WebSocket Streamer - OANDA
Conecta ao OANDA e recebe ticks em tempo real
Constri candles de 1 minuto e salva no banco
"""
import asyncio
import websockets
import json
from datetime import datetime, timezone
import logging
from typing import Dict, List
from src.config import (
    OANDA_API_KEY, 
    OANDA_ACCOUNT_ID, 
    OANDA_STREAM_URL,
    TRADING_PAIRS,
    validate_config
)
from src.database.repositories import price_repo

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveDataStreamer:
    """WebSocket streamer para OANDA"""
    
    def __init__(self, pairs: List[str] = None):
        self.pairs = pairs or TRADING_PAIRS
        self.current_candles: Dict[str, Dict] = {}
        self.running = False
        
        # Converter pares para formato OANDA (EURUSD -> EUR_USD)
        self.oanda_pairs = [
            pair if '_' in pair else f"{pair[:3]}_{pair[3:]}"
            for pair in self.pairs
        ]
        
        logger.info(f"Streamer inicializado para pares: {self.oanda_pairs}")
    
    async def connect_and_stream(self):
        """Conecta ao WebSocket OANDA e inicia streaming"""
        if not validate_config():
            logger.error("Configurao invlida. Verifique o .env")
            return
        
        url = OANDA_STREAM_URL.format(OANDA_ACCOUNT_ID)
        instruments = ",".join(self.oanda_pairs)
        full_url = f"{url}?instruments={instruments}"
        
        headers = {
            "Authorization": f"Bearer {OANDA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries:
            try:
                logger.info(f" Conectando ao OANDA WebSocket...")
                logger.info(f" Pares: {', '.join(self.oanda_pairs)}")
                
                async with websockets.connect(
                    full_url,
                    extra_headers=headers,
                    ping_interval=20,
                    ping_timeout=10
                ) as websocket:
                    logger.info(" Conectado ao OANDA!")
                    self.running = True
                    retry_count = 0  # Reset contador em conexo bem-sucedida
                    
                    async for message in websocket:
                        try:
                            await self.process_message(json.loads(message))
                        except Exception as e:
                            logger.error(f"Erro ao processar mensagem: {e}")
                            continue
                            
            except websockets.exceptions.WebSocketException as e:
                retry_count += 1
                logger.error(f" WebSocket error (tentativa {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    wait_time = min(2 ** retry_count, 60)  # Exponential backoff
                    logger.info(f" Reconectando em {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(" Mximo de tentativas atingido. Abortando.")
                    break
                    
            except Exception as e:
                logger.error(f" Erro inesperado: {e}", exc_info=True)
                break
        
        self.running = False
        logger.info(" Streamer parado")
    
    async def process_message(self, data: dict):
        """Processa mensagem recebida do WebSocket"""
        msg_type = data.get('type')
        
        if msg_type == 'HEARTBEAT':
            logger.debug(" Heartbeat recebido")
            return
        
        if msg_type != 'PRICE':
            logger.debug(f"Mensagem ignorada: {msg_type}")
            return
        
        # Extrair dados do tick
        try:
            instrument = data['instrument']
            pair = instrument.replace('_', '')  # EUR_USD -> EURUSD
            
            # Pegar bid/ask
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            if not bids or not asks:
                return
            
            bid = float(bids[0]['price'])
            ask = float(asks[0]['price'])
            mid = (bid + ask) / 2
            
            # Timestamp
            time_str = data['time']
            timestamp = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            
            # Construir candle
            await self.build_candle(pair, mid, timestamp)
            
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Erro ao extrair dados do tick: {e}")
    
    async def build_candle(self, pair: str, price: float, timestamp: datetime):
        """Constri candles de 1 minuto a partir dos ticks"""
        # Arredondar timestamp para o minuto
        candle_time = timestamp.replace(second=0, microsecond=0)
        key = f"{pair}_{candle_time.isoformat()}"
        
        if key not in self.current_candles:
            # Novo candle
            self.current_candles[key] = {
                'pair': pair,
                'timestamp': candle_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'ticks': 1
            }
            logger.debug(f" Novo candle iniciado: {pair} @ {candle_time}")
        else:
            # Atualiza candle existente
            candle = self.current_candles[key]
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            candle['ticks'] += 1
        
        # Verificar candles completos
        await self.check_completed_candles(candle_time)
    
    async def check_completed_candles(self, current_time: datetime):
        """Verifica e salva candles completos"""
        completed_keys = []
        
        for key, candle in self.current_candles.items():
            # Candle est completo se passou mais de 1 minuto
            time_diff = (current_time - candle['timestamp']).total_seconds()
            
            if time_diff >= 60:
                # Salvar no banco
                try:
                    price_repo.save_candle(candle)
                    logger.info(
                        f" Candle salvo: {candle['pair']} @ {candle['timestamp'].strftime('%H:%M')} "
                        f"| O:{candle['open']:.5f} H:{candle['high']:.5f} "
                        f"L:{candle['low']:.5f} C:{candle['close']:.5f} "
                        f"| {candle['ticks']} ticks"
                    )
                    completed_keys.append(key)
                    
                    # Trigger signal generation
                    await self.trigger_signal_generation(candle)
                    
                except Exception as e:
                    logger.error(f" Erro ao salvar candle: {e}")
        
        # Limpar candles completos do buffer
        for key in completed_keys:
            del self.current_candles[key]
    
    async def trigger_signal_generation(self, candle: dict):
        """Notifica signal generator sobre novo candle"""
        try:
            from src.signals.generator import generate_signal_for_candle
            await generate_signal_for_candle(candle)
        except Exception as e:
            logger.error(f" Erro ao gerar sinal: {e}")
    
    def stop(self):
        """Para o streamer"""
        logger.info(" Parando streamer...")
        self.running = False


async def main():
    """Funo principal para rodar o streamer"""
    logger.info("=" * 60)
    logger.info(" TradeSigns Live Data Streamer")
    logger.info("=" * 60)
    
    # Validar configurao
    if not validate_config():
        logger.error(" Configurao invlida. Execute:")
        logger.error("   1. cp .env.example .env")
        logger.error("   2. Edite .env com suas credenciais OANDA")
        return
    
    # Criar e iniciar streamer
    streamer = LiveDataStreamer()
    
    try:
        await streamer.connect_and_stream()
    except KeyboardInterrupt:
        logger.info("\n Interrompido pelo usurio")
        streamer.stop()
    except Exception as e:
        logger.error(f" Erro fatal: {e}", exc_info=True)
    finally:
        logger.info(" Streamer finalizado")


if __name__ == "__main__":
    asyncio.run(main())
