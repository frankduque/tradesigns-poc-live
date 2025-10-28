"""
Script para iniciar todos os componentes do sistema
"""
import asyncio
import subprocess
import sys
import os
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Adicionar src ao path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

from src.config import validate_config


class SystemManager:
    """Gerencia todos os componentes do sistema"""
    
    def __init__(self):
        self.processes = {}
    
    async def start_all(self):
        """Inicia todos os componentes"""
        logger.info("=" * 70)
        logger.info(" TradeSigns Live System - Iniciando...")
        logger.info("=" * 70)
        
        # 1. Validar configuração
        if not validate_config():
            logger.error("\n Configurao invlida!")
            logger.error(" Passos para configurar:")
            logger.error("   1. cp .env.example .env")
            logger.error("   2. Edite .env com suas credenciais OANDA")
            logger.error("   3. https://www.oanda.com/demo-account/")
            return False
        
        logger.info(" Configurao vlida")
        
        # 2. Verificar PostgreSQL
        if not await self.check_postgres():
            logger.error("\n PostgreSQL no est rodando!")
            logger.error(" Execute: docker-compose up -d postgres")
            return False
        
        logger.info(" PostgreSQL conectado")
        
        # 3. Iniciar componentes
        logger.info("\n Iniciando componentes...\n")
        
        try:
            # Data Streamer
            logger.info(" Iniciando Data Streamer...")
            await self.start_streamer()
            await asyncio.sleep(2)
            
            # Performance Tracker
            logger.info(" Iniciando Performance Tracker...")
            await self.start_tracker()
            await asyncio.sleep(2)
            
            logger.info("\n" + "=" * 70)
            logger.info(" Sistema iniciado com sucesso!")
            logger.info("=" * 70)
            logger.info("\n Componentes rodando:")
            logger.info("    Data Streamer - Recebendo ticks do OANDA")
            logger.info("    Signal Generator - Gerando sinais automticos")
            logger.info("    Performance Tracker - Monitorando P&L")
            
            logger.info("\n Dicas:")
            logger.info("    Para ver o dashboard: streamlit run dashboard/app.py")
            logger.info("    Logs salvos em: logs/")
            logger.info("    Pressione Ctrl+C para parar")
            
            logger.info("\n Sistema LIVE - Aguardando dados...\n")
            
            return True
            
        except Exception as e:
            logger.error(f" Erro ao iniciar componentes: {e}")
            return False
    
    async def start_streamer(self):
        """Inicia o data streamer"""
        from src.data.live_streamer import main as streamer_main
        
        # Criar task assíncrona
        task = asyncio.create_task(streamer_main())
        self.processes['streamer'] = task
    
    async def start_tracker(self):
        """Inicia o performance tracker"""
        from src.performance.tracker import performance_tracker
        
        # Criar task assíncrona
        task = asyncio.create_task(performance_tracker.start())
        self.processes['tracker'] = task
    
    async def check_postgres(self) -> bool:
        """Verifica se PostgreSQL está acessível"""
        try:
            from src.database.connection import engine
            from sqlalchemy import text
            
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.debug(f"PostgreSQL check failed: {e}")
            return False
    
    async def wait_for_completion(self):
        """Aguarda os processos rodarem"""
        try:
            # Aguarda tasks rodarem
            tasks = list(self.processes.values())
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except KeyboardInterrupt:
            logger.info("\n Recebido sinal de interrupo...")
    
    async def stop_all(self):
        """Para todos os componentes"""
        logger.info("\n Parando sistema...")
        
        for name, task in self.processes.items():
            logger.info(f"   Parando {name}...")
            task.cancel()
        
        logger.info(" Sistema parado")


async def main():
    """Função principal"""
    manager = SystemManager()
    
    try:
        # Iniciar sistema
        success = await manager.start_all()
        
        if not success:
            return
        
        # Aguardar execução
        await manager.wait_for_completion()
        
    except KeyboardInterrupt:
        logger.info("\n Interrompido pelo usurio")
    except Exception as e:
        logger.error(f" Erro fatal: {e}", exc_info=True)
    finally:
        await manager.stop_all()


if __name__ == "__main__":
    # Configurar logging para arquivo também
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / "system.log")
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)
    
    # Rodar sistema
    asyncio.run(main())
