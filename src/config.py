"""
Configuraes globais do sistema
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Database
DATABASE_URL = os.getenv('DATABASE_URL')

# OANDA API
OANDA_API_KEY = os.getenv('OANDA_API_KEY')
OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID')
OANDA_ENVIRONMENT = os.getenv('OANDA_ENVIRONMENT', 'practice')

# URLs OANDA
OANDA_URLS = {
    'practice': {
        'stream': 'wss://stream-fxpractice.oanda.com/v3/accounts/{}/pricing/stream',
        'api': 'https://api-fxpractice.oanda.com/v3'
    },
    'live': {
        'stream': 'wss://stream-fxtrade.oanda.com/v3/accounts/{}/pricing/stream',
        'api': 'https://api-fxtrade.oanda.com/v3'
    }
}

OANDA_STREAM_URL = OANDA_URLS[OANDA_ENVIRONMENT]['stream']
OANDA_API_URL = OANDA_URLS[OANDA_ENVIRONMENT]['api']

# Trading Config
TRADING_PAIRS = os.getenv('TRADING_PAIRS', 'EUR_USD,GBP_USD').split(',')
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.02'))
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.04'))
MAX_TRADE_DURATION_MINUTES = int(os.getenv('MAX_TRADE_DURATION_MINUTES', '60'))

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Validao
def validate_config():
    """Valida se todas as configuraes necessrias esto presentes"""
    errors = []
    
    if not DATABASE_URL:
        errors.append("DATABASE_URL no configurada no .env")
    
    if not OANDA_API_KEY:
        errors.append("OANDA_API_KEY no configurada no .env")
    
    if not OANDA_ACCOUNT_ID:
        errors.append("OANDA_ACCOUNT_ID no configurado no .env")
    
    if errors:
        print(" Erros de configurao:")
        for error in errors:
            print(f"   - {error}")
        print("\n Copie .env.example para .env e configure suas credenciais")
        return False
    
    return True
