"""
Script de teste rápido para validar setup
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

def test_imports():
    """Testa se todas as importações funcionam"""
    print("🧪 Testando importações...")
    
    try:
        from src import config
        print("✅ src.config")
        
        from src.database import connection
        print("✅ src.database.connection")
        
        from src.database import repositories
        print("✅ src.database.repositories")
        
        from src.indicators import technical
        print("✅ src.indicators.technical")
        
        from src.strategies import base, sma_cross
        print("✅ src.strategies")
        
        from src.signals import generator
        print("✅ src.signals.generator")
        
        from src.performance import tracker
        print("✅ src.performance.tracker")
        
        from src.data import live_streamer
        print("✅ src.data.live_streamer")
        
        print("\n✅ Todas as importações OK!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Erro de importação: {e}")
        return False

def test_config():
    """Testa configuração"""
    print("\n🧪 Testando configuração...")
    
    try:
        from src.config import validate_config, TRADING_PAIRS
        
        if validate_config():
            print(f"✅ Configuração válida")
            print(f"   Pares: {', '.join(TRADING_PAIRS)}")
            return True
        else:
            print("⚠️ Configuração incompleta (esperado se .env não existe)")
            return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_database():
    """Testa conexão com banco"""
    print("\n🧪 Testando conexão com banco...")
    
    try:
        from src.database.connection import engine
        from sqlalchemy import text
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ PostgreSQL conectado")
            
            # Testar se tabelas existem
            result = conn.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result]
            
            expected = ['live_prices', 'signals', 'performance_stats']
            missing = [t for t in expected if t not in tables]
            
            if missing:
                print(f"⚠️ Tabelas faltando: {', '.join(missing)}")
                print("   Execute: python scripts/setup_database.py")
            else:
                print("✅ Todas as tabelas existem")
            
            return True
            
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        print("   Execute: docker-compose up -d postgres")
        return False

def main():
    """Executa todos os testes"""
    print("=" * 60)
    print("🧪 TradeSigns PoC - Teste Rápido")
    print("=" * 60)
    
    results = []
    
    results.append(("Importações", test_imports()))
    results.append(("Configuração", test_config()))
    results.append(("Banco de Dados", test_database()))
    
    print("\n" + "=" * 60)
    print("📊 Resultado dos Testes")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✅ Sistema pronto para uso!")
        print("\n🚀 Próximos passos:")
        print("   1. Configure o .env com credenciais OANDA")
        print("   2. Execute: python start.py")
    else:
        print("\n⚠️ Alguns testes falharam - verifique acima")

if __name__ == "__main__":
    main()
