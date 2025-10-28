"""
Script para criar as tabelas do banco de dados
"""
import os
import sys
from pathlib import Path

# Adiciona o diretrio raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def setup_database():
    """Cria as tabelas no banco de dados"""
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print(" DATABASE_URL no encontrada no .env")
        sys.exit(1)
    
    print(" Conectando ao PostgreSQL...")
    engine = create_engine(database_url)
    
    # Ler arquivo SQL
    sql_file = ROOT_DIR / 'scripts' / 'init_db.sql'
    
    if not sql_file.exists():
        print(f" Arquivo SQL no encontrado: {sql_file}")
        sys.exit(1)
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_commands = f.read()
    
    try:
        with engine.connect() as conn:
            # Executar comandos SQL
            for statement in sql_commands.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
                    conn.commit()
        
        print(" Database schema criado com sucesso!")
        print(" Tabelas criadas:")
        print("   - live_prices (hypertable)")
        print("   - signals")
        print("   - performance_stats")
        print("   - stats_24h (view)")
        
    except Exception as e:
        print(f" Erro ao criar schema: {e}")
        sys.exit(1)
    finally:
        engine.dispose()

if __name__ == "__main__":
    setup_database()
