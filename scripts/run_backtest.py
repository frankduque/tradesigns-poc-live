"""
Script de Backtest
Testa modelo treinado em dados histricos
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import logging
import pandas as pd
import numpy as np
from datetime import datetime

from src.ml.trainer import MLTrainer
from src.backtest.engine import BacktestEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/backtest.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Executa backtest com modelo treinado"""
    
    logger.info("=" * 70)
    logger.info(" BACKTEST - Validao de Modelo")
    logger.info("=" * 70)
    
    # 1. Carregar modelo
    logger.info("\n Carregando modelo treinado...")
    
    models_dir = Path("models/trained")
    model_files = list(models_dir.glob("*.pkl"))
    
    if not model_files:
        logger.error(f" Nenhum modelo encontrado em {models_dir}")
        logger.error("   Execute: python scripts/train_model.py")
        return False
    
    # Usar o modelo mais recente
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f" Usando modelo: {latest_model.name}")
    
    trainer = MLTrainer.load_model(latest_model)
    
    # 2. Carregar dataset
    logger.info("\n Carregando dataset...")
    
    dataset_file = Path("data/features/ml_dataset_full.parquet")
    df = pd.read_parquet(dataset_file)
    
    logger.info(f" Dataset carregado: {len(df):,} samples")
    
    # 3. Escolher perodo de backtest
    logger.info("\n Escolhendo perodo de backtest...")
    
    print(f"\n Dados disponveis:")
    print(f"   Incio: {df['timestamp'].min()}")
    print(f"   Fim: {df['timestamp'].max()}")
    
    backtest_start = input(f"\n Data de incio do backtest [2024-01-01]: ").strip()
    if not backtest_start:
        backtest_start = '2024-01-01'
    
    backtest_end = input(f" Data de fim do backtest [2024-12-31]: ").strip()
    if not backtest_end:
        backtest_end = '2024-12-31'
    
    # Filtrar perodo
    mask = (df['timestamp'] >= backtest_start) & (df['timestamp'] <= backtest_end)
    df_backtest = df[mask].copy()
    
    logger.info(f"\n Perodo de backtest:")
    logger.info(f"   {backtest_start} at {backtest_end}")
    logger.info(f"   {len(df_backtest):,} candles")
    
    # 4. Gerar predies
    logger.info("\n Gerando predies do modelo...")
    
    feature_cols = trainer.feature_names
    X = df_backtest[feature_cols]
    
    # Predizer
    y_pred_proba = trainer.model.predict_proba(X)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    
    # Converter de volta para labels originais
    y_pred = pd.Series(y_pred_encoded).map({0: -1, 1: 0, 2: 1})
    
    # Confiana da predio
    confidence = np.max(y_pred_proba, axis=1)
    
    # Filtrar por confiana mnima
    min_confidence = 0.5
    logger.info(f"   Filtrando predies com confiana < {min_confidence}")
    
    y_pred_filtered = y_pred.copy()
    y_pred_filtered[confidence < min_confidence] = 0  # HOLD se confiana baixa
    
    logger.info(f"   Sinais gerados:")
    logger.info(f"      BUY (1):  {(y_pred_filtered == 1).sum():,}")
    logger.info(f"      SELL (-1): {(y_pred_filtered == -1).sum():,}")
    logger.info(f"      HOLD (0): {(y_pred_filtered == 0).sum():,}")
    
    # 5. Executar backtest
    logger.info("\n Executando backtest...")
    
    engine = BacktestEngine(
        initial_capital=10000,
        take_profit_pct=0.004,
        stop_loss_pct=0.002,
        max_duration_minutes=60,
        fee_pct=0.0002
    )
    
    metrics = engine.run(df_backtest, y_pred_filtered)
    
    # 6. Exibir resultados
    engine.print_summary(metrics)
    
    # 7. Salvar resultados
    logger.info("\n Salvando resultados...")
    
    # Salvar trades
    trades_df = engine.get_trades_df()
    trades_file = Path(f"models/metadata/backtest_trades_{datetime.now().strftime('%Y%m%d')}.csv")
    trades_df.to_csv(trades_file, index=False)
    logger.info(f"   Trades salvos: {trades_file}")
    
    # Salvar equity curve
    equity_df = pd.DataFrame({
        'step': range(len(engine.equity_curve)),
        'equity': engine.equity_curve
    })
    equity_file = Path(f"models/metadata/backtest_equity_{datetime.now().strftime('%Y%m%d')}.csv")
    equity_df.to_csv(equity_file, index=False)
    logger.info(f"   Equity curve salva: {equity_file}")
    
    # Salvar mtricas
    import json
    metrics_file = Path(f"models/metadata/backtest_metrics_{datetime.now().strftime('%Y%m%d')}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"   Mtricas salvas: {metrics_file}")
    
    # 8. Anlise final
    logger.info("\n" + "=" * 70)
    logger.info(" ANLISE FINAL")
    logger.info("=" * 70)
    
    if metrics['win_rate'] > 0.55:
        logger.info(" WIN RATE EXCELENTE (>55%)!")
    elif metrics['win_rate'] > 0.50:
        logger.info(" Win rate OK, mas pode melhorar")
    else:
        logger.info(" Win rate BAIXO - modelo precisa melhorar")
    
    if metrics['sharpe_ratio'] > 1.0:
        logger.info(" SHARPE RATIO EXCELENTE (>1.0)!")
    elif metrics['sharpe_ratio'] > 0.5:
        logger.info(" Sharpe OK")
    else:
        logger.info(" Sharpe BAIXO - muita volatilidade")
    
    if abs(metrics['max_drawdown']) < 0.10:
        logger.info(" DRAWDOWN CONTROLADO (<10%)!")
    elif abs(metrics['max_drawdown']) < 0.20:
        logger.info(" Drawdown aceitvel")
    else:
        logger.info(" Drawdown ALTO - risco elevado")
    
    logger.info("\n Prximos passos:")
    if metrics['win_rate'] > 0.52 and metrics['sharpe_ratio'] > 0.5:
        logger.info("    Modelo est bom! Pode integrar ao sistema live:")
        logger.info("    Editar src/signals/generator.py")
        logger.info("    Carregar modelo treinado")
        logger.info("    Usar predies em produo")
    else:
        logger.info("    Modelo precisa melhorar:")
        logger.info("    Ajustar hiperparmetros")
        logger.info("    Adicionar mais features")
        logger.info("    Testar outras estratgias de labeling")
    
    return True


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)
