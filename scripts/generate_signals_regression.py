"""
Script: Gerar Sinais e Backtest com Multi-Target Regression

Carrega modelo treinado, gera sinais e executa backtest.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
import pandas as pd
import numpy as np
from src.ml.trainer_regression import MultiTargetRegressionTrainer
from src.ml.signal_generator_regression import RegressionSignalGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/generate_signals_regression.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_model_and_data() -> tuple:
    """Carrega modelo treinado e dataset"""
    logger.info("\nCarregando modelo e dados...")
    
    # Carregar modelo
    model_path = Path("models/regression_model.joblib")
    if not model_path.exists():
        logger.error(f"Modelo não encontrado: {model_path}")
        logger.error("Execute primeiro: python scripts/train_model_regression.py")
        return None, None, None
    
    trainer = MultiTargetRegressionTrainer.load(model_path)
    logger.info(" Modelo carregado!")
    
    # Carregar dataset
    dataset_path = Path("data/features/ml_dataset_regression.parquet")
    df_full = pd.read_parquet(dataset_path)
    logger.info(f" Dataset carregado: {len(df_full):,} samples")
    
    # Separar features
    target_cols = ['return_5m', 'return_10m', 'return_30m']
    exclude_cols = target_cols + [
        'open', 'high', 'low', 'close', 'volume',
        'max_return_5m', 'max_return_10m', 'max_return_30m',
        'min_return_5m', 'min_return_10m', 'min_return_30m'
    ]
    feature_cols = [col for col in df_full.columns if col not in exclude_cols]
    
    # Remover NaNs
    df_clean = df_full.dropna(subset=feature_cols)
    X = df_clean[feature_cols].values
    
    logger.info(f"   Features shape: {X.shape}")
    
    return trainer, df_clean, X


def main():
    """Gerar sinais e backtest"""
    try:
        logger.info("="*70)
        logger.info("GERACAO DE SINAIS - MULTI-TARGET REGRESSION")
        logger.info("="*70)
        
        # ===== PASSO 1: Carregar modelo e dados =====
        trainer, df, X = load_model_and_data()
        
        if trainer is None:
            return False
        
        # ===== PASSO 2: Gerar sinais =====
        logger.info("\nPASSO 2: Gerando sinais...")
        
        signal_generator = RegressionSignalGenerator(
            model=trainer,
            base_threshold=0.05,  # 5 pips
            atr_multiplier=2.0,
            use_confidence=True
        )
        
        # Testar ambos os métodos
        logger.info("\n--- METODO SIMPLES ---")
        signals_simple, predictions = signal_generator.generate_signals(
            df=df,
            features=X,
            method='simple'
        )
        
        logger.info("\n--- METODO AVANCADO ---")
        signals_advanced, _ = signal_generator.generate_signals(
            df=df,
            features=X,
            method='advanced'
        )
        
        # ===== PASSO 3: Backtest =====
        logger.info("\n" + "="*70)
        logger.info("PASSO 3: Executando Backtest")
        logger.info("="*70)
        
        logger.info("\n>>> BACKTEST - METODO SIMPLES <<<")
        trades_simple = signal_generator.backtest_signals(
            df=df,
            signals=signals_simple,
            predictions_df=predictions,
            tp_multiplier=1.0,
            sl_multiplier=0.5,
            max_duration=30
        )
        
        logger.info("\n>>> BACKTEST - METODO AVANCADO <<<")
        trades_advanced = signal_generator.backtest_signals(
            df=df,
            signals=signals_advanced,
            predictions_df=predictions,
            tp_multiplier=1.0,
            sl_multiplier=0.5,
            max_duration=30
        )
        
        # ===== PASSO 4: Salvar resultados =====
        logger.info("\nPASSO 4: Salvando resultados...")
        
        output_dir = Path("data/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar trades
        trades_simple.to_csv(output_dir / "backtest_simple.csv", index=False)
        trades_advanced.to_csv(output_dir / "backtest_advanced.csv", index=False)
        
        # Salvar sinais e previsões
        results_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        results_df['signal_simple'] = signals_simple
        results_df['signal_advanced'] = signals_advanced
        results_df = pd.concat([results_df, predictions], axis=1)
        results_df.to_parquet(output_dir / "signals_and_predictions.parquet")
        
        logger.info(f" Resultados salvos em: {output_dir}")
        
        # ===== RESUMO FINAL =====
        logger.info("\n" + "="*70)
        logger.info("RESUMO FINAL")
        logger.info("="*70)
        
        logger.info("\nMETODO SIMPLES:")
        logger.info(f"   Sinais gerados: {(signals_simple != 0).sum():,}")
        logger.info(f"   Trades executados: {len(trades_simple):,}")
        if len(trades_simple) > 0:
            win_rate = (trades_simple['outcome'] == 'WIN').sum() / len(trades_simple) * 100
            logger.info(f"   Win Rate: {win_rate:.1f}%")
            logger.info(f"   Avg P&L: {trades_simple['pnl_pct'].mean():.4f}%")
        
        logger.info("\nMETODO AVANCADO:")
        logger.info(f"   Sinais gerados: {(signals_advanced != 0).sum():,}")
        logger.info(f"   Trades executados: {len(trades_advanced):,}")
        if len(trades_advanced) > 0:
            win_rate = (trades_advanced['outcome'] == 'WIN').sum() / len(trades_advanced) * 100
            logger.info(f"   Win Rate: {win_rate:.1f}%")
            logger.info(f"   Avg P&L: {trades_advanced['pnl_pct'].mean():.4f}%")
        
        logger.info("\nProximo passo:")
        logger.info("   Analisar resultados em data/results/")
        logger.info("   Ajustar thresholds se necessário")
        logger.info("   python scripts/live_trading_sim.py (para testar ao vivo)")
        
        return True
    
    except Exception as e:
        logger.error(f"ERRO na geração de sinais: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
