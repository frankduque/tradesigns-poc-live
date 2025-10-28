"""
GRID SEARCH - Testa múltiplas combinações de TP/SL
Encontra automaticamente os melhores parâmetros!
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import logging
from src.ml.data_loader import HistDataLoader
from src.ml.feature_engineer import FeatureEngineer
from src.ml.label_creator import LabelCreator
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_label_parameters():
    """
    Testa múltiplas combinações de TP/SL/Duration
    Mostra qual tem melhor balanço WIN/LOSS/HOLD
    """
    
    logger.info("="*70)
    logger.info("GRID SEARCH - OTIMIZACAO DE PARAMETROS DE LABEL")
    logger.info("="*70)
    
    # 1. Carregar e preparar dados (fazemos isso UMA vez)
    logger.info("\nCarregando dados...")
    loader = HistDataLoader()
    df = loader.load_processed()
    logger.info(f"OK - {len(df):,} candles carregados")
    
    logger.info("\nCriando features...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    logger.info(f"OK - {len(df_features):,} linhas com features")
    
    # Para acelerar, usar apenas uma amostra (10% dos dados)
    sample_size = len(df_features) // 10
    logger.info(f"\nUsando amostra de {sample_size:,} linhas para velocidade")
    df_sample = df_features.head(sample_size).copy()
    
    # 2. Definir grid de parametros para testar
    param_grid = [
        # Scalping extremo (5-8 pips)
        {'tp': 0.0005, 'sl': 0.0005, 'duration': 10, 'name': 'Scalping Ultra (5pip/10min)'},
        {'tp': 0.0008, 'sl': 0.0008, 'duration': 15, 'name': 'Scalping Agressivo (8pip/15min)'},
        
        # Scalping moderado (10 pips)
        {'tp': 0.0010, 'sl': 0.0008, 'duration': 15, 'name': 'Scalping RR 1.25 (10:8/15min)'},
        {'tp': 0.0010, 'sl': 0.0010, 'duration': 20, 'name': 'Scalping Balanced (10pip/20min)'},
        
        # Intraday (15 pips)
        {'tp': 0.0015, 'sl': 0.0010, 'duration': 30, 'name': 'Intraday RR 1.5 (15:10/30min)'},
        {'tp': 0.0015, 'sl': 0.0015, 'duration': 30, 'name': 'Intraday Balanced (15pip/30min)'},
        
        # Day Trading (20-30 pips)
        {'tp': 0.0020, 'sl': 0.0015, 'duration': 45, 'name': 'DayTrade RR 1.33 (20:15/45min)'},
        {'tp': 0.0030, 'sl': 0.0020, 'duration': 60, 'name': 'DayTrade RR 1.5 (30:20/60min)'},
        
        # Swing (40+ pips)
        {'tp': 0.0040, 'sl': 0.0020, 'duration': 90, 'name': 'Swing RR 2.0 (40:20/90min)'},
        {'tp': 0.0050, 'sl': 0.0025, 'duration': 120, 'name': 'Swing RR 2.0 (50:25/120min)'},
    ]
    
    # 3. Testar cada combinacao
    results = []
    
    logger.info("\n" + "="*70)
    logger.info("TESTANDO COMBINACOES...")
    logger.info("="*70 + "\n")
    
    for i, params in enumerate(param_grid, 1):
        logger.info(f"[{i}/{len(param_grid)}] Testando: {params['name']}")
        
        start = time.time()
        
        # Criar label creator com parametros
        label_creator = LabelCreator(
            take_profit_pct=params['tp'],
            stop_loss_pct=params['sl'],
            max_duration_candles=params['duration'],
            fee_pct=0.0001
        )
        
        # Criar labels
        df_labeled = label_creator.create_realistic_labels(df_sample.copy())
        
        # Calcular estatisticas
        label_counts = df_labeled['label'].value_counts()
        total = len(df_labeled)
        
        win_pct = (label_counts.get(1, 0) / total * 100) if total > 0 else 0
        loss_pct = (label_counts.get(-1, 0) / total * 100) if total > 0 else 0
        hold_pct = (label_counts.get(0, 0) / total * 100) if total > 0 else 0
        
        # Calcular score (queremos WIN e LOSS proximos, HOLD baixo)
        action_pct = win_pct + loss_pct  # Quanto mais acao, melhor
        balance_score = 100 - abs(win_pct - loss_pct)  # Quanto mais balanceado, melhor
        overall_score = (action_pct * 0.6) + (balance_score * 0.4)  # Score final
        
        elapsed = time.time() - start
        
        result = {
            'name': params['name'],
            'tp_pips': int(params['tp'] * 10000),
            'sl_pips': int(params['sl'] * 10000),
            'duration': params['duration'],
            'win_pct': win_pct,
            'loss_pct': loss_pct,
            'hold_pct': hold_pct,
            'action_pct': action_pct,
            'balance_score': balance_score,
            'overall_score': overall_score,
            'time_sec': elapsed
        }
        
        results.append(result)
        
        logger.info(f"  WIN: {win_pct:.1f}% | LOSS: {loss_pct:.1f}% | HOLD: {hold_pct:.1f}%")
        logger.info(f"  Action: {action_pct:.1f}% | Balance: {balance_score:.1f} | Score: {overall_score:.1f}")
        logger.info(f"  Tempo: {elapsed:.1f}s\n")
    
    # 4. Criar DataFrame com resultados
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('overall_score', ascending=False)
    
    # 5. Mostrar resultados
    logger.info("\n" + "="*70)
    logger.info("RESULTADOS - ORDENADOS POR SCORE")
    logger.info("="*70 + "\n")
    
    print("\n" + "="*120)
    print(f"{'RANK':<6} {'NOME':<35} {'TP':<5} {'SL':<5} {'DUR':<5} {'WIN%':<7} {'LOSS%':<7} {'HOLD%':<7} {'SCORE':<7}")
    print("="*120)
    
    for idx, row in df_results.iterrows():
        rank = list(df_results.index).index(idx) + 1
        print(f"{rank:<6} {row['name']:<35} {row['tp_pips']:<5} {row['sl_pips']:<5} {row['duration']:<5} "
              f"{row['win_pct']:<7.1f} {row['loss_pct']:<7.1f} {row['hold_pct']:<7.1f} {row['overall_score']:<7.1f}")
    
    print("="*120)
    
    # 6. Recomendar top 3
    top3 = df_results.head(3)
    
    logger.info("\n" + "="*70)
    logger.info("TOP 3 RECOMENDACOES")
    logger.info("="*70)
    
    for i, (idx, row) in enumerate(top3.iterrows(), 1):
        logger.info(f"\n{i}. {row['name']}")
        logger.info(f"   TP: {row['tp_pips']} pips | SL: {row['sl_pips']} pips | Duration: {row['duration']} min")
        logger.info(f"   WIN: {row['win_pct']:.1f}% | LOSS: {row['loss_pct']:.1f}% | HOLD: {row['hold_pct']:.1f}%")
        logger.info(f"   Score: {row['overall_score']:.1f}/100")
        
        if i == 1:
            logger.info(f"   >>> MELHOR OPCAO! <<<")
    
    # 7. Salvar resultados
    output_file = Path("data/features/label_params_optimization.csv")
    df_results.to_csv(output_file, index=False)
    logger.info(f"\nResultados salvos em: {output_file}")
    
    # 8. Gerar codigo para o melhor
    best = df_results.iloc[0]
    logger.info("\n" + "="*70)
    logger.info("CODIGO PARA USAR O MELHOR PARAMETRO:")
    logger.info("="*70)
    logger.info(f"""
label_creator = LabelCreator(
    take_profit_pct={best['tp_pips']/10000},   # {best['tp_pips']} pips
    stop_loss_pct={best['sl_pips']/10000},     # {best['sl_pips']} pips
    max_duration_candles={int(best['duration'])},  # {int(best['duration'])} minutos
    fee_pct=0.0001
)
""")
    
    logger.info("\n" + "="*70)
    logger.info("OTIMIZACAO COMPLETA!")
    logger.info("="*70)
    
    return df_results


if __name__ == "__main__":
    results = test_label_parameters()
