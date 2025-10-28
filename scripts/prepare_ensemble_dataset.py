"""
ENSEMBLE DATASET PREPARATION - Paralelizado com GPU + Multi-CPU

Cria dataset com:
- 150+ features (vs 61 atuais)
- 6 horizontes de previsão (5m, 15m, 30m, 1h, 4h, 1d)
- 18 targets (3 por horizonte: return, upside, downside)
- Usa RAPIDS cuDF para processamento GPU
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from concurrent.futures import ProcessPoolExecutor
import logging

# RAPIDS cuDF não funciona em GPUs antigas (Compute Capability < 7.0)
# GTX 1060 = 6.1, então sempre usa pandas (CPU)
# GPU será usada no treinamento (LightGBM/XGBoost)
GPU_AVAILABLE = False
print("ℹ️  Feature engineering em CPU (pandas)")
print("💡 GPU será usada no treinamento dos modelos!")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_advanced_features(df):
    """
    Cria 150+ features técnicas
    
    CATEGORIAS:
    1. Trend (20 features)
    2. Momentum (25 features)
    3. Volatility (20 features)
    4. Volume (15 features)
    5. Price Action (20 features)
    6. Time/Seasonal (15 features)
    7. Microstructure (10 features)
    8. Derived/Ratios (25 features)
    9. Statistical (20 features)
    """
    
    logger.info("Criando features avançadas...")
    logger.info("📊 Progresso: 0% - Iniciando feature engineering")
    
    # Feature engineering sempre em pandas (GPU não compatível com GTX 1060)
    
    # 1. TREND FEATURES (20)
    logger.info("📈 [10%] Calculando TREND features (SMAs, EMAs)...")
    for window in [5, 10, 20, 30, 50, 100, 200]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        logger.info(f"   ✓ SMA/EMA {window} completo")
    
    # Distâncias de SMAs
    logger.info("   ✓ Calculando distâncias e crossovers...")
    df['price_sma20_dist'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['price_sma50_dist'] = (df['close'] - df['sma_50']) / df['sma_50']
    df['sma20_sma50_dist'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
    
    # Crossovers
    df['sma10_cross_sma30'] = ((df['sma_10'] > df['sma_30']).astype(int).diff())
    df['sma20_cross_sma50'] = ((df['sma_20'] > df['sma_50']).astype(int).diff())
    logger.info("✅ [20%] TREND features completos!")
    
    # 2. MOMENTUM FEATURES (25)
    logger.info("🔥 [30%] Calculando MOMENTUM features (RSI, MACD, Stochastic)...")
    # RSI em múltiplos períodos
    for window in [7, 14, 21, 28]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        logger.info(f"   ✓ RSI {window} completo")
    
    # MACD em múltiplas configurações
    logger.info("   ✓ Calculando MACD multi-config...")
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        df[f'macd_{fast}_{slow}'] = macd
        df[f'macd_signal_{fast}_{slow}'] = macd_signal
        df[f'macd_hist_{fast}_{slow}'] = macd - macd_signal
    
    # Stochastic
    logger.info("   ✓ Calculando Stochastic...")
    for window in [14, 21]:
        low_min = df['low'].rolling(window).min()
        high_max = df['high'].rolling(window).max()
        df[f'stoch_k_{window}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'stoch_d_{window}'] = df[f'stoch_k_{window}'].rolling(3).mean()
    
    # ROC (Rate of Change)
    logger.info("   ✓ Calculando ROC...")
    for window in [5, 10, 20]:
        df[f'roc_{window}'] = df['close'].pct_change(window) * 100
    logger.info("✅ [45%] MOMENTUM features completos!")
    
    # 3. VOLATILITY FEATURES (20)
    logger.info("📉 [55%] Calculando VOLATILITY features (ATR, Bollinger)...")
    for window in [14, 21, 30]:
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{window}'] = tr.rolling(window).mean()
        
        # Historical Volatility
        df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
    
    # Bollinger Bands (múltiplas configurações)
    logger.info("   ✓ Calculando Bollinger Bands...")
    for window, num_std in [(20, 2), (20, 3), (50, 2)]:
        sma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        df[f'bb_upper_{window}_{num_std}'] = sma + (num_std * std)
        df[f'bb_lower_{window}_{num_std}'] = sma - (num_std * std)
        df[f'bb_width_{window}_{num_std}'] = (df[f'bb_upper_{window}_{num_std}'] - df[f'bb_lower_{window}_{num_std}']) / sma
        df[f'bb_percent_{window}_{num_std}'] = (df['close'] - df[f'bb_lower_{window}_{num_std}']) / (df[f'bb_upper_{window}_{num_std}'] - df[f'bb_lower_{window}_{num_std}'])
    logger.info("✅ [65%] VOLATILITY features completos!")
    
    # 4. VOLUME FEATURES (15)
    logger.info("📊 [70%] Calculando VOLUME features...")
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    for window in [10, 20, 30]:
        df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
        # volume_trend removido temporariamente (muito lento com polyfit)
        # Usando aproximação mais rápida: diferença entre médias
        df[f'volume_trend_{window}'] = df['volume'].rolling(window).mean().diff()
    
    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    logger.info("✅ [75%] VOLUME features completos!")
    
    # 5. PRICE ACTION FEATURES (20)
    logger.info("🕯️  [80%] Calculando PRICE ACTION features...")
    df['candle_size'] = df['high'] - df['low']
    df['candle_body'] = np.abs(df['close'] - df['open'])
    df['candle_body_pct'] = df['candle_body'] / df['candle_size']
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    
    # Padrões de consecutivos
    df['consecutive_ups'] = (df['close'] > df['close'].shift()).astype(int).rolling(5).sum()
    df['consecutive_downs'] = (df['close'] < df['close'].shift()).astype(int).rolling(5).sum()
    
    # Distância de high/low
    for window in [10, 20, 50]:
        df[f'distance_from_high_{window}'] = (df['high'].rolling(window).max() - df['close']) / df['close']
        df[f'distance_from_low_{window}'] = (df['close'] - df['low'].rolling(window).min()) / df['close']
    logger.info("✅ [85%] PRICE ACTION features completos!")
    
    # 6. TIME/SEASONAL FEATURES (15)
    logger.info("🕐 [88%] Calculando TIME/SEASONAL features...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    
    # Sessões de trading
    df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
    df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
    df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 17)).astype(int)
    logger.info("✅ [90%] TIME/SEASONAL features completos!")
    
    # 7. STATISTICAL FEATURES (20)
    logger.info("📈 [92%] Calculando STATISTICAL features...")
    for window in [10, 20, 50]:
        df[f'skew_{window}'] = df['close'].rolling(window).skew()
        df[f'kurt_{window}'] = df['close'].rolling(window).kurt()
    
    # Autocorrelação
    for lag in [1, 5, 10, 20]:
        df[f'autocorr_lag{lag}'] = df['close'].rolling(50).apply(lambda x: x.autocorr(lag=lag))
    
    logger.info(f"✅ [100%] {len(df.columns)} features criadas - COMPLETO!")
    return df


def create_multihorizon_labels(df, horizons=[5, 15, 30, 60, 240, 1440]):
    """
    Cria labels para múltiplos horizontes
    
    Para cada horizonte, calcula:
    1. expected_return: retorno médio
    2. upside_potential: máximo ganho possível
    3. downside_risk: máxima perda possível
    """
    
    logger.info(f"Criando labels para {len(horizons)} horizontes...")
    
    targets = []
    
    for horizon in horizons:
        logger.info(f"  Processando horizonte: {horizon} minutos")
        
        # 1. Expected return (close to close)
        df[f'return_{horizon}m'] = (df['close'].shift(-horizon) - df['close']) / df['close'] * 100
        
        # 2. Upside potential (máximo ganho no período)
        df[f'upside_{horizon}m'] = (df['high'].rolling(horizon).max().shift(-horizon) - df['close']) / df['close'] * 100
        
        # 3. Downside risk (máxima perda no período)
        df[f'downside_{horizon}m'] = (df['low'].rolling(horizon).min().shift(-horizon) - df['close']) / df['close'] * 100
        
        targets.extend([f'return_{horizon}m', f'upside_{horizon}m', f'downside_{horizon}m'])
    
    logger.info(f"✅ {len(targets)} targets criados")
    return df, targets


def main():
    logger.info("="*70)
    logger.info("🚀 ENSEMBLE DATASET PREPARATION")
    logger.info("="*70)
    
    # 1. Carregar dados
    dataset_path = "data/processed/eurusd_m1_full.parquet"
    logger.info(f"\n📂 Carregando: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df = df.set_index('timestamp')
    logger.info(f"✓ {len(df):,} candles carregados")
    
    # 2. Features avançadas
    df = create_advanced_features(df)
    
    # 3. Labels multi-horizonte
    horizons = [5, 15, 30, 60, 240, 1440]  # 5m, 15m, 30m, 1h, 4h, 1 dia
    df, target_names = create_multihorizon_labels(df, horizons)
    
    # 4. Limpar NaNs
    df = df.dropna()
    logger.info(f"\n✓ Dataset final: {len(df):,} samples")
    
    # 5. Separar features e targets
    feature_cols = [col for col in df.columns if not col.startswith(('return_', 'upside_', 'downside_'))]
    X = df[feature_cols]
    y = df[target_names]
    
    logger.info(f"✓ Features: {len(feature_cols)}")
    logger.info(f"✓ Targets: {len(target_names)}")
    
    # 6. Salvar
    output_path = "data/features/ml_dataset_ensemble.parquet"
    df.to_parquet(output_path, compression='snappy')
    logger.info(f"\n💾 Salvo em: {output_path}")
    
    # Salvar metadados
    metadata = {
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'n_targets': len(target_names),
        'feature_names': feature_cols,
        'target_names': target_names,
        'horizons': horizons
    }
    joblib.dump(metadata, 'data/features/ensemble_metadata.pkl')
    
    logger.info("\n✅ DATASET ENSEMBLE PRONTO!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
