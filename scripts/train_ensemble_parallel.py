"""
ENSEMBLE TRAINING MASSIVO - Paralelização Total (GPU + Multi-CPU)

ESTRATÉGIA:
1. Hyperparameter Search paralelo (500+ combinações)
2. Walk-Forward Cross-Validation
3. Treina múltiplos modelos simultaneamente
4. Seleciona os Top 50 melhores
5. Cria ensemble com stacking

PARALELISMO:
- GPU: LightGBM/XGBoost com device='gpu'
- Multi-CPU: Ray Tune para hyperparameter search
- Distribuído: Ray para escalar em cluster (futuro)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import StackingRegressor
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Ray Tune para hyperparameter search distribuído
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("⚠️  Ray não disponível, usando joblib paralelo")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """
    Treina ensemble massivo de modelos em paralelo
    """
    
    def __init__(self, n_models=50, use_gpu=True, n_jobs=-1):
        self.n_models = n_models
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs
        self.models = []
        self.metadata = {}
        
    def generate_hyperparameter_configs(self, n_configs=500):
        """
        Gera 500+ configurações de hiperparâmetros para testar
        
        ESTRATÉGIA:
        - LightGBM: 300 configs
        - XGBoost: 200 configs
        - Variedade: depth, leaves, learning rate, etc
        """
        
        configs = []
        
        # LightGBM configs (300)
        for i in range(300):
            config = {
                'model_type': 'lightgbm',
                'params': {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': np.random.choice(['gbdt', 'dart']),
                    'num_leaves': np.random.randint(20, 200),
                    'max_depth': np.random.randint(3, 15),
                    'learning_rate': np.random.uniform(0.01, 0.3),
                    'n_estimators': np.random.randint(100, 500),
                    'min_child_samples': np.random.randint(10, 100),
                    'subsample': np.random.uniform(0.6, 1.0),
                    'colsample_bytree': np.random.uniform(0.6, 1.0),
                    'reg_alpha': np.random.uniform(0, 10),
                    'reg_lambda': np.random.uniform(0, 10),
                    'device': 'gpu' if self.use_gpu else 'cpu',
                    'verbose': -1
                }
            }
            configs.append(config)
        
        # XGBoost configs (200)
        for i in range(200):
            config = {
                'model_type': 'xgboost',
                'params': {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'booster': np.random.choice(['gbtree', 'dart']),
                    'max_depth': np.random.randint(3, 15),
                    'learning_rate': np.random.uniform(0.01, 0.3),
                    'n_estimators': np.random.randint(100, 500),
                    'min_child_weight': np.random.randint(1, 10),
                    'subsample': np.random.uniform(0.6, 1.0),
                    'colsample_bytree': np.random.uniform(0.6, 1.0),
                    'reg_alpha': np.random.uniform(0, 10),
                    'reg_lambda': np.random.uniform(0, 10),
                    'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
                    'verbosity': 0
                }
            }
            configs.append(config)
        
        logger.info(f"✓ {len(configs)} configurações geradas")
        return configs
    
    def train_single_model(self, config, X_train, y_train, X_val, y_val, target_idx=0):
        """
        Treina um modelo individual
        
        Retorna: (score, model, config)
        """
        
        try:
            if config['model_type'] == 'lightgbm':
                # MultiOutput wrapper para targets múltiplos
                model = MultiOutputRegressor(
                    lgb.LGBMRegressor(**config['params'])
                )
            else:  # xgboost
                model = MultiOutputRegressor(
                    xgb.XGBRegressor(**config['params'])
                )
            
            # Treinar
            model.fit(X_train, y_train)
            
            # Avaliar no validation set
            y_pred = model.predict(X_val)
            
            # Score: média de R² em todos os targets
            r2_scores = []
            for i in range(y_val.shape[1]):
                r2 = r2_score(y_val.iloc[:, i], y_pred[:, i])
                r2_scores.append(r2)
            
            avg_r2 = np.mean(r2_scores)
            
            return {
                'score': avg_r2,
                'model': model,
                'config': config,
                'r2_per_target': r2_scores
            }
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo: {e}")
            return None
    
    def parallel_train(self, configs, X_train, y_train, X_val, y_val):
        """
        Treina modelos em paralelo
        """
        
        logger.info(f"\n🔥 Iniciando treinamento paralelo de {len(configs)} modelos...")
        logger.info(f"   GPU: {'✓ Ativada' if self.use_gpu else '✗ Desativada'}")
        logger.info(f"   Cores: {self.n_jobs if self.n_jobs > 0 else 'Todos'}")
        
        results = []
        
        # Usar ProcessPoolExecutor para paralelizar
        with ProcessPoolExecutor(max_workers=4) as executor:  # 4 workers para não sobrecarregar GPU
            futures = []
            
            for config in configs:
                future = executor.submit(
                    self.train_single_model,
                    config, X_train, y_train, X_val, y_val
                )
                futures.append(future)
            
            # Coletar resultados com progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Treinando"):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        # Ordenar por score (R² médio)
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        logger.info(f"\n✅ {len(results)} modelos treinados com sucesso")
        logger.info(f"   Melhor R²: {results[0]['score']:.4f}")
        logger.info(f"   Pior R²: {results[-1]['score']:.4f}")
        
        return results
    
    def create_ensemble(self, top_models, X_train, y_train, X_val, y_val):
        """
        Cria ensemble com os top N modelos
        
        ESTRATÉGIA: Stacking com meta-learner
        """
        
        logger.info(f"\n🎯 Criando ensemble com top {len(top_models)} modelos...")
        
        # Preparar base estimators
        estimators = []
        for i, result in enumerate(top_models):
            estimators.append((f'model_{i}', result['model']))
        
        # Meta-learner (LightGBM simples)
        meta_learner = MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                device='gpu' if self.use_gpu else 'cpu',
                verbose=-1
            )
        )
        
        # Criar stacking ensemble
        # Nota: StackingRegressor do sklearn não suporta MultiOutput diretamente
        # Vamos fazer manualmente
        
        # 1. Gerar previsões de todos os modelos (meta-features)
        meta_train = np.hstack([
            model['model'].predict(X_train) for model in top_models
        ])
        meta_val = np.hstack([
            model['model'].predict(X_val) for model in top_models
        ])
        
        # 2. Treinar meta-learner
        meta_learner.fit(meta_train, y_train)
        
        # 3. Avaliar
        y_pred = meta_learner.predict(meta_val)
        
        r2_scores = []
        for i in range(y_val.shape[1]):
            r2 = r2_score(y_val.iloc[:, i], y_pred[:, i])
            r2_scores.append(r2)
        
        avg_r2 = np.mean(r2_scores)
        
        logger.info(f"✅ Ensemble R²: {avg_r2:.4f}")
        logger.info(f"   Melhoria vs melhor modelo: {avg_r2 - top_models[0]['score']:.4f}")
        
        ensemble = {
            'base_models': [m['model'] for m in top_models],
            'meta_learner': meta_learner,
            'score': avg_r2,
            'r2_per_target': r2_scores
        }
        
        return ensemble
    
    def save_ensemble(self, ensemble, metadata, output_dir='data/models'):
        """
        Salva ensemble e metadados
        """
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salvar ensemble
        ensemble_path = f"{output_dir}/ensemble_{timestamp}.pkl"
        joblib.dump(ensemble, ensemble_path)
        logger.info(f"💾 Ensemble salvo: {ensemble_path}")
        
        # Salvar metadados
        metadata_path = f"{output_dir}/ensemble_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Metadados salvos: {metadata_path}")
        
        return ensemble_path


def main():
    logger.info("="*70)
    logger.info("🚀 ENSEMBLE TRAINING - PARALELIZAÇÃO MASSIVA")
    logger.info("="*70)
    
    # 1. Carregar dataset
    dataset_path = "data/features/ml_dataset_ensemble.parquet"
    logger.info(f"\n📂 Carregando: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    metadata = joblib.load('data/features/ensemble_metadata.pkl')
    feature_cols = metadata['feature_names']
    target_cols = metadata['target_names']
    
    logger.info(f"✓ Samples: {len(df):,}")
    logger.info(f"✓ Features: {len(feature_cols)}")
    logger.info(f"✓ Targets: {len(target_cols)}")
    
    # 2. Separar X e y
    X = df[feature_cols]
    y = df[target_cols]
    
    # 3. Walk-Forward Split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"\n📊 Split:")
    logger.info(f"   Train: {len(X_train):,} samples")
    logger.info(f"   Val: {len(X_val):,} samples")
    
    # 4. Criar trainer
    trainer = EnsembleTrainer(
        n_models=50,
        use_gpu=True,
        n_jobs=-1
    )
    
    # 5. Gerar configs
    configs = trainer.generate_hyperparameter_configs(n_configs=500)
    
    # 6. Treinar em paralelo
    all_results = trainer.parallel_train(configs, X_train, y_train, X_val, y_val)
    
    # 7. Selecionar top 50
    top_models = all_results[:50]
    logger.info(f"\n🏆 Top 50 selecionados")
    logger.info(f"   Melhor: {top_models[0]['score']:.4f}")
    logger.info(f"   #50: {top_models[-1]['score']:.4f}")
    
    # 8. Criar ensemble
    ensemble = trainer.create_ensemble(top_models, X_train, y_train, X_val, y_val)
    
    # 9. Salvar
    final_metadata = {
        'n_models_trained': len(all_results),
        'n_models_ensemble': len(top_models),
        'ensemble_score': ensemble['score'],
        'best_single_score': top_models[0]['score'],
        'feature_names': feature_cols,
        'target_names': target_cols,
        'horizons': metadata['horizons'],
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'timestamp': datetime.now().isoformat()
    }
    
    ensemble_path = trainer.save_ensemble(ensemble, final_metadata)
    
    logger.info("\n" + "="*70)
    logger.info("✅ ENSEMBLE PRONTO PARA PRODUÇÃO!")
    logger.info("="*70)
    logger.info(f"\n📊 RESUMO:")
    logger.info(f"   Modelos treinados: {len(all_results)}")
    logger.info(f"   Modelos no ensemble: {len(top_models)}")
    logger.info(f"   R² final: {ensemble['score']:.4f}")
    logger.info(f"\n💰 Próximo passo: Backtest e geração de sinais")
    logger.info(f"   python scripts/backtest_ensemble.py --model {ensemble_path}")


if __name__ == "__main__":
    main()
