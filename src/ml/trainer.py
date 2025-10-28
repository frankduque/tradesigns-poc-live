"""
ML Trainer - Treina modelos de Machine Learning
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class MLTrainer:
    """Treina e avalia modelos de ML para trading"""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.training_date = None
        
        logger.info(f"ML Trainer inicializado: {model_type}")
    
    def prepare_data(
        self, 
        df: pd.DataFrame,
        train_start: str = '2020-01-01',
        train_end: str = '2023-12-31',
        test_start: str = '2024-01-01',
        test_end: str = '2024-12-31'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepara dados para treino (walk-forward split)
        
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        logger.info(" Preparando dados para treino...")
        
        # Filtrar por perodo
        train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        logger.info(f"   Train: {train_start} at {train_end} = {len(df_train):,} samples")
        logger.info(f"   Test:  {test_start} at {test_end} = {len(df_test):,} samples")
        
        # Remover linhas sem label
        df_train = df_train[df_train['label'].notna()]
        df_test = df_test[df_test['label'].notna()]
        
        logger.info(f"   Aps remover NaN: Train={len(df_train):,}, Test={len(df_test):,}")
        
        # Separar features e labels
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'label', 'label_pnl', 'label_duration', 'label_outcome']]
        
        X_train = df_train[feature_cols]
        y_train = df_train['label']
        X_test = df_test[feature_cols]
        y_test = df_test['label']
        
        self.feature_names = feature_cols
        
        logger.info(f" Features: {len(feature_cols)}")
        logger.info(f"   Distribuio Train:")
        logger.info(f"      WIN:  {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
        logger.info(f"      LOSS: {(y_train == -1).sum():,} ({(y_train == -1).sum()/len(y_train)*100:.1f}%)")
        logger.info(f"      HOLD: {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ):
        """Treina modelo XGBoost"""
        logger.info(" Treinando XGBoost...")
        
        # Converter labels para 0, 1, 2 (XGBoost precisa)
        y_train_encoded = y_train.map({-1: 0, 0: 1, 1: 2})
        
        # Configurao do modelo
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            tree_method='hist'  # Mais rpido
        )
        
        # Validao (se fornecida)
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_encoded = y_val.map({-1: 0, 0: 1, 1: 2})
            eval_set = [(X_val, y_val_encoded)]
        
        # Treinar
        self.model.fit(
            X_train, 
            y_train_encoded,
            eval_set=eval_set,
            verbose=50
        )
        
        self.training_date = datetime.now()
        logger.info(" XGBoost treinado!")
    
    def train_lightgbm(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ):
        """Treina modelo LightGBM"""
        logger.info(" Treinando LightGBM...")
        
        # Converter labels
        y_train_encoded = y_train.map({-1: 0, 0: 1, 1: 2})
        
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multiclass',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Validao
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_encoded = y_val.map({-1: 0, 0: 1, 1: 2})
            eval_set = [(X_val, y_val_encoded)]
        
        self.model.fit(
            X_train, 
            y_train_encoded,
            eval_set=eval_set,
            eval_metric='multi_logloss'
        )
        
        self.training_date = datetime.now()
        logger.info(" LightGBM treinado!")
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict:
        """Avalia modelo no dataset de teste"""
        logger.info(" Avaliando modelo...")
        
        # Predies
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        
        # Converter de volta para labels originais
        y_pred = pd.Series(y_pred_encoded).map({0: -1, 1: 0, 2: 1})
        y_test_values = y_test.values
        
        # Mtricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test_values, y_pred),
            'precision': precision_score(y_test_values, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test_values, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test_values, y_pred, average='weighted', zero_division=0)
        }
        
        # Mtricas especficas para trading
        win_predictions = y_pred == 1
        actual_wins = y_test_values == 1
        
        if win_predictions.sum() > 0:
            metrics['win_precision'] = (win_predictions & actual_wins).sum() / win_predictions.sum()
        else:
            metrics['win_precision'] = 0
        
        if actual_wins.sum() > 0:
            metrics['win_recall'] = (win_predictions & actual_wins).sum() / actual_wins.sum()
        else:
            metrics['win_recall'] = 0
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_values, y_pred, labels=[-1, 0, 1])
        
        logger.info("\n" + "="*60)
        logger.info(" RESULTADOS DA AVALIAO")
        logger.info("="*60)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"\n Trading Especfico:")
        logger.info(f"WIN Precision: {metrics['win_precision']:.4f} (dos sinais BUY, quantos acertaram)")
        logger.info(f"WIN Recall:    {metrics['win_recall']:.4f} (dos trades WIN, quantos pegamos)")
        
        logger.info(f"\n Confusion Matrix:")
        logger.info(f"           LOSS  HOLD  WIN")
        logger.info(f"LOSS  [{cm[0,0]:6d} {cm[0,1]:6d} {cm[0,2]:6d}]")
        logger.info(f"HOLD  [{cm[1,0]:6d} {cm[1,1]:6d} {cm[1,2]:6d}]")
        logger.info(f"WIN   [{cm[2,0]:6d} {cm[2,1]:6d} {cm[2,2]:6d}]")
        
        # Classification Report
        logger.info(f"\n Classification Report:")
        logger.info("\n" + classification_report(
            y_test_values, 
            y_pred, 
            target_names=['LOSS', 'HOLD', 'WIN'],
            zero_division=0
        ))
        
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Retorna features mais importantes"""
        if self.model is None:
            logger.error("Modelo no treinado!")
            return None
        
        importance = self.model.feature_importances_
        
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n Top {top_n} Features:")
        for idx, row in df_importance.head(top_n).iterrows():
            logger.info(f"   {row['feature']:30s} : {row['importance']:.4f}")
        
        return df_importance
    
    def save_model(self, filepath: str = None):
        """Salva modelo treinado"""
        if filepath is None:
            filepath = f"models/trained/{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_date': self.training_date
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f" Modelo salvo: {filepath}")
        
        return filepath
    
    @staticmethod
    def load_model(filepath: str):
        """Carrega modelo treinado"""
        logger.info(f" Carregando modelo de {filepath}")
        
        model_data = joblib.load(filepath)
        
        trainer = MLTrainer(model_type=model_data['model_type'])
        trainer.model = model_data['model']
        trainer.feature_names = model_data['feature_names']
        trainer.training_date = model_data.get('training_date')
        
        logger.info(f" Modelo carregado: {model_data['model_type']}")
        logger.info(f"   Features: {len(trainer.feature_names)}")
        logger.info(f"   Treinado em: {trainer.training_date}")
        
        return trainer


def main():
    """Teste do trainer"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info(" Este  um teste. Execute scripts/train_model.py para treino completo")


if __name__ == "__main__":
    main()
