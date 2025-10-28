"""
Multi-Target Regression Trainer

Treina modelo para prever múltiplos horizontes simultaneamente.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MultiTargetRegressionTrainer:
    """
    Treina modelo de regressão para prever múltiplos horizontes:
    - return_5m, return_10m, return_30m
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 15,
        min_samples_split: int = 100,
        min_samples_leaf: int = 50,
        random_state: int = 42
    ):
        """
        Args:
            n_estimators: Número de árvores
            max_depth: Profundidade máxima
            min_samples_split: Mínimo de samples para split
            min_samples_leaf: Mínimo de samples por folha
            random_state: Seed para reprodutibilidade
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'n_jobs': -1
        }
        
        # Modelo para múltiplos targets
        self.model = MultiOutputRegressor(
            RandomForestRegressor(**self.params)
        )
        
        self.feature_names = None
        self.target_names = None
        self.trained = False
        
        logger.info("Multi-Target Regression Trainer inicializado")
        logger.info(f"   Params: {self.params}")
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        target_names: list,
        test_size: float = 0.2,
        validation: bool = True
    ) -> Dict:
        """
        Treina modelo
        
        Args:
            X: Features (N x 64)
            y: Targets (N x 3) [return_5m, return_10m, return_30m]
            feature_names: Nomes das features
            target_names: Nomes dos targets
            test_size: Proporção de dados para teste
            validation: Se True, faz cross-validation
        
        Returns:
            Dict com métricas de treinamento
        """
        logger.info("\nIniciando treinamento...")
        logger.info(f"   Samples: {X.shape[0]:,}")
        logger.info(f"   Features: {X.shape[1]}")
        logger.info(f"   Targets: {y.shape[1]}")
        
        self.feature_names = feature_names
        self.target_names = target_names
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        logger.info(f"\n   Train: {X_train.shape[0]:,} samples")
        logger.info(f"   Test:  {X_test.shape[0]:,} samples")
        
        # Treinar
        logger.info("\n Treinando modelo...")
        logger.info("   (Isso pode levar 30-60 minutos sem progresso visível...)")
        logger.info("   Dica: Use 'verbose=2' para ver progresso por árvore\n")
        
        start = time.time()
        
        # Configurar verbose para mostrar progresso
        self.model.verbose = 2  # Mostra progresso de cada árvore
        
        self.model.fit(X_train, y_train)
        train_time = time.time() - start
        logger.info(f"\n Treinamento concluído em {train_time/60:.1f} minutos")
        
        self.trained = True
        
        # Avaliar
        metrics = self._evaluate(X_train, y_train, X_test, y_test)
        metrics['train_time'] = train_time
        
        # Cross-validation (opcional)
        if validation:
            cv_scores = self._cross_validate(X_train, y_train)
            metrics['cv_scores'] = cv_scores
        
        # Feature importance
        importances = self._get_feature_importance()
        metrics['feature_importance'] = importances
        
        return metrics
    
    def _evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Avalia modelo em train e test"""
        logger.info("\n Avaliando modelo...")
        
        # Predições
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        metrics = {'train': {}, 'test': {}}
        
        # Métricas para cada target
        for i, target_name in enumerate(self.target_names):
            # Train
            train_mse = mean_squared_error(y_train[:, i], y_train_pred[:, i])
            train_mae = mean_absolute_error(y_train[:, i], y_train_pred[:, i])
            train_r2 = r2_score(y_train[:, i], y_train_pred[:, i])
            
            # Test
            test_mse = mean_squared_error(y_test[:, i], y_test_pred[:, i])
            test_mae = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
            test_r2 = r2_score(y_test[:, i], y_test_pred[:, i])
            
            metrics['train'][target_name] = {
                'mse': train_mse,
                'mae': train_mae,
                'rmse': np.sqrt(train_mse),
                'r2': train_r2
            }
            
            metrics['test'][target_name] = {
                'mse': test_mse,
                'mae': test_mae,
                'rmse': np.sqrt(test_mse),
                'r2': test_r2
            }
            
            logger.info(f"\n{target_name}:")
            logger.info(f"   Train - RMSE: {np.sqrt(train_mse):.4f}% | MAE: {train_mae:.4f}% | R2: {train_r2:.3f}")
            logger.info(f"   Test  - RMSE: {np.sqrt(test_mse):.4f}% | MAE: {test_mae:.4f}% | R2: {test_r2:.3f}")
        
        return metrics
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 3) -> Dict:
        """Cross-validation"""
        logger.info(f"\n Executando {cv}-fold cross-validation...")
        
        cv_scores = {}
        
        for i, target_name in enumerate(self.target_names):
            # Score para cada target individualmente
            scores = cross_val_score(
                self.model.estimators_[i],
                X,
                y[:, i],
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            rmse_scores = np.sqrt(-scores)
            
            cv_scores[target_name] = {
                'rmse_mean': rmse_scores.mean(),
                'rmse_std': rmse_scores.std()
            }
            
            logger.info(f"   {target_name}: RMSE = {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
        
        return cv_scores
    
    def _get_feature_importance(self) -> pd.DataFrame:
        """Calcula feature importance média entre todos os targets"""
        logger.info("\n Calculando feature importance...")
        
        # Importance média entre os 3 estimators
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        
        avg_importance = np.mean(importances, axis=0)
        
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Features:")
        for idx, row in df_importance.head(10).iterrows():
            logger.info(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        return df_importance
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz previsões
        
        Args:
            X: Features (N x 64)
        
        Returns:
            Predictions (N x 3) [return_5m, return_10m, return_30m]
        """
        if not self.trained:
            raise ValueError("Modelo não treinado! Execute .train() primeiro.")
        
        return self.model.predict(X)
    
    def save(self, output_path: Path):
        """Salva modelo treinado"""
        if not self.trained:
            raise ValueError("Modelo não treinado!")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar modelo e metadados
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'params': self.params
        }
        
        joblib.dump(model_data, output_path)
        
        file_size = output_path.stat().st_size / (1024 ** 2)
        logger.info(f"\nModelo salvo:")
        logger.info(f"   Path: {output_path}")
        logger.info(f"   Size: {file_size:.2f} MB")
    
    @classmethod
    def load(cls, model_path: Path):
        """Carrega modelo treinado"""
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        model_data = joblib.load(model_path)
        
        trainer = cls(**model_data['params'])
        trainer.model = model_data['model']
        trainer.feature_names = model_data['feature_names']
        trainer.target_names = model_data['target_names']
        trainer.trained = True
        
        logger.info(f"Modelo carregado: {model_path}")
        
        return trainer


if __name__ == "__main__":
    print("\nMulti-Target Regression Trainer ready!")
    print("Use train_model_regression.py para treinar.")
