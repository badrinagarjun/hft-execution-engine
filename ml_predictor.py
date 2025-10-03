"""
ML predictor for short-term price direction using order-flow features.
Trains logistic regression or simple MLP to predict next tick movement.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle
import pathlib


class OrderFlowPredictor:
    """Predicts short-term price direction from order flow features."""
    
    def __init__(self, model_type: str = 'logistic', 
                 horizon: int = 1,
                 threshold: float = 0.0):
        """
        Initialize predictor.
        
        Args:
            model_type: 'logistic' or 'mlp'
            horizon: Prediction horizon (ticks ahead)
            threshold: Minimum price change for signal (in price units)
        """
        self.model_type = model_type
        self.horizon = horizon
        self.threshold = threshold
        self.scaler = StandardScaler()
        
        # Initialize model
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_features(self, feature_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training.
        
        Args:
            feature_df: DataFrame with features
            
        Returns:
            (X, y) arrays
        """
        # Select numeric features only
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['timestamp', 'midprice']  # Don't use these directly as features
        
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Store feature columns for later
        if self.feature_columns is None:
            self.feature_columns = feature_cols
        
        X = feature_df[feature_cols].fillna(0).values
        
        # Create labels based on future price movement
        future_midprice = feature_df['midprice'].shift(-self.horizon)
        current_midprice = feature_df['midprice']
        price_change = future_midprice - current_midprice
        
        # Binary classification: 1 = up, 0 = down/neutral
        y = (price_change > self.threshold).astype(int)
        
        # Remove NaN rows
        valid_idx = ~(feature_df['midprice'].isna() | future_midprice.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        return X, y
    
    def train(self, feature_df: pd.DataFrame, 
             test_size: float = 0.2,
             verbose: bool = True) -> Dict:
        """
        Train predictor on feature data.
        
        Args:
            feature_df: DataFrame with features
            test_size: Fraction of data for testing
            verbose: Print training results
            
        Returns:
            Dict with training metrics
        """
        # Prepare data
        X, y = self.prepare_features(feature_df)
        
        if len(X) == 0:
            raise ValueError("No valid training samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = np.nan
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'class_distribution': np.bincount(y)
        }
        
        if verbose:
            print(f"Training Results:")
            print(f"  Model: {self.model_type}")
            print(f"  Samples: {len(X)} (train: {len(X_train)}, test: {len(X_test)})")
            print(f"  Features: {X.shape[1]}")
            print(f"  Train accuracy: {train_score:.4f}")
            print(f"  Test accuracy: {test_score:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")
            print(f"\n  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
            print(f"\n  Confusion Matrix:")
            print(f"  {conf_matrix}")
        
        return results
    
    def predict_proba(self, features: Dict) -> float:
        """
        Predict probability of price going up.
        
        Args:
            features: Dict of feature values
            
        Returns:
            Probability of upward movement [0, 1]
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Extract feature values in correct order
        feature_values = []
        for col in self.feature_columns:
            value = features.get(col, 0)
            # Handle NaN
            if pd.isna(value):
                value = 0
            feature_values.append(value)
        
        # Reshape and scale
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict probability
        proba = self.model.predict_proba(X_scaled)[0, 1]
        return proba
    
    def predict(self, features: Dict) -> int:
        """
        Predict class (0 = down, 1 = up).
        
        Args:
            features: Dict of feature values
            
        Returns:
            Predicted class
        """
        proba = self.predict_proba(features)
        return 1 if proba > 0.5 else 0
    
    def save(self, filepath: str):
        """Save model to disk."""
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'horizon': self.horizon,
            'threshold': self.threshold,
            'is_trained': self.is_trained
        }
        pathlib.Path(filepath).write_bytes(pickle.dumps(data))
    
    @classmethod
    def load(cls, filepath: str) -> 'OrderFlowPredictor':
        """Load model from disk."""
        data = pickle.loads(pathlib.Path(filepath).read_bytes())
        
        predictor = cls(
            model_type=data['model_type'],
            horizon=data['horizon'],
            threshold=data['threshold']
        )
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_columns = data['feature_columns']
        predictor.is_trained = data['is_trained']
        
        return predictor


def test_predictor():
    """Test ML predictor with synthetic data."""
    print("=" * 60)
    print("Testing ML Predictor")
    print("=" * 60)
    
    # Generate synthetic feature data
    print("\n1. Generating synthetic feature data...")
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic-looking features
    data = {
        'timestamp': np.arange(n_samples),
        'midprice': 50000 + np.cumsum(np.random.randn(n_samples) * 10),
        'spread': np.abs(np.random.randn(n_samples) * 2 + 5),
        'imbalance_1': np.random.randn(n_samples) * 0.3,
        'imbalance_5': np.random.randn(n_samples) * 0.2,
        'imbalance_10': np.random.randn(n_samples) * 0.15,
        'bid_depth_5': np.abs(np.random.randn(n_samples) * 5 + 20),
        'ask_depth_5': np.abs(np.random.randn(n_samples) * 5 + 20),
        'trade_imbalance': np.random.randn(n_samples) * 0.4,
        'midprice_change': np.random.randn(n_samples) * 5,
        'spread_change': np.random.randn(n_samples) * 0.5
    }
    
    # Add some correlation: positive imbalance → price increase
    for i in range(1, n_samples):
        if data['imbalance_5'][i-1] > 0.2:
            data['midprice'][i] += 2  # Slight upward bias
        elif data['imbalance_5'][i-1] < -0.2:
            data['midprice'][i] -= 2  # Slight downward bias
    
    feature_df = pd.DataFrame(data)
    print(f"   Generated {len(feature_df)} samples")
    print(f"   Features: {list(feature_df.columns)}")
    
    # Test Logistic Regression predictor
    print("\n2. Training Logistic Regression predictor...")
    predictor_lr = OrderFlowPredictor(model_type='logistic', horizon=1, threshold=0.5)
    results_lr = predictor_lr.train(feature_df, test_size=0.2, verbose=True)
    
    # Test MLP predictor
    print("\n3. Training MLP predictor...")
    predictor_mlp = OrderFlowPredictor(model_type='mlp', horizon=1, threshold=0.5)
    results_mlp = predictor_mlp.train(feature_df, test_size=0.2, verbose=True)
    
    # Test prediction
    print("\n4. Testing prediction on single sample...")
    test_features = feature_df.iloc[100].to_dict()
    
    proba_lr = predictor_lr.predict_proba(test_features)
    pred_lr = predictor_lr.predict(test_features)
    
    proba_mlp = predictor_mlp.predict_proba(test_features)
    pred_mlp = predictor_mlp.predict(test_features)
    
    print(f"   Sample features:")
    print(f"     Imbalance: {test_features['imbalance_5']:.4f}")
    print(f"     Spread: {test_features['spread']:.2f}")
    print(f"     Trade imbalance: {test_features['trade_imbalance']:.4f}")
    print(f"\n   Logistic Regression:")
    print(f"     P(up): {proba_lr:.4f}")
    print(f"     Prediction: {'UP' if pred_lr == 1 else 'DOWN'}")
    print(f"\n   MLP:")
    print(f"     P(up): {proba_mlp:.4f}")
    print(f"     Prediction: {'UP' if pred_mlp == 1 else 'DOWN'}")
    
    # Test save/load
    print("\n5. Testing save/load...")
    model_path = 'data/test_predictor.pkl'
    pathlib.Path('data').mkdir(exist_ok=True)
    
    predictor_lr.save(model_path)
    print(f"   Saved model to {model_path}")
    
    loaded_predictor = OrderFlowPredictor.load(model_path)
    proba_loaded = loaded_predictor.predict_proba(test_features)
    print(f"   Loaded model prediction: {proba_loaded:.4f}")
    print(f"   Match original: {np.isclose(proba_lr, proba_loaded)}")
    
    print("\n" + "=" * 60)
    print("Predictor tests complete!")
    
    return predictor_lr, predictor_mlp


if __name__ == "__main__":
    test_predictor()
