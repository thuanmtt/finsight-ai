"""
Module huấn luyện các mô hình AI tài chính
"""

import pandas as pd
import numpy as np
import os
import joblib
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
# Import TensorFlow với xử lý lỗi
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow không khả dụng, bỏ qua neural network models")
import warnings
warnings.filterwarnings('ignore')

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialModelTrainer:
    """Lớp huấn luyện các mô hình AI tài chính"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_score = -np.inf
        
        # Tạo thư mục models nếu chưa tồn tại
        os.makedirs(models_dir, exist_ok=True)
    
    def prepare_data(self, df: pd.DataFrame, target: str, test_size: float = 0.2) -> Tuple:
        """Chuẩn bị dữ liệu cho huấn luyện"""
        # Loại bỏ dữ liệu thiếu
        df_clean = df.dropna()
        
        # Tách features và target
        X = df_clean.drop(columns=[target])
        y = df_clean[target]
        
        # Chỉ giữ lại các cột số
        X = X.select_dtypes(include=[np.number])
        
        # Xử lý dữ liệu vô hạn và quá lớn
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Loại bỏ các cột có độ lệch chuẩn = 0 (không có thông tin)
        X = X.loc[:, X.std() > 0]
        
        # Chuẩn hóa dữ liệu để tránh giá trị quá lớn
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Loại bỏ dữ liệu thiếu sau khi xử lý
        mask = ~(X_scaled.isnull().any(axis=1) | y.isnull())
        X_scaled = X_scaled[mask]
        y = y[mask]
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Dữ liệu huấn luyện: {X_train.shape}")
        logger.info(f"Dữ liệu kiểm tra: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Huấn luyện các mô hình tuyến tính"""
        logger.info("Huấn luyện các mô hình tuyến tính...")
        
        linear_models = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(alpha=1.0),
            'Lasso_Regression': Lasso(alpha=0.1)
        }
        
        results = {}
        
        for name, model in linear_models.items():
            logger.info(f"Huấn luyện {name}...")
            
            try:
                # Huấn luyện mô hình
                model.fit(X_train, y_train)
                
                # Dự đoán
                y_pred = model.predict(X_test)
                
                # Đánh giá
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                logger.info(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            except Exception as e:
                logger.error(f"Lỗi khi huấn luyện {name}: {str(e)}")
                continue
        
        return results
    
    def train_ensemble_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Huấn luyện các mô hình ensemble"""
        logger.info("Huấn luyện các mô hình ensemble...")
        
        ensemble_models = {
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
            'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=False)
        }
        
        results = {}
        
        for name, model in ensemble_models.items():
            logger.info(f"Huấn luyện {name}...")
            
            try:
                # Huấn luyện mô hình
                model.fit(X_train, y_train)
                
                # Dự đoán
                y_pred = model.predict(X_test)
                
                # Đánh giá
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                logger.info(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            except Exception as e:
                logger.error(f"Lỗi khi huấn luyện {name}: {str(e)}")
                continue
        
        return results
    
    def train_neural_network(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Huấn luyện mạng neural"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow không khả dụng, bỏ qua neural network")
            return {}
        
        logger.info("Huấn luyện mạng neural...")
        
        # Chuyển đổi dữ liệu
        X_train_nn = X_train.values
        X_test_nn = X_test.values
        y_train_nn = y_train.values
        y_test_nn = y_test.values
        
        # Xây dựng mô hình
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        # Biên dịch mô hình
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Huấn luyện
        history = model.fit(
            X_train_nn, y_train_nn,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Dự đoán
        y_pred = model.predict(X_test_nn).flatten()
        
        # Đánh giá
        mse = mean_squared_error(y_test_nn, y_pred)
        mae = mean_absolute_error(y_test_nn, y_pred)
        r2 = r2_score(y_test_nn, y_pred)
        
        results = {
            'Neural_Network': {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred,
                'history': history
            }
        }
        
        logger.info(f"Neural Network - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        return results
    
    def train_lstm_model(self, df: pd.DataFrame, target: str, sequence_length: int = 10) -> Dict:
        """Huấn luyện mô hình LSTM cho chuỗi thời gian"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow không khả dụng, bỏ qua LSTM model")
            return {}
        
        logger.info("Huấn luyện mô hình LSTM...")
        
        # Chuẩn bị dữ liệu chuỗi thời gian
        data = df[target].values.reshape(-1, 1)
        
        # Tạo sequences
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Chia dữ liệu
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Xây dựng mô hình LSTM
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Huấn luyện
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Dự đoán
        y_pred = model.predict(X_test)
        
        # Đánh giá
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'LSTM': {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred,
                'history': history
            }
        }
        
        logger.info(f"LSTM - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                             model_name: str = 'Random_Forest') -> Any:
        """Tối ưu hóa hyperparameters"""
        logger.info(f"Tối ưu hóa hyperparameters cho {model_name}...")
        
        if model_name == 'Random_Forest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name == 'XGBoost':
            model = xgb.XGBRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        else:
            logger.warning(f"Không hỗ trợ tối ưu hóa cho {model_name}")
            return None
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_all_models(self, df: pd.DataFrame, target: str, feature_names: List[str] = None) -> Dict:
        """Huấn luyện tất cả các mô hình"""
        logger.info("Bắt đầu huấn luyện tất cả các mô hình...")
        
        # Lưu feature names
        if feature_names:
            self.feature_names = feature_names
            logger.info(f"Lưu {len(feature_names)} feature names")
        
        # Chuẩn bị dữ liệu
        X_train, X_test, y_train, y_test = self.prepare_data(df, target)
        
        # Huấn luyện các loại mô hình
        linear_results = self.train_linear_models(X_train, X_test, y_train, y_test)
        ensemble_results = self.train_ensemble_models(X_train, X_test, y_train, y_test)
        nn_results = self.train_neural_network(X_train, X_test, y_train, y_test)
        
        # Kết hợp kết quả
        all_results = {**linear_results, **ensemble_results, **nn_results}
        
        # Tìm mô hình tốt nhất
        best_model_name = max(all_results.keys(), 
                            key=lambda x: all_results[x]['r2'])
        self.best_model = all_results[best_model_name]['model']
        self.best_score = all_results[best_model_name]['r2']
        
        logger.info(f"Mô hình tốt nhất: {best_model_name} với R2: {self.best_score:.4f}")
        
        # Lưu kết quả
        self.models = all_results
        self.model_scores = {name: results['r2'] for name, results in all_results.items()}
        
        return all_results
    
    def save_models(self):
        """Lưu các mô hình đã huấn luyện"""
        logger.info("Lưu các mô hình...")
        
        for name, results in self.models.items():
            model = results['model']
            model_path = os.path.join(self.models_dir, f"{name}.joblib")
            
            # Lưu mô hình
            if hasattr(model, 'save'):
                # Cho neural networks
                model.save(os.path.join(self.models_dir, f"{name}.h5"))
            else:
                # Cho sklearn models
                joblib.dump(model, model_path)
            
            logger.info(f"Đã lưu mô hình: {name}")
        
        # Lưu thông tin mô hình
        model_info = {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'model_scores': self.model_scores,
            'feature_names': self.feature_names if hasattr(self, 'feature_names') else None
        }
        joblib.dump(model_info, os.path.join(self.models_dir, 'model_info.joblib'))
    
    def load_models(self):
        """Tải các mô hình đã lưu"""
        logger.info("Tải các mô hình...")
        
        model_info_path = os.path.join(self.models_dir, 'model_info.joblib')
        if os.path.exists(model_info_path):
            model_info = joblib.load(model_info_path)
            self.best_model = model_info['best_model']
            self.best_score = model_info['best_score']
            self.model_scores = model_info['model_scores']
            logger.info("Đã tải thông tin mô hình")
        else:
            logger.warning("Không tìm thấy file thông tin mô hình")

def main():
    """Hàm chính để test huấn luyện mô hình"""
    # Tạo dữ liệu mẫu
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'Revenue': np.random.uniform(1000, 5000, n_samples),
        'Total_Assets': np.random.uniform(5000, 15000, n_samples),
        'Total_Equity': np.random.uniform(2000, 8000, n_samples),
        'Net_Income': np.random.uniform(100, 500, n_samples),
        'ROE': np.random.uniform(0.05, 0.25, n_samples),
        'ROA': np.random.uniform(0.03, 0.15, n_samples),
        'Current_Ratio': np.random.uniform(1.0, 3.0, n_samples),
        'Debt_Ratio': np.random.uniform(0.2, 0.8, n_samples),
        'Target_Price': np.random.uniform(50, 200, n_samples)
    })
    
    # Khởi tạo trainer
    trainer = FinancialModelTrainer()
    
    # Huấn luyện tất cả mô hình
    results = trainer.train_all_models(sample_data, 'Target_Price')
    
    # Lưu mô hình
    trainer.save_models()
    
    # Hiển thị kết quả
    logger.info("\n=== KẾT QUẢ HUẤN LUYỆN ===")
    for name, result in results.items():
        logger.info(f"{name}: R2 = {result['r2']:.4f}, MSE = {result['mse']:.4f}")

if __name__ == "__main__":
    main()
