"""
Module dự đoán sử dụng các mô hình AI đã huấn luyện
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialPredictor:
    """Lớp dự đoán tài chính sử dụng các mô hình AI"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.models = {}
        self.model_info = None
        self.best_model = None
        self.feature_engineer = None
        
        # Tải các mô hình
        self.load_models()
    
    def load_models(self):
        """Tải các mô hình đã huấn luyện"""
        try:
            # Tải thông tin mô hình
            model_info_path = os.path.join(self.models_dir, 'model_info.joblib')
            if os.path.exists(model_info_path):
                self.model_info = joblib.load(model_info_path)
                self.best_model = self.model_info['best_model']
                self.feature_names = self.model_info.get('feature_names', [])
                logger.info("Đã tải thông tin mô hình")
            
            # Tải các mô hình riêng lẻ
            model_files = [f for f in os.listdir(self.models_dir) 
                          if f.endswith('.joblib') and f != 'model_info.joblib']
            
            for model_file in model_files:
                model_name = model_file.replace('.joblib', '')
                model_path = os.path.join(self.models_dir, model_file)
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Đã tải mô hình: {model_name}")
            
            # Tải neural network models
            nn_files = [f for f in os.listdir(self.models_dir) if f.endswith('.h5')]
            for nn_file in nn_files:
                model_name = nn_file.replace('.h5', '')
                model_path = os.path.join(self.models_dir, nn_file)
                try:
                    import tensorflow as tf
                    self.models[model_name] = tf.keras.models.load_model(model_path)
                    logger.info(f"Đã tải neural network: {model_name}")
                except ImportError:
                    logger.warning("TensorFlow không được cài đặt, bỏ qua neural network models")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {str(e)}")
    
    def prepare_input_data(self, df: pd.DataFrame, target: str = None) -> pd.DataFrame:
        """Chuẩn bị dữ liệu đầu vào cho dự đoán"""
        df_prep = df.copy()
        
        # Loại bỏ cột target nếu có
        if target and target in df_prep.columns:
            df_prep = df_prep.drop(columns=[target])
        
        # Chỉ giữ lại các cột số
        df_prep = df_prep.select_dtypes(include=[np.number])
        
        # Xử lý dữ liệu thiếu
        df_prep = df_prep.fillna(df_prep.median())
        
        # Xử lý infinity
        df_prep = df_prep.replace([np.inf, -np.inf], np.nan)
        df_prep = df_prep.fillna(df_prep.median())
        
        # Nếu có feature names từ training, chỉ giữ lại các features đó
        if hasattr(self, 'feature_names') and self.feature_names:
            available_features = [f for f in self.feature_names if f in df_prep.columns]
            if available_features:
                df_prep = df_prep[available_features]
                logger.info(f"Sử dụng {len(available_features)} features cho dự đoán")
            else:
                logger.warning("Không tìm thấy features phù hợp, sử dụng tất cả features")
        else:
            logger.info("Không có feature names, sử dụng tất cả features")
        
        return df_prep
    
    def predict_single_model(self, model_name: str, input_data: pd.DataFrame) -> np.ndarray:
        """Dự đoán sử dụng một mô hình cụ thể"""
        if model_name not in self.models:
            raise ValueError(f"Mô hình {model_name} không tồn tại")
        
        model = self.models[model_name]
        
        # Chuẩn bị dữ liệu
        X = input_data.values
        
        # Dự đoán
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
            
            # Xử lý kết quả neural network
            if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
            
            return predictions
        else:
            raise ValueError(f"Mô hình {model_name} không có phương thức predict")
    
    def predict_all_models(self, input_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Dự đoán sử dụng tất cả các mô hình"""
        predictions = {}
        
        for model_name in self.models.keys():
            try:
                pred = self.predict_single_model(model_name, input_data)
                predictions[model_name] = pred
                logger.info(f"Dự đoán thành công với {model_name}")
            except Exception as e:
                logger.error(f"Lỗi dự đoán với {model_name}: {str(e)}")
        
        return predictions
    
    def ensemble_predict(self, input_data: pd.DataFrame, 
                        method: str = 'average') -> np.ndarray:
        """Dự đoán ensemble từ nhiều mô hình"""
        predictions = self.predict_all_models(input_data)
        
        if not predictions:
            raise ValueError("Không có dự đoán nào thành công")
        
        # Chuyển đổi thành array
        pred_arrays = list(predictions.values())
        
        if method == 'average':
            # Trung bình cộng
            ensemble_pred = np.mean(pred_arrays, axis=0)
        elif method == 'weighted':
            # Trung bình có trọng số dựa trên R2 score
            if self.model_info and 'model_scores' in self.model_info:
                scores = self.model_info['model_scores']
                weights = []
                for model_name in predictions.keys():
                    if model_name in scores:
                        weights.append(scores[model_name])
                    else:
                        weights.append(0.5)  # Trọng số mặc định
                
                # Chuẩn hóa trọng số
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                ensemble_pred = np.average(pred_arrays, axis=0, weights=weights)
            else:
                ensemble_pred = np.mean(pred_arrays, axis=0)
        elif method == 'median':
            # Trung vị
            ensemble_pred = np.median(pred_arrays, axis=0)
        else:
            raise ValueError(f"Phương thức ensemble {method} không được hỗ trợ")
        
        return ensemble_pred
    
    def predict_financial_metrics(self, company_data: pd.DataFrame, 
                                target_metrics: List[str]) -> Dict[str, Dict]:
        """Dự đoán các chỉ số tài chính"""
        results = {}
        
        for metric in target_metrics:
            if metric not in company_data.columns:
                logger.warning(f"Chỉ số {metric} không có trong dữ liệu")
                continue
            
            # Chuẩn bị dữ liệu
            input_data = self.prepare_input_data(company_data, target=metric)
            
            # Dự đoán ensemble
            try:
                prediction = self.ensemble_predict(input_data, method='weighted')
                
                results[metric] = {
                    'predicted_value': prediction[0] if len(prediction) == 1 else prediction,
                    'confidence': self.calculate_confidence(prediction),
                    'model_used': 'ensemble'
                }
                
                pred_value = prediction[0] if isinstance(prediction[0], (int, float)) else float(prediction[0])
                logger.info(f"Dự đoán {metric}: {pred_value:.2f}")
                
            except Exception as e:
                logger.error(f"Lỗi dự đoán {metric}: {str(e)}")
                results[metric] = {
                    'predicted_value': None,
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return results
    
    def predict_stock_price(self, company_data: pd.DataFrame, 
                          days_ahead: int = 30) -> Dict:
        """Dự đoán giá cổ phiếu trong tương lai"""
        # Chuẩn bị dữ liệu
        input_data = self.prepare_input_data(company_data)
        
        # Dự đoán giá hiện tại
        current_price_pred = self.ensemble_predict(input_data, method='weighted')
        
        # Dự đoán xu hướng
        trend_prediction = self.predict_trend(company_data, days_ahead)
        
        # Tính toán giá tương lai
        future_prices = []
        current_price = current_price_pred[0]
        
        for day in range(1, days_ahead + 1):
            # Áp dụng xu hướng và biến động
            daily_change = trend_prediction['daily_change']
            volatility = trend_prediction['volatility']
            
            # Thêm biến động ngẫu nhiên
            random_change = np.random.normal(daily_change, volatility)
            future_price = current_price * (1 + random_change)
            
            future_prices.append(future_price)
            current_price = future_price
        
        return {
            'current_price_prediction': current_price_pred[0],
            'future_prices': future_prices,
            'trend': trend_prediction['trend'],
            'confidence': self.calculate_confidence(current_price_pred),
            'days_ahead': days_ahead
        }
    
    def predict_trend(self, company_data: pd.DataFrame, 
                     days_ahead: int = 30) -> Dict:
        """Dự đoán xu hướng giá cổ phiếu"""
        # Tính toán các chỉ báo xu hướng
        if 'Close' in company_data.columns:
            prices = company_data['Close'].values
            
            # Tính toán trung bình động
            sma_short = np.mean(prices[-5:])  # 5 ngày
            sma_long = np.mean(prices[-20:])  # 20 ngày
            
            # Xác định xu hướng
            if sma_short > sma_long:
                trend = 'upward'
                daily_change = 0.001  # 0.1% tăng mỗi ngày
            else:
                trend = 'downward'
                daily_change = -0.001  # 0.1% giảm mỗi ngày
            
            # Tính toán biến động
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
        else:
            # Giá trị mặc định nếu không có dữ liệu giá
            trend = 'stable'
            daily_change = 0.0
            volatility = 0.02
        
        return {
            'trend': trend,
            'daily_change': daily_change,
            'volatility': volatility
        }
    
    def calculate_confidence(self, predictions: np.ndarray) -> float:
        """Tính toán độ tin cậy của dự đoán"""
        if len(predictions) == 1:
            return 0.8  # Độ tin cậy mặc định cho dự đoán đơn
        
        # Tính toán độ lệch chuẩn của các dự đoán
        std_pred = np.std(predictions)
        mean_pred = np.mean(predictions)
        
        # Độ tin cậy dựa trên độ nhất quán của các mô hình
        if mean_pred != 0:
            coefficient_of_variation = std_pred / abs(mean_pred)
            confidence = max(0.1, 1.0 - coefficient_of_variation)
        else:
            confidence = 0.5
        
        return confidence
    
    def generate_investment_recommendation(self, predictions: Dict) -> Dict:
        """Tạo khuyến nghị đầu tư dựa trên dự đoán"""
        recommendations = {
            'action': 'hold',
            'confidence': 0.5,
            'reasoning': [],
            'risk_level': 'medium'
        }
        
        # Phân tích các chỉ số
        if 'ROE' in predictions:
            roe = predictions['ROE']['predicted_value']
            if roe and roe > 0.15:
                recommendations['reasoning'].append('ROE cao (>15%)')
            elif roe and roe < 0.05:
                recommendations['reasoning'].append('ROE thấp (<5%)')
        
        if 'Current_Ratio' in predictions:
            cr = predictions['Current_Ratio']['predicted_value']
            if cr and cr > 2.0:
                recommendations['reasoning'].append('Tỷ lệ thanh toán hiện tại tốt')
            elif cr and cr < 1.0:
                recommendations['reasoning'].append('Tỷ lệ thanh toán hiện tại thấp')
        
        # Xác định hành động
        positive_signals = len([r for r in recommendations['reasoning'] 
                              if 'cao' in r or 'tốt' in r])
        negative_signals = len([r for r in recommendations['reasoning'] 
                              if 'thấp' in r])
        
        if positive_signals > negative_signals:
            recommendations['action'] = 'buy'
            recommendations['confidence'] = min(0.9, 0.5 + positive_signals * 0.1)
        elif negative_signals > positive_signals:
            recommendations['action'] = 'sell'
            recommendations['confidence'] = min(0.9, 0.5 + negative_signals * 0.1)
        else:
            recommendations['action'] = 'hold'
            recommendations['confidence'] = 0.5
        
        # Xác định mức độ rủi ro
        if recommendations['confidence'] > 0.7:
            recommendations['risk_level'] = 'low'
        elif recommendations['confidence'] < 0.4:
            recommendations['risk_level'] = 'high'
        
        return recommendations
    
    def get_model_performance(self) -> Dict:
        """Lấy thông tin hiệu suất của các mô hình"""
        if not self.model_info:
            return {}
        
        return {
            'best_model': self.best_model.__class__.__name__ if self.best_model else None,
            'best_score': self.model_info.get('best_score', 0),
            'model_scores': self.model_info.get('model_scores', {}),
            'total_models': len(self.models)
        }

def main():
    """Hàm chính để test dự đoán"""
    # Tạo dữ liệu mẫu
    sample_data = pd.DataFrame({
        'Revenue': [5000],
        'Total_Assets': [15000],
        'Total_Equity': [8000],
        'Net_Income': [500],
        'ROE': [0.15],
        'ROA': [0.08],
        'Current_Ratio': [2.5],
        'Debt_Ratio': [0.4]
    })
    
    # Khởi tạo predictor
    predictor = FinancialPredictor()
    
    # Kiểm tra xem có mô hình nào không
    if not predictor.models:
        logger.warning("Không có mô hình nào được tải. Vui lòng huấn luyện mô hình trước.")
        return
    
    # Dự đoán các chỉ số tài chính
    target_metrics = ['Net_Income', 'ROE', 'ROA']
    predictions = predictor.predict_financial_metrics(sample_data, target_metrics)
    
    # Hiển thị kết quả
    logger.info("=== KẾT QUẢ DỰ ĐOÁN ===")
    for metric, result in predictions.items():
        if result['predicted_value'] is not None:
            pred_val = result['predicted_value']
            if pred_val is not None:
                if isinstance(pred_val, (int, float)):
                    pred_val_str = f"{pred_val:.2f}"
                else:
                    pred_val_str = f"{float(pred_val):.2f}"
            else:
                pred_val_str = "N/A"
            logger.info(f"{metric}: {pred_val_str} (Độ tin cậy: {result['confidence']:.2f})")
        else:
            logger.info(f"{metric}: Lỗi - {result.get('error', 'Unknown')}")
    
    # Tạo khuyến nghị đầu tư
    recommendation = predictor.generate_investment_recommendation(predictions)
    logger.info(f"\nKhuyến nghị: {recommendation['action'].upper()}")
    conf_val = recommendation['confidence']
    if isinstance(conf_val, (int, float)):
        conf_str = f"{conf_val:.2f}"
    else:
        conf_str = f"{float(conf_val):.2f}"
    logger.info(f"Độ tin cậy: {conf_str}")
    logger.info(f"Mức độ rủi ro: {recommendation['risk_level']}")
    logger.info(f"Lý do: {', '.join(recommendation['reasoning'])}")

if __name__ == "__main__":
    main()
