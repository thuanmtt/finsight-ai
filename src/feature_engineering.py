"""
Module kỹ thuật tính năng cho mô hình AI tài chính
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import ta
import logging
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialFeatureEngineer:
    """Lớp kỹ thuật tính năng cho dữ liệu tài chính"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo các chỉ báo kỹ thuật"""
        df_tech = df.copy()
        
        # Chỉ báo trung bình động
        if 'Close' in df_tech.columns:
            df_tech['SMA_5'] = df_tech['Close'].rolling(window=5).mean()
            df_tech['SMA_20'] = df_tech['Close'].rolling(window=20).mean()
            df_tech['EMA_12'] = df_tech['Close'].ewm(span=12).mean()
            df_tech['EMA_26'] = df_tech['Close'].ewm(span=26).mean()
            
            # MACD
            df_tech['MACD'] = df_tech['EMA_12'] - df_tech['EMA_26']
            df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span=9).mean()
            df_tech['MACD_Histogram'] = df_tech['MACD'] - df_tech['MACD_Signal']
        
        # Chỉ báo RSI
        if 'Close' in df_tech.columns:
            delta = df_tech['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_tech['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if 'Close' in df_tech.columns:
            df_tech['BB_Middle'] = df_tech['Close'].rolling(window=20).mean()
            bb_std = df_tech['Close'].rolling(window=20).std()
            df_tech['BB_Upper'] = df_tech['BB_Middle'] + (bb_std * 2)
            df_tech['BB_Lower'] = df_tech['BB_Middle'] - (bb_std * 2)
            df_tech['BB_Width'] = df_tech['BB_Upper'] - df_tech['BB_Lower']
        
        return df_tech
    
    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo các tính năng tài chính"""
        df_fin = df.copy()
        
        # Tính năng tăng trưởng
        numeric_columns = df_fin.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['Year', 'Quarter', 'Date']:
                # Tăng trưởng YoY (Year over Year)
                df_fin[f'{col}_YoY_Growth'] = df_fin[col].pct_change(periods=4)
                
                # Tăng trưởng QoQ (Quarter over Quarter)
                df_fin[f'{col}_QoQ_Growth'] = df_fin[col].pct_change(periods=1)
                
                # Tỷ lệ so với trung bình ngành (giả định)
                df_fin[f'{col}_Industry_Ratio'] = df_fin[col] / df_fin[col].rolling(window=20).mean()
        
        # Tính năng hiệu suất
        if 'Revenue' in df_fin.columns and 'Total Assets' in df_fin.columns:
            df_fin['Asset_Turnover'] = df_fin['Revenue'] / df_fin['Total Assets']
        
        if 'Net Income' in df_fin.columns and 'Total Assets' in df_fin.columns:
            df_fin['ROA'] = df_fin['Net Income'] / df_fin['Total Assets']
        
        if 'Net Income' in df_fin.columns and 'Total Equity' in df_fin.columns:
            df_fin['ROE'] = df_fin['Net Income'] / df_fin['Total Equity']
        
        # Tính năng rủi ro
        if 'Total Debt' in df_fin.columns and 'Total Assets' in df_fin.columns:
            df_fin['Debt_to_Assets'] = df_fin['Total Debt'] / df_fin['Total Assets']
        
        if 'Current Assets' in df_fin.columns and 'Current Liabilities' in df_fin.columns:
            df_fin['Current_Ratio'] = df_fin['Current Assets'] / df_fin['Current Liabilities']
        
        return df_fin
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo các tính năng thời gian"""
        df_time = df.copy()
        
        if 'Date' in df_time.columns:
            df_time['Date'] = pd.to_datetime(df_time['Date'])
            
            # Tính năng thời gian
            df_time['Year'] = df_time['Date'].dt.year
            df_time['Month'] = df_time['Date'].dt.month
            df_time['Quarter'] = df_time['Date'].dt.quarter
            df_time['Day_of_Week'] = df_time['Date'].dt.dayofweek
            df_time['Day_of_Year'] = df_time['Date'].dt.dayofyear
            
            # Tính năng mùa vụ
            df_time['Is_Q4'] = (df_time['Quarter'] == 4).astype(int)
            df_time['Is_Year_End'] = (df_time['Month'] == 12).astype(int)
            
            # Tính năng xu hướng thời gian
            df_time['Time_Trend'] = range(len(df_time))
        
        return df_time
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo các tính năng tương tác"""
        df_inter = df.copy()
        
        # Tương tác giữa các tỷ lệ tài chính
        if 'ROE' in df_inter.columns and 'ROA' in df_inter.columns:
            df_inter['ROE_ROA_Ratio'] = df_inter['ROE'] / df_inter['ROA']
        
        if 'Current_Ratio' in df_inter.columns and 'Debt_to_Assets' in df_inter.columns:
            df_inter['Liquidity_Debt_Score'] = df_inter['Current_Ratio'] * (1 - df_inter['Debt_to_Assets'])
        
        # Tương tác với tăng trưởng
        if 'Revenue' in df_inter.columns and 'Revenue_YoY_Growth' in df_inter.columns:
            df_inter['Revenue_Growth_Score'] = df_inter['Revenue'] * df_inter['Revenue_YoY_Growth']
        
        return df_inter
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mã hóa các tính năng phân loại"""
        df_encoded = df.copy()
        
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Chuẩn hóa các tính năng"""
        df_scaled = df.copy()
        
        # Chọn các cột số
        numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns
        
        if fit:
            df_scaled[numeric_columns] = self.scaler.fit_transform(df_scaled[numeric_columns])
        else:
            df_scaled[numeric_columns] = self.scaler.transform(df_scaled[numeric_columns])
        
        return df_scaled
    
    def select_features(self, df: pd.DataFrame, target: str, k: int = 50) -> pd.DataFrame:
        """Lựa chọn tính năng quan trọng"""
        df_selected = df.copy()
        
        # Loại bỏ cột target và các cột không phải số
        feature_columns = df_selected.select_dtypes(include=[np.number]).columns
        if target in feature_columns:
            feature_columns = feature_columns.drop(target)
        
        X = df_selected[feature_columns]
        y = df_selected[target]
        
        # Loại bỏ dữ liệu thiếu
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Xử lý infinity và NaN trước khi select
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Kiểm tra lại sau khi xử lý
        if X.isnull().any().any() or np.isinf(X.values).any():
            logger.warning("Vẫn còn NaN hoặc infinity sau khi xử lý, bỏ qua feature selection")
            return df_selected
        
        # Lựa chọn tính năng
        try:
            selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_columns)))
            X_selected = selector.fit_transform(X, y)
            
            # Lấy tên các tính năng được chọn
            selected_features = feature_columns[selector.get_support()]
            self.feature_names = list(selected_features)
            
            # Tạo DataFrame với các tính năng được chọn
            df_selected = df_selected[selected_features]
            df_selected[target] = df[target]
            
            return df_selected
        except Exception as e:
            logger.warning(f"Lỗi trong feature selection: {str(e)}, trả về dữ liệu gốc")
            return df_selected
    
    def create_all_features(self, df: pd.DataFrame, target: str = None) -> pd.DataFrame:
        """Tạo tất cả các tính năng"""
        logger.info("Bắt đầu tạo tính năng...")
        
        # Tạo các loại tính năng
        df_features = self.create_technical_indicators(df)
        df_features = self.create_financial_features(df_features)
        df_features = self.create_time_features(df_features)
        df_features = self.create_interaction_features(df_features)
        
        # Mã hóa tính năng phân loại
        df_features = self.encode_categorical_features(df_features)
        
        # Xử lý dữ liệu thiếu
        df_features = df_features.fillna(df_features.median())
        
        # Lựa chọn tính năng nếu có target
        if target and target in df_features.columns:
            df_features = self.select_features(df_features, target)
        
        # Chuẩn hóa tính năng
        if target and target in df_features.columns:
            target_col = df_features[target]
            df_features = df_features.drop(columns=[target])
            df_features = self.scale_features(df_features)
            df_features[target] = target_col
        else:
            df_features = self.scale_features(df_features)
        
        logger.info(f"Hoàn thành tạo tính năng. Kích thước: {df_features.shape}")
        return df_features
    
    def get_feature_importance(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Lấy độ quan trọng của các tính năng"""
        from sklearn.ensemble import RandomForestRegressor
        
        X = df.drop(columns=[target])
        y = df[target]
        
        # Loại bỏ dữ liệu thiếu
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Huấn luyện Random Forest để lấy độ quan trọng
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Tạo dictionary độ quan trọng
        feature_importance = dict(zip(X.columns, rf.feature_importances_))
        
        # Sắp xếp theo độ quan trọng giảm dần
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance

def main():
    """Hàm chính để test feature engineering"""
    # Tạo dữ liệu mẫu
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='Q')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Company': ['AAPL'] * len(dates),
        'Revenue': np.random.uniform(1000, 5000, len(dates)),
        'Net_Income': np.random.uniform(100, 500, len(dates)),
        'Total_Assets': np.random.uniform(5000, 15000, len(dates)),
        'Total_Equity': np.random.uniform(2000, 8000, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates))
    })
    
    # Khởi tạo feature engineer
    fe = FinancialFeatureEngineer()
    
    # Tạo tính năng
    df_features = fe.create_all_features(sample_data, target='Net_Income')
    
    # Lấy độ quan trọng tính năng
    importance = fe.get_feature_importance(df_features, 'Net_Income')
    
    logger.info("Top 10 tính năng quan trọng nhất:")
    for i, (feature, score) in enumerate(list(importance.items())[:10]):
        logger.info(f"{i+1}. {feature}: {score:.4f}")

if __name__ == "__main__":
    main()
