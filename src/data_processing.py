"""
Module xử lý dữ liệu tài chính
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialDataProcessor:
    """Lớp xử lý dữ liệu tài chính"""
    
    def __init__(self, data_path: str = 'data/raw'):
        self.data_path = data_path
        self.data = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Tải tất cả dữ liệu CSV từ thư mục raw"""
        try:
            files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
            
            for file in files:
                file_path = os.path.join(self.data_path, file)
                df_name = file.replace('.csv', '')
                self.data[df_name] = pd.read_csv(file_path)
                logger.info(f"Đã tải {file}: {self.data[df_name].shape}")
                
            return self.data
            
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu: {str(e)}")
            return {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Làm sạch dữ liệu"""
        df_clean = df.copy()
        
        # Xử lý dữ liệu thiếu
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].median())
        
        # Xử lý dữ liệu text
        text_columns = df_clean.select_dtypes(include=['object']).columns
        df_clean[text_columns] = df_clean[text_columns].fillna('Unknown')
        
        # Loại bỏ hàng trùng lặp
        df_clean = df_clean.drop_duplicates()
        
        return df_clean
    
    def process_financial_statements(self) -> pd.DataFrame:
        """Xử lý báo cáo tài chính"""
        if 'financial_statements' not in self.data:
            logger.error("Không tìm thấy dữ liệu báo cáo tài chính")
            return pd.DataFrame()
        
        df = self.data['financial_statements'].copy()
        
        # Làm sạch dữ liệu
        df = self.clean_data(df)
        
        # Chuyển đổi cột ngày tháng
        date_columns = ['Date', 'Year', 'Quarter']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Chuyển đổi cột số
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def calculate_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán các tỷ lệ tài chính"""
        df_ratios = df.copy()
        
        # Tỷ lệ thanh toán hiện tại
        if 'Current Assets' in df_ratios.columns and 'Current Liabilities' in df_ratios.columns:
            df_ratios['Current Ratio'] = df_ratios['Current Assets'] / df_ratios['Current Liabilities']
        
        # Tỷ lệ nợ
        if 'Total Debt' in df_ratios.columns and 'Total Assets' in df_ratios.columns:
            df_ratios['Debt Ratio'] = df_ratios['Total Debt'] / df_ratios['Total Assets']
        
        # ROE (Return on Equity)
        if 'Net Income' in df_ratios.columns and 'Total Equity' in df_ratios.columns:
            df_ratios['ROE'] = df_ratios['Net Income'] / df_ratios['Total Equity']
        
        # ROA (Return on Assets)
        if 'Net Income' in df_ratios.columns and 'Total Assets' in df_ratios.columns:
            df_ratios['ROA'] = df_ratios['Net Income'] / df_ratios['Total Assets']
        
        # Tỷ lệ lợi nhuận gộp
        if 'Gross Profit' in df_ratios.columns and 'Revenue' in df_ratios.columns:
            df_ratios['Gross Margin'] = df_ratios['Gross Profit'] / df_ratios['Revenue']
        
        # Tỷ lệ lợi nhuận ròng
        if 'Net Income' in df_ratios.columns and 'Revenue' in df_ratios.columns:
            df_ratios['Net Margin'] = df_ratios['Net Income'] / df_ratios['Revenue']
        
        return df_ratios
    
    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo features chuỗi thời gian"""
        df_ts = df.copy()
        
        # Sắp xếp theo thời gian
        if 'Date' in df_ts.columns:
            df_ts = df_ts.sort_values('Date')
        
        # Tính toán thay đổi theo thời gian
        numeric_columns = df_ts.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['Year', 'Quarter']:
                # Thay đổi tuyệt đối
                df_ts[f'{col}_Change'] = df_ts[col].diff()
                # Thay đổi phần trăm
                df_ts[f'{col}_Pct_Change'] = df_ts[col].pct_change()
                # Trung bình động 3 tháng
                df_ts[f'{col}_MA3'] = df_ts[col].rolling(window=3).mean()
                # Trung bình động 12 tháng
                df_ts[f'{col}_MA12'] = df_ts[col].rolling(window=12).mean()
        
        return df_ts
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Lưu dữ liệu đã xử lý"""
        os.makedirs('data/processed', exist_ok=True)
        file_path = os.path.join('data/processed', filename)
        df.to_csv(file_path, index=False)
        logger.info(f"Đã lưu dữ liệu đã xử lý: {file_path}")
    
    def process_all_data(self) -> Dict[str, pd.DataFrame]:
        """Xử lý tất cả dữ liệu"""
        logger.info("Bắt đầu xử lý dữ liệu...")
        
        # Tải dữ liệu
        self.load_data()
        
        processed_data = {}
        
        # Xử lý từng loại dữ liệu
        for name, df in self.data.items():
            logger.info(f"Đang xử lý: {name}")
            
            # Làm sạch dữ liệu
            df_clean = self.clean_data(df)
            
            # Tính toán tỷ lệ tài chính
            df_ratios = self.calculate_financial_ratios(df_clean)
            
            # Tạo features chuỗi thời gian
            df_final = self.create_time_series_features(df_ratios)
            
            processed_data[name] = df_final
            
            # Lưu dữ liệu đã xử lý
            self.save_processed_data(df_final, f"{name}_processed.csv")
        
        logger.info("Hoàn thành xử lý dữ liệu!")
        return processed_data

def main():
    """Hàm chính để chạy xử lý dữ liệu"""
    processor = FinancialDataProcessor()
    processed_data = processor.process_all_data()
    
    # Hiển thị thông tin dữ liệu đã xử lý
    for name, df in processed_data.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Kích thước: {df.shape}")
        logger.info(f"  Cột: {list(df.columns)}")

if __name__ == "__main__":
    main()
