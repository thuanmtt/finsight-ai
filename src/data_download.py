"""
Script tải dữ liệu từ Kaggle dataset
Financial Statements of Major Companies (2009-2023)
"""

import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_kaggle_dataset():
    """Tải dataset từ Kaggle"""
    try:
        # Khởi tạo Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Tạo thư mục data nếu chưa tồn tại
        os.makedirs('data/raw', exist_ok=True)
        
        # Tải dataset
        dataset_name = "rish59/financial-statements-of-major-companies2009-2023"
        logger.info(f"Đang tải dataset: {dataset_name}")
        
        api.dataset_download_files(dataset_name, path='data/raw', unzip=True)
        
        logger.info("Tải dataset thành công!")
        
        # Liệt kê các file đã tải
        files = os.listdir('data/raw')
        logger.info(f"Các file đã tải: {files}")
        
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi tải dataset: {str(e)}")
        return False

def explore_dataset():
    """Khám phá dataset đã tải"""
    try:
        raw_data_path = 'data/raw'
        files = os.listdir(raw_data_path)
        
        logger.info("=== KHÁM PHÁ DATASET ===")
        
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(raw_data_path, file)
                df = pd.read_csv(file_path)
                
                logger.info(f"\nFile: {file}")
                logger.info(f"Kích thước: {df.shape}")
                logger.info(f"Cột: {list(df.columns)}")
                logger.info(f"Thông tin cơ bản:")
                logger.info(f"  - Số hàng: {len(df)}")
                logger.info(f"  - Số cột: {len(df.columns)}")
                logger.info(f"  - Dữ liệu thiếu: {df.isnull().sum().sum()}")
                
                # Hiển thị 5 hàng đầu
                logger.info(f"5 hàng đầu:")
                print(df.head())
                
    except Exception as e:
        logger.error(f"Lỗi khi khám phá dataset: {str(e)}")

def main():
    """Hàm chính"""
    logger.info("Bắt đầu tải dữ liệu từ Kaggle...")
    
    # Tải dataset
    success = download_kaggle_dataset()
    
    if success:
        # Khám phá dataset
        explore_dataset()
        logger.info("Hoàn thành tải và khám phá dataset!")
    else:
        logger.error("Không thể tải dataset!")

if __name__ == "__main__":
    main()
