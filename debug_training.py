#!/usr/bin/env python3
"""
Script debug để kiểm tra và sửa lỗi training
"""

import pandas as pd
import numpy as np
import os
import sys

# Thêm thư mục src vào path
sys.path.append('src')

def debug_data_cleaning():
    """Debug quá trình làm sạch dữ liệu"""
    print("🔍 Bắt đầu debug dữ liệu...")
    
    # Kiểm tra dữ liệu đã xử lý
    processed_file = 'data/processed/financial_statements_processed.csv'
    if not os.path.exists(processed_file):
        print(f"❌ Không tìm thấy file: {processed_file}")
        return None
    
    # Đọc dữ liệu
    df = pd.read_csv(processed_file)
    print(f"📊 Dữ liệu gốc: {df.shape}")
    
    # Kiểm tra các giá trị infinity và NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"🔢 Các cột số: {len(numeric_cols)}")
    
    # Kiểm tra infinity
    inf_count = 0
    for col in numeric_cols:
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            inf_count += inf_mask.sum()
            print(f"⚠️  Cột {col}: {inf_mask.sum()} giá trị infinity")
    
    print(f"📈 Tổng số giá trị infinity: {inf_count}")
    
    # Kiểm tra NaN
    nan_count = df[numeric_cols].isna().sum().sum()
    print(f"📉 Tổng số giá trị NaN: {nan_count}")
    
    # Làm sạch dữ liệu
    print("\n🧹 Bắt đầu làm sạch dữ liệu...")
    df_clean = df.copy()
    
    # Xử lý infinity
    for col in numeric_cols:
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    
    # Xử lý NaN cho cột số
    for col in numeric_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
    
    # Xử lý cột không phải số
    non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Kiểm tra sau khi làm sạch
    inf_count_after = 0
    for col in numeric_cols:
        inf_mask = np.isinf(df_clean[col])
        if inf_mask.any():
            inf_count_after += inf_mask.sum()
    
    nan_count_after = df_clean[numeric_cols].isna().sum().sum()
    
    print(f"✅ Sau khi làm sạch:")
    print(f"   - Infinity: {inf_count_after}")
    print(f"   - NaN: {nan_count_after}")
    print(f"   - Shape: {df_clean.shape}")
    
    return df_clean

def debug_feature_engineering(df_clean):
    """Debug quá trình tạo features"""
    print("\n🔧 Bắt đầu debug feature engineering...")
    
    try:
        from feature_engineering import FinancialFeatureEngineer
        
        # Chọn target variable
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        target_variable = 'Net_Income' if 'Net_Income' in numeric_cols else numeric_cols[0]
        print(f"🎯 Target variable: {target_variable}")
        
        # Khởi tạo feature engineer
        feature_engineer = FinancialFeatureEngineer()
        
        # Tạo features
        print("🔄 Đang tạo features...")
        df_features = feature_engineer.create_all_features(df_clean, target_variable)
        
        print(f"✅ Features được tạo: {df_features.shape}")
        
        # Kiểm tra features
        feature_numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        # Kiểm tra infinity trong features
        inf_count = 0
        for col in feature_numeric_cols:
            inf_mask = np.isinf(df_features[col])
            if inf_mask.any():
                inf_count += inf_mask.sum()
                print(f"⚠️  Feature {col}: {inf_mask.sum()} giá trị infinity")
        
        print(f"📊 Tổng số infinity trong features: {inf_count}")
        
        # Kiểm tra NaN trong features
        nan_count = df_features[feature_numeric_cols].isna().sum().sum()
        print(f"📉 Tổng số NaN trong features: {nan_count}")
        
        # Làm sạch features nếu cần
        if inf_count > 0 or nan_count > 0:
            print("🧹 Làm sạch features...")
            for col in feature_numeric_cols:
                df_features[col] = df_features[col].replace([np.inf, -np.inf], np.nan)
                df_features[col] = df_features[col].fillna(df_features[col].median())
        
        return df_features, target_variable
        
    except Exception as e:
        print(f"❌ Lỗi trong feature engineering: {str(e)}")
        return None, None

def debug_model_training(df_features, target_variable):
    """Debug quá trình training"""
    print("\n🤖 Bắt đầu debug model training...")
    
    try:
        from model_training import FinancialModelTrainer
        
        # Khởi tạo model trainer
        model_trainer = FinancialModelTrainer()
        
        # Kiểm tra dữ liệu trước khi training
        print(f"📊 Dữ liệu training: {df_features.shape}")
        print(f"🎯 Target: {target_variable}")
        
        # Kiểm tra target variable
        if target_variable not in df_features.columns:
            print(f"❌ Target variable {target_variable} không có trong features")
            return None
        
        # Loại bỏ các hàng có target là NaN
        df_train = df_features.dropna(subset=[target_variable])
        print(f"📈 Dữ liệu sau khi loại bỏ target NaN: {df_train.shape}")
        
        # Kiểm tra lại infinity và NaN
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        inf_count = 0
        for col in numeric_cols:
            inf_mask = np.isinf(df_train[col])
            if inf_mask.any():
                inf_count += inf_mask.sum()
        
        nan_count = df_train[numeric_cols].isna().sum().sum()
        
        print(f"🔍 Kiểm tra cuối cùng:")
        print(f"   - Infinity: {inf_count}")
        print(f"   - NaN: {nan_count}")
        
        if inf_count > 0 or nan_count > 0:
            print("🧹 Làm sạch lần cuối...")
            for col in numeric_cols:
                df_train[col] = df_train[col].replace([np.inf, -np.inf], np.nan)
                df_train[col] = df_train[col].fillna(df_train[col].median())
        
        # Training
        print("🚀 Bắt đầu training...")
        results = model_trainer.train_all_models(df_train, target_variable)
        
        print("✅ Training hoàn thành!")
        for name, result in results.items():
            print(f"   {name}: R² = {result['r2']:.4f}, MSE = {result['mse']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Lỗi trong model training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Hàm chính"""
    print("🚀 Bắt đầu debug training process...")
    
    # Debug data cleaning
    df_clean = debug_data_cleaning()
    if df_clean is None:
        return
    
    # Debug feature engineering
    df_features, target_variable = debug_feature_engineering(df_clean)
    if df_features is None:
        return
    
    # Debug model training
    results = debug_model_training(df_features, target_variable)
    
    if results:
        print("\n🎉 Debug hoàn thành thành công!")
    else:
        print("\n❌ Debug thất bại!")

if __name__ == "__main__":
    main()
