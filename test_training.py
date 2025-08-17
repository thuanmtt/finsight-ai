"""
Script test huấn luyện mô hình
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')

def test_training():
    """Test huấn luyện mô hình"""
    
    # Tải dữ liệu
    try:
        df = pd.read_csv('data/processed/financial_statements_processed.csv')
        print(f"✅ Đã tải dữ liệu: {df.shape}")
    except Exception as e:
        print(f"❌ Lỗi tải dữ liệu: {e}")
        return
    
    # Kiểm tra dữ liệu
    print("\n📊 Thông tin dữ liệu:")
    print(f"Số hàng: {len(df)}")
    print(f"Số cột: {len(df.columns)}")
    
    # Kiểm tra dữ liệu thiếu
    missing_data = df.isnull().sum()
    print(f"\nDữ liệu thiếu: {missing_data.sum()} giá trị")
    
    # Kiểm tra dữ liệu vô hạn
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    print(f"Dữ liệu vô hạn: {inf_count} giá trị")
    
    # Làm sạch dữ liệu
    df_clean = df.copy()
    
    # Xử lý dữ liệu vô hạn
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Xử lý dữ liệu thiếu - chỉ cho các cột số
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    # Xử lý cột không phải số
    non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    print(f"\n✅ Dữ liệu đã được làm sạch")
    
    # Test huấn luyện
    try:
        from model_training import FinancialModelTrainer
        
        trainer = FinancialModelTrainer()
        
        # Chọn target variable
        target_cols = ['Net_Income', 'ROE', 'ROA', 'Revenue']
        available_targets = [col for col in target_cols if col in df_clean.columns]
        
        if not available_targets:
            print("❌ Không tìm thấy target variable phù hợp")
            return
        
        target = available_targets[0]
        print(f"\n🎯 Sử dụng target: {target}")
        
        # Huấn luyện mô hình
        print("\n🚀 Bắt đầu huấn luyện...")
        results = trainer.train_all_models(df_clean, target)
        
        if results:
            print(f"\n✅ Huấn luyện thành công! {len(results)} mô hình")
            
            # Hiển thị kết quả
            print("\n📊 Kết quả huấn luyện:")
            for name, result in results.items():
                print(f"{name}: R² = {result['r2']:.4f}, MSE = {result['mse']:.4f}")
        else:
            print("❌ Không có mô hình nào được huấn luyện thành công")
            
    except Exception as e:
        print(f"❌ Lỗi khi huấn luyện: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training()
