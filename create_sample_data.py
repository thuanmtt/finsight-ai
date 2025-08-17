"""
Script tạo dữ liệu mẫu để test ứng dụng FinSight AI
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_sample_financial_data():
    """Tạo dữ liệu tài chính mẫu"""
    
    # Tạo thư mục data nếu chưa tồn tại
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Các công ty mẫu
    companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    
    # Tạo dữ liệu theo quý từ 2020-2023
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Tạo danh sách các quý
    quarters = []
    current_date = start_date
    while current_date <= end_date:
        quarters.append(current_date)
        current_date += timedelta(days=90)
    
    # Tạo dữ liệu
    data = []
    
    for company in companies:
        # Giá trị cơ bản cho mỗi công ty
        base_revenue = np.random.uniform(20000, 100000)
        base_assets = base_revenue * np.random.uniform(1.5, 3.0)
        base_equity = base_assets * np.random.uniform(0.3, 0.7)
        
        for quarter in quarters:
            # Thêm xu hướng tăng trưởng theo thời gian
            time_factor = (quarter - start_date).days / 365.25
            
            # Revenue với xu hướng tăng trưởng
            revenue = base_revenue * (1 + 0.1 * time_factor + np.random.normal(0, 0.05))
            
            # Net Income (khoảng 10-20% của Revenue)
            net_income = revenue * np.random.uniform(0.1, 0.2) * (1 + np.random.normal(0, 0.1))
            
            # Total Assets với xu hướng tăng
            total_assets = base_assets * (1 + 0.08 * time_factor + np.random.normal(0, 0.03))
            
            # Total Equity
            total_equity = base_equity * (1 + 0.12 * time_factor + np.random.normal(0, 0.04))
            
            # Current Assets (khoảng 30-50% của Total Assets)
            current_assets = total_assets * np.random.uniform(0.3, 0.5)
            
            # Current Liabilities (khoảng 20-40% của Total Assets)
            current_liabilities = total_assets * np.random.uniform(0.2, 0.4)
            
            # Total Debt (khoảng 20-60% của Total Assets)
            total_debt = total_assets * np.random.uniform(0.2, 0.6)
            
            # Gross Profit (khoảng 40-60% của Revenue)
            gross_profit = revenue * np.random.uniform(0.4, 0.6)
            
            # Tính toán các tỷ lệ
            roe = net_income / total_equity if total_equity > 0 else 0
            roa = net_income / total_assets if total_assets > 0 else 0
            current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
            debt_ratio = total_debt / total_assets if total_assets > 0 else 0
            gross_margin = gross_profit / revenue if revenue > 0 else 0
            net_margin = net_income / revenue if revenue > 0 else 0
            
            # Giá cổ phiếu (dựa trên ROE và tăng trưởng)
            base_price = 100 + (roe * 1000) + (time_factor * 10)
            stock_price = base_price * (1 + np.random.normal(0, 0.1))
            
            data.append({
                'Company': company,
                'Date': quarter.strftime('%Y-%m-%d'),
                'Year': quarter.year,
                'Quarter': (quarter.month - 1) // 3 + 1,
                'Revenue': revenue,
                'Net_Income': net_income,
                'Total_Assets': total_assets,
                'Total_Equity': total_equity,
                'Current_Assets': current_assets,
                'Current_Liabilities': current_liabilities,
                'Total_Debt': total_debt,
                'Gross_Profit': gross_profit,
                'ROE': roe,
                'ROA': roa,
                'Current_Ratio': current_ratio,
                'Debt_Ratio': debt_ratio,
                'Gross_Margin': gross_margin,
                'Net_Margin': net_margin,
                'Close': stock_price
            })
    
    # Tạo DataFrame
    df = pd.DataFrame(data)
    
    # Lưu dữ liệu thô
    df.to_csv('data/raw/financial_statements.csv', index=False)
    print(f"Đã tạo dữ liệu thô: {len(df)} bản ghi")
    
    # Tạo dữ liệu đã xử lý
    processed_df = df.copy()
    
    # Thêm các tính năng bổ sung
    processed_df['Asset_Turnover'] = processed_df['Revenue'] / processed_df['Total_Assets']
    processed_df['Equity_Multiplier'] = processed_df['Total_Assets'] / processed_df['Total_Equity']
    processed_df['Interest_Coverage'] = processed_df['Net_Income'] / (processed_df['Total_Debt'] * 0.05)  # Giả sử lãi suất 5%
    
    # Thêm các chỉ báo kỹ thuật
    for company in companies:
        company_data = processed_df[processed_df['Company'] == company].copy()
        company_data = company_data.sort_values('Date')
        
        # Moving averages
        company_data['SMA_5'] = company_data['Close'].rolling(window=5).mean()
        company_data['SMA_20'] = company_data['Close'].rolling(window=20).mean()
        
        # Price changes
        company_data['Price_Change'] = company_data['Close'].pct_change()
        company_data['Price_Change_5Q'] = company_data['Close'].pct_change(periods=5)
        
        # Revenue growth
        company_data['Revenue_Growth'] = company_data['Revenue'].pct_change()
        company_data['Revenue_Growth_YoY'] = company_data['Revenue'].pct_change(periods=4)
        
        # Cập nhật lại dữ liệu
        processed_df.loc[processed_df['Company'] == company] = company_data
    
    # Lưu dữ liệu đã xử lý
    processed_df.to_csv('data/processed/financial_statements_processed.csv', index=False)
    print(f"Đã tạo dữ liệu đã xử lý: {len(processed_df)} bản ghi")
    
    return df, processed_df

def create_sample_stock_data():
    """Tạo dữ liệu giá cổ phiếu mẫu"""
    
    companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    # Tạo dữ liệu hàng ngày trong 1 năm
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    dates = pd.date_range(start_date, end_date, freq='D')
    
    data = []
    
    for company in companies:
        # Giá cổ phiếu ban đầu
        base_price = np.random.uniform(100, 500)
        current_price = base_price
        
        for date in dates:
            # Thêm biến động ngẫu nhiên
            daily_return = np.random.normal(0.001, 0.02)  # Trung bình 0.1%, độ lệch chuẩn 2%
            current_price *= (1 + daily_return)
            
            # Tính toán các chỉ báo
            high = current_price * (1 + abs(np.random.normal(0, 0.01)))
            low = current_price * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.uniform(1000000, 10000000)
            
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Company': company,
                'Open': current_price * (1 + np.random.normal(0, 0.005)),
                'High': high,
                'Low': low,
                'Close': current_price,
                'Volume': volume
            })
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/stock_prices.csv', index=False)
    print(f"Đã tạo dữ liệu giá cổ phiếu: {len(df)} bản ghi")
    
    return df

if __name__ == "__main__":
    print("Đang tạo dữ liệu mẫu...")
    
    # Tạo dữ liệu tài chính
    financial_df, processed_df = create_sample_financial_data()
    
    # Tạo dữ liệu giá cổ phiếu
    stock_df = create_sample_stock_data()
    
    print("✅ Hoàn thành tạo dữ liệu mẫu!")
    print(f"📊 Dữ liệu tài chính: {len(financial_df)} bản ghi")
    print(f"📈 Dữ liệu giá cổ phiếu: {len(stock_df)} bản ghi")
    print(f"🏢 Số công ty: {len(financial_df['Company'].unique())}")
