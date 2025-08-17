"""
Ứng dụng FinSight AI - Phiên bản đơn giản
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys

# Thêm thư mục src vào path
sys.path.append('src')

# Import các module cơ bản
try:
    from data_processing import FinancialDataProcessor
    from visualization import FinancialVisualizer
except ImportError as e:
    st.error(f"Lỗi import module: {e}")
    st.stop()

# Cấu hình trang
st.set_page_config(
    page_title="FinSight AI - Quản Lý và Dự Báo Tài Chính",
    page_icon="📊",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Tải dữ liệu"""
    try:
        # Kiểm tra dữ liệu đã xử lý
        if os.path.exists('data/processed/financial_statements_processed.csv'):
            df = pd.read_csv('data/processed/financial_statements_processed.csv')
            st.success(f"✅ Đã tải dữ liệu: {len(df)} bản ghi")
            return df
        else:
            st.warning("⚠️ Không tìm thấy dữ liệu đã xử lý")
            return None
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu: {e}")
        return None

def show_dashboard(df):
    """Hiển thị dashboard chính"""
    st.markdown('<h1 class="main-header">📊 FinSight AI</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Quản Lý và Dự Báo Tài Chính Doanh Nghiệp</h2>', unsafe_allow_html=True)
    
    # Thống kê tổng quan
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏢 Tổng số công ty", len(df['Company'].unique()))
    
    with col2:
        st.metric("📊 Tổng số bản ghi", len(df))
    
    with col3:
        if 'Date' in df.columns:
            date_range = f"{df['Date'].min()[:7]} - {df['Date'].max()[:7]}"
            st.metric("📅 Khoảng thời gian", date_range)
    
    with col4:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("📈 Số chỉ số", len(numeric_cols))
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Điều khiển")
        
        # Chọn công ty
        companies = df['Company'].unique()
        selected_company = st.selectbox("Chọn công ty:", companies)
        
        # Chọn chỉ số
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_metrics = st.multiselect(
            "Chọn chỉ số tài chính:",
            numeric_columns,
            default=numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns
        )
        
        # Chọn chức năng
        function = st.selectbox(
            "Chọn chức năng:",
            ["📊 Dashboard", "📈 Phân tích", "🔍 So sánh", "📋 Thống kê"]
        )
    
    # Hiển thị chức năng được chọn
    if function == "📊 Dashboard":
        show_main_dashboard(df, selected_company, selected_metrics)
    elif function == "📈 Phân tích":
        show_analysis(df, selected_company, selected_metrics)
    elif function == "🔍 So sánh":
        show_comparison(df, selected_metrics)
    elif function == "📋 Thống kê":
        show_statistics(df, selected_metrics)

def show_main_dashboard(df, company, metrics):
    """Hiển thị dashboard chính"""
    st.header("📊 Dashboard Tổng Quan")
    
    if company and metrics:
        st.subheader(f"📈 Phân tích {company}")
        
        # Lọc dữ liệu cho công ty được chọn
        company_data = df[df['Company'] == company]
        
        if not company_data.empty and 'Date' in company_data.columns:
            # Biểu đồ chỉ số theo thời gian
            fig = go.Figure()
            
            for metric in metrics[:3]:  # Chỉ hiển thị 3 chỉ số đầu
                if metric in company_data.columns:
                    fig.add_trace(go.Scatter(
                        x=company_data['Date'],
                        y=company_data[metric],
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title=f'Chỉ số tài chính {company} theo thời gian',
                xaxis_title='Thời gian',
                yaxis_title='Giá trị',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bảng dữ liệu
            st.subheader("📋 Dữ liệu chi tiết")
            display_data = company_data[['Date'] + metrics[:5]].tail(10)
            st.dataframe(display_data, use_container_width=True)

def show_analysis(df, company, metrics):
    """Hiển thị phân tích"""
    st.header("📈 Phân tích Chi tiết")
    
    if company and metrics:
        company_data = df[df['Company'] == company]
        
        if not company_data.empty:
            # Phân tích xu hướng
            st.subheader("📊 Phân tích xu hướng")
            
            trend_data = []
            for metric in metrics[:3]:
                if metric in company_data.columns:
                    values = company_data[metric].dropna()
                    if len(values) > 1:
                        # Tính toán xu hướng
                        x = np.arange(len(values))
                        slope, intercept = np.polyfit(x, values, 1)
                        trend_direction = "📈 Tăng" if slope > 0 else "📉 Giảm"
                        
                        trend_data.append({
                            'Chỉ số': metric,
                            'Xu hướng': trend_direction,
                            'Độ dốc': f"{slope:.4f}",
                            'Giá trị đầu': f"{values.iloc[0]:.2f}",
                            'Giá trị cuối': f"{values.iloc[-1]:.2f}",
                            'Thay đổi (%)': f"{((values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100):.2f}%"
                        })
            
            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                st.dataframe(trend_df, use_container_width=True)
            
            # Biểu đồ phân tích
            if len(metrics) >= 2:
                st.subheader("📊 Biểu đồ phân tích")
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Phân tích xu hướng', 'Phân bố giá trị', 'Tương quan', 'So sánh quý']
                )
                
                # Biểu đồ 1: Xu hướng
                if 'Date' in company_data.columns and len(metrics) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=company_data['Date'],
                            y=company_data[metrics[0]],
                            mode='lines+markers',
                            name=metrics[0]
                        ),
                        row=1, col=1
                    )
                
                # Biểu đồ 2: Histogram
                if len(metrics) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=company_data[metrics[0]],
                            name=metrics[0]
                        ),
                        row=1, col=2
                    )
                
                # Biểu đồ 3: Scatter plot
                if len(metrics) >= 2:
                    fig.add_trace(
                        go.Scatter(
                            x=company_data[metrics[0]],
                            y=company_data[metrics[1]],
                            mode='markers',
                            name=f'{metrics[0]} vs {metrics[1]}'
                        ),
                        row=2, col=1
                    )
                
                # Biểu đồ 4: Box plot
                if len(metrics) > 0:
                    fig.add_trace(
                        go.Box(
                            y=company_data[metrics[0]],
                            name=metrics[0]
                        ),
                        row=2, col=2
                    )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

def show_comparison(df, metrics):
    """Hiển thị so sánh"""
    st.header("🔍 So sánh Giữa Các Công Ty")
    
    if metrics:
        # So sánh chỉ số giữa các công ty
        st.subheader("📊 So sánh chỉ số tài chính")
        
        companies = df['Company'].unique()
        
        # Lấy giá trị mới nhất của mỗi công ty
        latest_data = []
        for company in companies:
            company_data = df[df['Company'] == company]
            if not company_data.empty:
                latest_row = company_data.iloc[-1]
                row_data = {'Company': company}
                for metric in metrics[:5]:  # Chỉ so sánh 5 chỉ số đầu
                    if metric in latest_row:
                        row_data[metric] = latest_row[metric]
                latest_data.append(row_data)
        
        if latest_data:
            comparison_df = pd.DataFrame(latest_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Biểu đồ so sánh
            if len(metrics) > 0:
                fig = go.Figure()
                
                for metric in metrics[:3]:
                    fig.add_trace(go.Bar(
                        x=comparison_df['Company'],
                        y=comparison_df[metric],
                        name=metric,
                        text=comparison_df[metric].round(2),
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title='So sánh chỉ số tài chính giữa các công ty',
                    barmode='group',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_statistics(df, metrics):
    """Hiển thị thống kê"""
    st.header("📋 Thống kê Mô tả")
    
    if metrics:
        # Thống kê mô tả
        st.subheader("📊 Thống kê các chỉ số tài chính")
        
        stats_df = df[metrics].describe()
        st.dataframe(stats_df, use_container_width=True)
        
        # Biểu đồ phân bố
        st.subheader("📈 Phân bố các chỉ số")
        
        if len(metrics) > 0:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[f'Phân bố {metric}' for metric in metrics[:4]]
            )
            
            for i, metric in enumerate(metrics[:4]):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                fig.add_trace(
                    go.Histogram(
                        x=df[metric],
                        name=metric,
                        nbinsx=20
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Hàm chính"""
    st.markdown('<h1 class="main-header">📊 FinSight AI</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Quản Lý và Dự Báo Tài Chính Doanh Nghiệp</h2>', unsafe_allow_html=True)
    
    # Tải dữ liệu
    df = load_data()
    
    if df is not None:
        show_dashboard(df)
    else:
        st.error("❌ Không thể tải dữ liệu. Vui lòng kiểm tra lại.")
        
        # Hướng dẫn tạo dữ liệu mẫu
        st.info("💡 Để tạo dữ liệu mẫu, chạy lệnh: `python create_sample_data.py`")

if __name__ == "__main__":
    main()
