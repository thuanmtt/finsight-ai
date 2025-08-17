"""
Ứng dụng AI Quản Lý và Dự Báo Tài Chính Doanh Nghiệp
Giao diện Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Cấu hình trang - PHẢI ĐỨNG ĐẦU TIÊN
st.set_page_config(
    page_title="FinSight AI - Quản Lý và Dự Báo Tài Chính",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm thư mục src vào path
sys.path.append('src')

# Import các module tự tạo
try:
    from data_processing import FinancialDataProcessor
    DATA_PROCESSING_AVAILABLE = True
except ImportError:
    DATA_PROCESSING_AVAILABLE = False
    st.warning("Module data_processing không khả dụng")

try:
    from feature_engineering import FinancialFeatureEngineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    st.warning("Module feature_engineering không khả dụng")

try:
    from model_training import FinancialModelTrainer
    MODEL_TRAINING_AVAILABLE = True
except ImportError:
    MODEL_TRAINING_AVAILABLE = False
    st.warning("Module model_training không khả dụng")

try:
    from prediction import FinancialPredictor
    PREDICTION_AVAILABLE = True
except ImportError:
    PREDICTION_AVAILABLE = False
    st.warning("Module prediction không khả dụng")

try:
    from visualization import FinancialVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    st.warning("Module visualization không khả dụng")

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
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
    .success-card {
        border-left-color: #2ca02c;
    }
    .warning-card {
        border-left-color: #ff7f0e;
    }
    .danger-card {
        border-left-color: #d62728;
    }
</style>
""", unsafe_allow_html=True)

class FinSightApp:
    """Lớp ứng dụng chính FinSight AI"""
    
    def __init__(self):
        # Khởi tạo các module nếu có sẵn
        self.data_processor = None
        self.feature_engineer = None
        self.model_trainer = None
        self.predictor = None
        self.visualizer = None
        
        if DATA_PROCESSING_AVAILABLE:
            self.data_processor = FinancialDataProcessor()
        
        if FEATURE_ENGINEERING_AVAILABLE:
            self.feature_engineer = FinancialFeatureEngineer()
        
        if MODEL_TRAINING_AVAILABLE:
            self.model_trainer = FinancialModelTrainer()
        
        if PREDICTION_AVAILABLE:
            self.predictor = FinancialPredictor()
        
        if VISUALIZATION_AVAILABLE:
            self.visualizer = FinancialVisualizer()
        
        self.data = None
        self.processed_data = None
        
    def load_data(self):
        """Tải dữ liệu"""
        try:
            # Kiểm tra dữ liệu đã xử lý
            processed_files = [f for f in os.listdir('data/processed') 
                             if f.endswith('_processed.csv')]
            
            if processed_files:
                # Tải dữ liệu đã xử lý
                file_path = os.path.join('data/processed', processed_files[0])
                self.processed_data = pd.read_csv(file_path)
                st.success(f"Đã tải dữ liệu đã xử lý: {len(self.processed_data)} bản ghi")
            else:
                # Tải dữ liệu thô
                raw_files = [f for f in os.listdir('data/raw') if f.endswith('.csv')]
                if raw_files:
                    file_path = os.path.join('data/raw', raw_files[0])
                    self.data = pd.read_csv(file_path)
                    st.info(f"Đã tải dữ liệu thô: {len(self.data)} bản ghi")
                else:
                    st.warning("Không tìm thấy dữ liệu. Vui lòng tải dữ liệu từ Kaggle trước.")
                    return False
            return True
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu: {str(e)}")
            return False
    
    def show_dashboard(self):
        """Hiển thị dashboard chính"""
        st.markdown('<h1 class="main-header">📊 FinSight AI</h1>', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; color: #666;">Quản Lý và Dự Báo Tài Chính Doanh Nghiệp</h2>', unsafe_allow_html=True)
        
        # Tải dữ liệu
        if not self.load_data():
            return
        
        # Sidebar
        with st.sidebar:
            st.header("🎛️ Điều khiển")
            
            # Chọn chức năng
            function = st.selectbox(
                "Chọn chức năng:",
                ["📊 Dashboard", "🤖 Huấn luyện AI", "🔮 Dự đoán", "📈 Phân tích", "⚙️ Cài đặt"]
            )
            
            # Chọn công ty nếu có dữ liệu
            if self.processed_data is not None and 'Company' in self.processed_data.columns:
                companies = self.processed_data['Company'].unique()
                selected_company = st.selectbox("Chọn công ty:", companies)
            else:
                selected_company = None
            
            # Chọn chỉ số
            if self.processed_data is not None:
                numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
                selected_metrics = st.multiselect(
                    "Chọn chỉ số tài chính:",
                    numeric_columns,
                    default=numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns
                )
            else:
                selected_metrics = []
        
        # Hiển thị chức năng được chọn
        if function == "📊 Dashboard":
            self.show_main_dashboard(selected_company, selected_metrics)
        elif function == "🤖 Huấn luyện AI":
            self.show_training_page()
        elif function == "🔮 Dự đoán":
            self.show_prediction_page(selected_company)
        elif function == "📈 Phân tích":
            self.show_analysis_page(selected_company, selected_metrics)
        elif function == "⚙️ Cài đặt":
            self.show_settings_page()
    
    def show_main_dashboard(self, company, metrics):
        """Hiển thị dashboard chính"""
        st.header("📊 Dashboard Tổng Quan")
        
        if self.processed_data is None:
            st.warning("Không có dữ liệu để hiển thị")
            return
        
        # Thống kê tổng quan
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tổng số công ty", len(self.processed_data['Company'].unique()))
        
        with col2:
            st.metric("Tổng số bản ghi", len(self.processed_data))
        
        with col3:
            if 'Date' in self.processed_data.columns:
                date_range = f"{self.processed_data['Date'].min()} - {self.processed_data['Date'].max()}"
                st.metric("Khoảng thời gian", date_range)
        
        with col4:
            if metrics:
                st.metric("Số chỉ số", len(metrics))
        
        # Biểu đồ chính
        if company and metrics:
            st.subheader(f"📈 Phân tích {company}")
            
            # Tạo biểu đồ chỉ số theo thời gian
            company_data = self.processed_data[self.processed_data['Company'] == company]
            
            if not company_data.empty and 'Date' in company_data.columns:
                if self.visualizer is not None:
                    try:
                        fig = self.visualizer.plot_financial_metrics_over_time(
                            company_data, metrics[:3]  # Chỉ hiển thị 3 chỉ số đầu
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Lỗi khi tạo biểu đồ: {str(e)}")
                else:
                    # Tạo biểu đồ đơn giản nếu không có visualizer
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    for metric in metrics[:3]:
                        if metric in company_data.columns:
                            fig.add_trace(go.Scatter(
                                x=company_data['Date'],
                                y=company_data[metric],
                                mode='lines+markers',
                                name=metric
                            ))
                    fig.update_layout(
                        title=f'Chỉ số tài chính {company}',
                        xaxis_title='Thời gian',
                        yaxis_title='Giá trị'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Dashboard tài chính
            if len(metrics) >= 2 and self.visualizer is not None:
                try:
                    dashboard_fig = self.visualizer.plot_financial_dashboard(
                        company_data, company, metrics[:2]
                    )
                    st.plotly_chart(dashboard_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi khi tạo dashboard: {str(e)}")
        
        # Ma trận tương quan
        if len(metrics) > 1:
            st.subheader("🔗 Ma trận tương quan")
            if self.visualizer is not None:
                try:
                    corr_fig = self.visualizer.plot_correlation_matrix(
                        self.processed_data, metrics
                    )
                    st.plotly_chart(corr_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi khi tạo ma trận tương quan: {str(e)}")
            else:
                # Tạo ma trận tương quan đơn giản
                import plotly.graph_objects as go
                import numpy as np
                
                corr_matrix = self.processed_data[metrics].corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0
                ))
                fig.update_layout(title='Ma trận tương quan')
                st.plotly_chart(fig, use_container_width=True)
    
    def show_training_page(self):
        """Hiển thị trang huấn luyện AI"""
        st.header("🤖 Huấn luyện Mô hình AI")
        
        if self.processed_data is None:
            st.warning("Không có dữ liệu để huấn luyện. Vui lòng tải dữ liệu trước.")
            return
        
        # Chọn target variable
        numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        target_variable = st.selectbox("Chọn biến mục tiêu:", numeric_columns)
        
        # Chọn loại mô hình
        model_types = st.multiselect(
            "Chọn loại mô hình:",
            ["Linear Models", "Ensemble Models", "Neural Network", "LSTM"],
            default=["Linear Models", "Ensemble Models"]
        )
        
        # Nút huấn luyện
        if st.button("🚀 Bắt đầu huấn luyện", type="primary"):
            with st.spinner("Đang huấn luyện mô hình..."):
                try:
                    # Xử lý dữ liệu trước khi training
                    st.info("Đang xử lý dữ liệu...")
                    
                    # Làm sạch dữ liệu
                    df_clean = self.processed_data.copy()
                    
                    # Xử lý các giá trị vô cùng và NaN
                    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        # Thay thế infinity bằng NaN
                        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                        # Thay thế NaN bằng median
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    
                    # Xử lý các cột không phải số
                    non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns
                    for col in non_numeric_cols:
                        df_clean[col] = df_clean[col].fillna('Unknown')
                    
                    # Loại bỏ các hàng có target variable là NaN
                    df_clean = df_clean.dropna(subset=[target_variable])
                    
                    st.info(f"Dữ liệu sau khi làm sạch: {len(df_clean)} hàng")
                    
                    # Tạo features
                    st.info("Đang tạo tính năng...")
                    if self.feature_engineer is not None:
                        df_features = self.feature_engineer.create_all_features(
                            df_clean, target_variable
                        )
                        # Lấy feature names từ feature engineer
                        feature_names = getattr(self.feature_engineer, 'feature_names', None)
                    else:
                        # Nếu không có feature engineer, sử dụng dữ liệu đã làm sạch
                        df_features = df_clean
                        feature_names = None
                    
                    # Huấn luyện mô hình
                    st.info("Đang huấn luyện mô hình...")
                    if self.model_trainer is not None:
                        results = self.model_trainer.train_all_models(df_features, target_variable, feature_names)
                    else:
                        st.error("Model trainer không khả dụng")
                        return
                    
                    # Lưu mô hình
                    if self.model_trainer is not None:
                        self.model_trainer.save_models()
                    
                    st.success("✅ Huấn luyện hoàn thành!")
                    
                    # Hiển thị kết quả
                    st.subheader("📊 Kết quả huấn luyện")
                    
                    # Bảng kết quả
                    results_df = pd.DataFrame([
                        {
                            'Mô hình': name,
                            'R² Score': f"{result['r2']:.4f}",
                            'MSE': f"{result['mse']:.4f}",
                            'MAE': f"{result['mae']:.4f}"
                        }
                        for name, result in results.items()
                    ])
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Biểu đồ so sánh hiệu suất
                    model_scores = {name: result['r2'] for name, result in results.items()}
                    if self.visualizer is not None:
                        try:
                            perf_fig = self.visualizer.plot_model_performance_comparison(model_scores)
                            st.plotly_chart(perf_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Lỗi khi tạo biểu đồ hiệu suất: {str(e)}")
                    else:
                        # Tạo biểu đồ đơn giản
                        import plotly.graph_objects as go
                        fig = go.Figure(data=[
                            go.Bar(x=list(model_scores.keys()), y=list(model_scores.values()))
                        ])
                        fig.update_layout(title='So sánh hiệu suất các mô hình')
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Lỗi trong quá trình huấn luyện: {str(e)}")
    
    def show_prediction_page(self, company):
        """Hiển thị trang dự đoán"""
        st.header("🔮 Dự đoán Tài chính")
        
        # Kiểm tra mô hình đã huấn luyện
        if not os.path.exists('models/model_info.joblib'):
            st.warning("Chưa có mô hình được huấn luyện. Vui lòng huấn luyện mô hình trước.")
            return
        
        # Tải lại predictor
        self.predictor = FinancialPredictor()
        
        if not self.predictor.models:
            st.error("Không thể tải mô hình. Vui lòng kiểm tra lại.")
            return
        
        # Chọn loại dự đoán
        prediction_type = st.selectbox(
            "Chọn loại dự đoán:",
            ["Dự đoán chỉ số tài chính", "Dự đoán giá cổ phiếu", "Khuyến nghị đầu tư"]
        )
        
        if prediction_type == "Dự đoán chỉ số tài chính":
            self.show_financial_metrics_prediction(company)
        elif prediction_type == "Dự đoán giá cổ phiếu":
            self.show_stock_price_prediction(company)
        elif prediction_type == "Khuyến nghị đầu tư":
            self.show_investment_recommendation(company)
    
    def show_financial_metrics_prediction(self, company):
        """Hiển thị dự đoán chỉ số tài chính"""
        st.subheader("📊 Dự đoán chỉ số tài chính")
        
        if self.processed_data is None:
            st.warning("Không có dữ liệu để dự đoán")
            return
        
        # Chọn công ty
        if company is None:
            companies = self.processed_data['Company'].unique()
            company = st.selectbox("Chọn công ty:", companies)
        
        # Chọn chỉ số cần dự đoán
        numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        target_metrics = st.multiselect(
            "Chọn chỉ số cần dự đoán:",
            numeric_columns,
            default=numeric_columns[:3]
        )
        
        if st.button("🔮 Thực hiện dự đoán", type="primary"):
            with st.spinner("Đang thực hiện dự đoán..."):
                try:
                    # Lấy dữ liệu công ty
                    company_data = self.processed_data[self.processed_data['Company'] == company]
                    
                    if company_data.empty:
                        st.error(f"Không tìm thấy dữ liệu cho công ty {company}")
                        return
                    
                    # Thực hiện dự đoán
                    predictions = self.predictor.predict_financial_metrics(
                        company_data, target_metrics
                    )
                    
                    # Hiển thị kết quả
                    st.success("✅ Dự đoán hoàn thành!")
                    
                    # Tạo bảng kết quả
                    results_data = []
                    for metric, result in predictions.items():
                        if result['predicted_value'] is not None:
                            results_data.append({
                                'Chỉ số': metric,
                                'Giá trị dự đoán': f"{result['predicted_value']:.2f}",
                                'Độ tin cậy': f"{result['confidence']:.2f}",
                                'Mô hình': result['model_used']
                            })
                    
                    if results_data:
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Biểu đồ so sánh
                        metrics = [r['Chỉ số'] for r in results_data]
                        values = [float(r['Giá trị dự đoán']) for r in results_data]
                        
                        fig = go.Figure(data=[
                            go.Bar(x=metrics, y=values, marker_color='lightblue')
                        ])
                        fig.update_layout(
                            title="Kết quả dự đoán các chỉ số tài chính",
                            xaxis_title="Chỉ số",
                            yaxis_title="Giá trị dự đoán"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Không có dự đoán nào thành công")
                        
                except Exception as e:
                    st.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
    
    def show_stock_price_prediction(self, company):
        """Hiển thị dự đoán giá cổ phiếu"""
        st.subheader("📈 Dự đoán giá cổ phiếu")
        
        # Số ngày dự đoán
        days_ahead = st.slider("Số ngày dự đoán:", 7, 90, 30)
        
        if st.button("🔮 Dự đoán giá cổ phiếu", type="primary"):
            with st.spinner("Đang dự đoán giá cổ phiếu..."):
                try:
                    # Tạo dữ liệu mẫu cho demo
                    sample_data = pd.DataFrame({
                        'Revenue': [5000],
                        'Total_Assets': [15000],
                        'Total_Equity': [8000],
                        'Net_Income': [500],
                        'ROE': [0.15],
                        'ROA': [0.08],
                        'Current_Ratio': [2.5],
                        'Debt_Ratio': [0.4],
                        'Close': [150]  # Giá cổ phiếu hiện tại
                    })
                    
                    # Thực hiện dự đoán
                    prediction = self.predictor.predict_stock_price(sample_data, days_ahead)
                    
                    # Hiển thị kết quả
                    st.success("✅ Dự đoán giá cổ phiếu hoàn thành!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Giá hiện tại dự đoán",
                            f"${prediction['current_price_prediction']:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Xu hướng",
                            prediction['trend'].title()
                        )
                    
                    with col3:
                        st.metric(
                            "Độ tin cậy",
                            f"{prediction['confidence']:.2f}"
                        )
                    
                    # Biểu đồ dự đoán
                    dates = pd.date_range(
                        start=datetime.now(),
                        periods=days_ahead + 1,
                        freq='D'
                    )
                    
                    fig = go.Figure()
                    
                    # Giá hiện tại
                    fig.add_trace(go.Scatter(
                        x=[dates[0]],
                        y=[prediction['current_price_prediction']],
                        mode='markers',
                        name='Giá hiện tại',
                        marker=dict(size=10, color='blue')
                    ))
                    
                    # Giá tương lai
                    fig.add_trace(go.Scatter(
                        x=dates[1:],
                        y=prediction['future_prices'],
                        mode='lines+markers',
                        name='Dự đoán',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Dự đoán giá cổ phiếu {company} trong {days_ahead} ngày tới",
                        xaxis_title="Thời gian",
                        yaxis_title="Giá cổ phiếu ($)",
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
    
    def show_investment_recommendation(self, company):
        """Hiển thị khuyến nghị đầu tư"""
        st.subheader("💡 Khuyến nghị đầu tư")
        
        if st.button("🎯 Tạo khuyến nghị", type="primary"):
            with st.spinner("Đang phân tích và tạo khuyến nghị..."):
                try:
                    # Tạo dữ liệu mẫu cho demo
                    sample_predictions = {
                        'ROE': {'predicted_value': 0.18},
                        'Current_Ratio': {'predicted_value': 2.2},
                        'ROA': {'predicted_value': 0.12}
                    }
                    
                    # Tạo khuyến nghị
                    recommendation = self.predictor.generate_investment_recommendation(
                        sample_predictions
                    )
                    
                    # Hiển thị kết quả
                    st.success("✅ Phân tích hoàn thành!")
                    
                    # Card khuyến nghị
                    action_color = {
                        'buy': 'success',
                        'sell': 'danger',
                        'hold': 'warning'
                    }.get(recommendation['action'], 'info')
                    
                    st.markdown(f"""
                    <div class="metric-card {action_color}-card">
                        <h3>Khuyến nghị: {recommendation['action'].upper()}</h3>
                        <p><strong>Độ tin cậy:</strong> {recommendation['confidence']:.2f}</p>
                        <p><strong>Mức độ rủi ro:</strong> {recommendation['risk_level']}</p>
                        <p><strong>Lý do:</strong> {', '.join(recommendation['reasoning'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Biểu đồ radar cho phân tích
                    if recommendation['reasoning']:
                        categories = ['ROE', 'ROA', 'Liquidity', 'Growth', 'Risk']
                        values = [0.8, 0.7, 0.6, 0.5, 0.4]  # Giá trị mẫu
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='Điểm đánh giá'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=False,
                            title="Phân tích đa chiều"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Lỗi trong quá trình phân tích: {str(e)}")
    
    def show_analysis_page(self, company, metrics):
        """Hiển thị trang phân tích"""
        st.header("📈 Phân tích Chi tiết")
        
        if self.processed_data is None:
            st.warning("Không có dữ liệu để phân tích")
            return
        
        # Chọn loại phân tích
        analysis_type = st.selectbox(
            "Chọn loại phân tích:",
            ["Phân tích xu hướng", "Phân tích rủi ro", "So sánh ngành", "Thống kê mô tả"]
        )
        
        if analysis_type == "Phân tích xu hướng":
            self.show_trend_analysis(company, metrics)
        elif analysis_type == "Phân tích rủi ro":
            self.show_risk_analysis(company, metrics)
        elif analysis_type == "So sánh ngành":
            self.show_industry_comparison(company, metrics)
        elif analysis_type == "Thống kê mô tả":
            self.show_descriptive_statistics(metrics)
    
    def show_trend_analysis(self, company, metrics):
        """Hiển thị phân tích xu hướng"""
        st.subheader("📈 Phân tích xu hướng")
        
        if company and metrics:
            company_data = self.processed_data[self.processed_data['Company'] == company]
            
            if not company_data.empty and 'Date' in company_data.columns:
                # Biểu đồ xu hướng
                if self.visualizer is not None:
                    try:
                        fig = self.visualizer.plot_financial_metrics_over_time(
                            company_data, metrics[:3]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Lỗi khi tạo biểu đồ xu hướng: {str(e)}")
                else:
                    # Tạo biểu đồ đơn giản
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    for metric in metrics[:3]:
                        if metric in company_data.columns:
                            fig.add_trace(go.Scatter(
                                x=company_data['Date'],
                                y=company_data[metric],
                                mode='lines+markers',
                                name=metric
                            ))
                    fig.update_layout(title='Phân tích xu hướng')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Phân tích xu hướng chi tiết
                st.subheader("📊 Thống kê xu hướng")
                
                trend_data = []
                for metric in metrics[:3]:
                    if metric in company_data.columns:
                        values = company_data[metric].dropna()
                        if len(values) > 1:
                            # Tính toán xu hướng
                            x = np.arange(len(values))
                            slope, intercept = np.polyfit(x, values, 1)
                            trend_direction = "Tăng" if slope > 0 else "Giảm"
                            
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
    
    def show_risk_analysis(self, company, metrics):
        """Hiển thị phân tích rủi ro"""
        st.subheader("⚠️ Phân tích rủi ro")
        
        # Chọn chỉ số rủi ro
        risk_metrics = st.multiselect(
            "Chọn chỉ số rủi ro:",
            metrics,
            default=metrics[:2] if len(metrics) >= 2 else metrics
        )
        
        if risk_metrics:
            if self.visualizer is not None:
                try:
                    fig = self.visualizer.plot_risk_analysis(
                        self.processed_data, risk_metrics
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi khi tạo biểu đồ rủi ro: {str(e)}")
            else:
                # Tạo biểu đồ đơn giản
                import plotly.graph_objects as go
                fig = go.Figure(data=[
                    go.Scatter(
                        x=self.processed_data[risk_metrics[0]],
                        y=self.processed_data[risk_metrics[1]] if len(risk_metrics) > 1 else self.processed_data[risk_metrics[0]],
                        mode='markers',
                        text=self.processed_data['Company']
                    )
                ])
                fig.update_layout(title='Phân tích rủi ro')
                st.plotly_chart(fig, use_container_width=True)
    
    def show_industry_comparison(self, company, metrics):
        """Hiển thị so sánh ngành"""
        st.subheader("🏭 So sánh với ngành")
        
        if company and metrics:
            if self.visualizer is not None:
                try:
                    fig = self.visualizer.plot_financial_ratios_comparison(
                        self.processed_data, metrics[:3]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi khi tạo biểu đồ so sánh: {str(e)}")
            else:
                # Tạo biểu đồ đơn giản
                import plotly.graph_objects as go
                companies = self.processed_data['Company'].unique()
                fig = go.Figure()
                for metric in metrics[:3]:
                    values = []
                    for company in companies:
                        company_data = self.processed_data[self.processed_data['Company'] == company]
                        if not company_data.empty:
                            values.append(company_data[metric].iloc[-1])
                        else:
                            values.append(0)
                    fig.add_trace(go.Bar(x=companies, y=values, name=metric))
                fig.update_layout(title='So sánh chỉ số tài chính')
                st.plotly_chart(fig, use_container_width=True)
    
    def show_descriptive_statistics(self, metrics):
        """Hiển thị thống kê mô tả"""
        st.subheader("📊 Thống kê mô tả")
        
        if metrics:
            if self.visualizer is not None:
                try:
                    fig = self.visualizer.create_summary_statistics_table(
                        self.processed_data, metrics
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi khi tạo bảng thống kê: {str(e)}")
            else:
                # Tạo bảng thống kê đơn giản
                stats_df = self.processed_data[metrics].describe()
                st.dataframe(stats_df, use_container_width=True)
    
    def show_settings_page(self):
        """Hiển thị trang cài đặt"""
        st.header("⚙️ Cài đặt")
        
        st.subheader("📥 Tải dữ liệu")
        
        if st.button("📥 Tải dữ liệu từ Kaggle", type="primary"):
            with st.spinner("Đang tải dữ liệu từ Kaggle..."):
                try:
                    # Import và chạy script tải dữ liệu
                    from data_download import download_kaggle_dataset
                    success = download_kaggle_dataset()
                    
                    if success:
                        st.success("✅ Tải dữ liệu thành công!")
                    else:
                        st.error("❌ Lỗi khi tải dữ liệu")
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
        
        st.subheader("🔄 Xử lý dữ liệu")
        
        if st.button("🔄 Xử lý dữ liệu", type="primary"):
            with st.spinner("Đang xử lý dữ liệu..."):
                try:
                    processed_data = self.data_processor.process_all_data()
                    st.success(f"✅ Xử lý dữ liệu thành công! {len(processed_data)} bộ dữ liệu")
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
        
        st.subheader("📊 Thông tin hệ thống")
        
        # Hiển thị thông tin mô hình
        if os.path.exists('models/model_info.joblib'):
            import joblib
            model_info = joblib.load('models/model_info.joblib')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mô hình tốt nhất", model_info.get('best_model', 'N/A'))
                st.metric("Điểm R² tốt nhất", f"{model_info.get('best_score', 0):.4f}")
            
            with col2:
                st.metric("Tổng số mô hình", model_info.get('total_models', 0))
        else:
            st.info("Chưa có mô hình được huấn luyện")

def main():
    """Hàm chính"""
    app = FinSightApp()
    app.show_dashboard()

if __name__ == "__main__":
    main()
