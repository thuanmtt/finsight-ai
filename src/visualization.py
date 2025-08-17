"""
Module trực quan hóa dữ liệu và kết quả dự đoán tài chính
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Cấu hình style cho matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialVisualizer:
    """Lớp trực quan hóa dữ liệu tài chính"""
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def plot_financial_metrics_over_time(self, df: pd.DataFrame, 
                                       metrics: List[str], 
                                       company_col: str = 'Company',
                                       date_col: str = 'Date') -> go.Figure:
        """Vẽ biểu đồ các chỉ số tài chính theo thời gian"""
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics, 1):
            if metric in df.columns:
                # Lấy dữ liệu cho từng công ty
                companies = df[company_col].unique()
                
                for company in companies:
                    company_data = df[df[company_col] == company]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=company_data[date_col],
                            y=company_data[metric],
                            mode='lines+markers',
                            name=f'{company} - {metric}',
                            line=dict(width=2),
                            marker=dict(size=6)
                        ),
                        row=i, col=1
                    )
        
        fig.update_layout(
            title=f'Chỉ số tài chính theo thời gian',
            height=300 * len(metrics),
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                              numeric_columns: Optional[List[str]] = None) -> go.Figure:
        """Vẽ ma trận tương quan"""
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Tính ma trận tương quan
        corr_matrix = df[numeric_columns].corr()
        
        # Tạo heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Ma trận tương quan các chỉ số tài chính',
            width=800,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def plot_financial_ratios_comparison(self, df: pd.DataFrame, 
                                       ratios: List[str],
                                       company_col: str = 'Company') -> go.Figure:
        """Vẽ biểu đồ so sánh tỷ lệ tài chính giữa các công ty"""
        fig = go.Figure()
        
        companies = df[company_col].unique()
        
        for i, ratio in enumerate(ratios):
            if ratio in df.columns:
                values = []
                labels = []
                
                for company in companies:
                    company_data = df[df[company_col] == company]
                    if not company_data.empty:
                        values.append(company_data[ratio].iloc[-1])  # Lấy giá trị mới nhất
                        labels.append(company)
                
                fig.add_trace(go.Bar(
                    name=ratio,
                    x=labels,
                    y=values,
                    text=[f'{v:.2f}' for v in values],
                    textposition='auto',
                ))
        
        fig.update_layout(
            title='So sánh tỷ lệ tài chính giữa các công ty',
            barmode='group',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_prediction_vs_actual(self, actual: np.ndarray, 
                                predicted: np.ndarray,
                                model_name: str = 'Model') -> go.Figure:
        """Vẽ biểu đồ so sánh giá trị thực tế và dự đoán"""
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Dự đoán vs Thực tế',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.6
            )
        ))
        
        # Đường chéo (y=x)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Đường chéo (y=x)',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'So sánh dự đoán và thực tế - {model_name}',
            xaxis_title='Giá trị thực tế',
            yaxis_title='Giá trị dự đoán',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_model_performance_comparison(self, model_scores: Dict[str, float]) -> go.Figure:
        """Vẽ biểu đồ so sánh hiệu suất các mô hình"""
        models = list(model_scores.keys())
        scores = list(model_scores.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=scores,
                text=[f'{score:.3f}' for score in scores],
                textposition='auto',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title='So sánh hiệu suất các mô hình AI',
            xaxis_title='Mô hình',
            yaxis_title='R² Score',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_stock_price_prediction(self, historical_prices: np.ndarray,
                                  predicted_prices: np.ndarray,
                                  dates: np.ndarray,
                                  confidence_interval: Optional[Tuple] = None) -> go.Figure:
        """Vẽ biểu đồ dự đoán giá cổ phiếu"""
        fig = go.Figure()
        
        # Dữ liệu lịch sử
        fig.add_trace(go.Scatter(
            x=dates[:len(historical_prices)],
            y=historical_prices,
            mode='lines',
            name='Giá lịch sử',
            line=dict(color='blue', width=2)
        ))
        
        # Dự đoán
        future_dates = dates[len(historical_prices):]
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predicted_prices,
            mode='lines',
            name='Dự đoán',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Khoảng tin cậy nếu có
        if confidence_interval:
            lower_bound, upper_bound = confidence_interval
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=upper_bound,
                mode='lines',
                name='Khoảng tin cậy trên',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=lower_bound,
                mode='lines',
                name='Khoảng tin cậy dưới',
                line=dict(color='gray', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.2)',
                showlegend=False
            ))
        
        fig.update_layout(
            title='Dự đoán giá cổ phiếu',
            xaxis_title='Thời gian',
            yaxis_title='Giá cổ phiếu',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              top_n: int = 20) -> go.Figure:
        """Vẽ biểu đồ độ quan trọng của các tính năng"""
        # Sắp xếp theo độ quan trọng
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        
        features = [item[0] for item in sorted_features]
        importance_scores = [item[1] for item in sorted_features]
        
        fig = go.Figure(data=[
            go.Bar(
                x=importance_scores,
                y=features,
                orientation='h',
                marker_color='lightgreen'
            )
        ])
        
        fig.update_layout(
            title=f'Top {top_n} tính năng quan trọng nhất',
            xaxis_title='Độ quan trọng',
            yaxis_title='Tính năng',
            template='plotly_white',
            height=max(400, len(features) * 20)
        )
        
        return fig
    
    def plot_financial_dashboard(self, df: pd.DataFrame, 
                               company: str,
                               metrics: List[str]) -> go.Figure:
        """Tạo dashboard tài chính tổng hợp"""
        company_data = df[df['Company'] == company]
        
        if company_data.empty:
            raise ValueError(f"Không tìm thấy dữ liệu cho công ty {company}")
        
        # Tạo subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'{metric} theo thời gian' for metric in metrics[:2]
            ] + [
                'Phân tích xu hướng',
                'So sánh với ngành'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Biểu đồ 1: Metric 1 theo thời gian
        if len(metrics) > 0 and metrics[0] in company_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=company_data['Date'],
                    y=company_data[metrics[0]],
                    mode='lines+markers',
                    name=metrics[0]
                ),
                row=1, col=1
            )
        
        # Biểu đồ 2: Metric 2 theo thời gian
        if len(metrics) > 1 and metrics[1] in company_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=company_data['Date'],
                    y=company_data[metrics[1]],
                    mode='lines+markers',
                    name=metrics[1]
                ),
                row=1, col=2
            )
        
        # Biểu đồ 3: Phân tích xu hướng
        if 'Revenue' in company_data.columns:
            revenue = company_data['Revenue'].values
            trend = np.polyfit(range(len(revenue)), revenue, 1)
            trend_line = np.polyval(trend, range(len(revenue)))
            
            fig.add_trace(
                go.Scatter(
                    x=company_data['Date'],
                    y=revenue,
                    mode='lines+markers',
                    name='Revenue thực tế'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=company_data['Date'],
                    y=trend_line,
                    mode='lines',
                    name='Xu hướng',
                    line=dict(dash='dash')
                ),
                row=2, col=1
            )
        
        # Biểu đồ 4: So sánh với ngành
        if 'ROE' in company_data.columns:
            industry_avg = df['ROE'].mean()
            company_roe = company_data['ROE'].iloc[-1]
            
            fig.add_trace(
                go.Bar(
                    x=['Công ty', 'Trung bình ngành'],
                    y=[company_roe, industry_avg],
                    name='ROE'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'Dashboard tài chính - {company}',
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def plot_risk_analysis(self, df: pd.DataFrame, 
                          risk_metrics: List[str]) -> go.Figure:
        """Vẽ biểu đồ phân tích rủi ro"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Phân tích rủi ro', 'Phân bố rủi ro'],
            specs=[[{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Scatter plot rủi ro
        if len(risk_metrics) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=df[risk_metrics[0]],
                    y=df[risk_metrics[1]],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=df[risk_metrics[0]],
                        colorscale='RdYlGn',
                        showscale=True
                    ),
                    text=df['Company'],
                    name='Rủi ro'
                ),
                row=1, col=1
            )
        
        # Histogram phân bố rủi ro
        if risk_metrics:
            fig.add_trace(
                go.Histogram(
                    x=df[risk_metrics[0]],
                    nbinsx=20,
                    name='Phân bố rủi ro'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Phân tích rủi ro tài chính',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_summary_statistics_table(self, df: pd.DataFrame, 
                                      numeric_columns: List[str]) -> go.Figure:
        """Tạo bảng thống kê tóm tắt"""
        # Tính toán thống kê
        stats = df[numeric_columns].describe()
        
        # Tạo bảng
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Thống kê'] + list(stats.columns),
                fill_color='lightblue',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[stats.index] + [stats[col].round(2) for col in stats.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title='Thống kê tóm tắt các chỉ số tài chính',
            height=400
        )
        
        return fig

def main():
    """Hàm chính để test visualization"""
    # Tạo dữ liệu mẫu
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='Q')
    companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    sample_data = []
    for company in companies:
        for date in dates:
            sample_data.append({
                'Company': company,
                'Date': date,
                'Revenue': np.random.uniform(1000, 5000),
                'Net_Income': np.random.uniform(100, 500),
                'ROE': np.random.uniform(0.05, 0.25),
                'ROA': np.random.uniform(0.03, 0.15),
                'Current_Ratio': np.random.uniform(1.0, 3.0),
                'Debt_Ratio': np.random.uniform(0.2, 0.8)
            })
    
    df = pd.DataFrame(sample_data)
    
    # Khởi tạo visualizer
    viz = FinancialVisualizer()
    
    # Tạo các biểu đồ
    print("Tạo biểu đồ chỉ số tài chính...")
    fig1 = viz.plot_financial_metrics_over_time(df, ['Revenue', 'Net_Income'])
    
    print("Tạo ma trận tương quan...")
    fig2 = viz.plot_correlation_matrix(df)
    
    print("Tạo dashboard tài chính...")
    fig3 = viz.plot_financial_dashboard(df, 'AAPL', ['Revenue', 'Net_Income'])
    
    print("Hoàn thành tạo biểu đồ!")

if __name__ == "__main__":
    main()
