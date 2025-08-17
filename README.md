# AI Quản Lý và Dự Báo Tài Chính Doanh Nghiệp

## Mô tả dự án
Dự án AI này cung cấp các công cụ phân tích và dự báo tài chính doanh nghiệp sử dụng machine learning và deep learning. Hệ thống có thể phân tích báo cáo tài chính, dự báo xu hướng và đưa ra các khuyến nghị đầu tư.

## Tính năng chính
- 📊 Phân tích báo cáo tài chính
- 🔮 Dự báo giá cổ phiếu và chỉ số tài chính
- 📈 Phân tích xu hướng và mô hình
- 🎯 Đánh giá rủi ro và hiệu suất
- 📱 Giao diện web thân thiện với Streamlit
- 🤖 Mô hình AI được huấn luyện trên dữ liệu thực

## Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd finsight-ai
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Cấu hình Kaggle API
- Tạo tài khoản Kaggle tại https://www.kaggle.com
- Tải file `kaggle.json` từ Settings > API
- Đặt file vào thư mục `~/.kaggle/` (Linux/Mac) hoặc `C:\Users\<username>\.kaggle\` (Windows)

### 5. Chạy ứng dụng
```bash
streamlit run app.py
```

## Cấu trúc dự án
```
finsight-ai/
├── app.py                 # Giao diện chính Streamlit
├── data/                  # Thư mục chứa dữ liệu
│   ├── raw/              # Dữ liệu thô từ Kaggle
│   └── processed/        # Dữ liệu đã xử lý
├── models/               # Mô hình đã huấn luyện
├── notebooks/            # Jupyter notebooks cho phân tích
├── src/                  # Mã nguồn chính
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── prediction.py
│   └── visualization.py
├── utils/                # Tiện ích
├── requirements.txt      # Dependencies
└── README.md
```

## Sử dụng

### 1. Tải dữ liệu
```bash
python src/data_download.py
```

### 2. Huấn luyện mô hình
```bash
python src/train_models.py
```

### 3. Chạy ứng dụng web
```bash
streamlit run app.py
```

## Mô hình AI được sử dụng
- **Linear Regression**: Dự báo giá cơ bản
- **Random Forest**: Phân tích mẫu và xu hướng
- **XGBoost**: Dự báo chính xác cao
- **LSTM**: Dự báo chuỗi thời gian
- **Transformer**: Phân tích văn bản báo cáo

## Dataset
Sử dụng dataset từ Kaggle: [Financial Statements of Major Companies (2009-2023)](https://www.kaggle.com/datasets/rish59/financial-statements-of-major-companies2009-2023)

## Đóng góp
Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## License
MIT License
