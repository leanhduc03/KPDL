import json
import streamlit as st
import joblib
import pandas as pd
from streamlit_lottie import st_lottie
from pygwalker.api.streamlit import StreamlitRenderer

# 1. Page config
st.set_page_config(
    page_title="🧊 Logistic Regression - Covid-19",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for dark mode & background image
st.markdown("""
    <style>
        html, body {
            background-color: #1e1e2f;
            color: #f0f0f0;
        }
        .stTabs [role="tablist"] {
            justify-content: center;
        }
        .stTabs [role="tab"] {
            flex-grow: 1;
            text-align: center;
            font-size: 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4a90e2 !important;
            color: white !important;
            border-radius: 6px;
        }
        .css-1aumxhk {
            color: white;
        }
        
        [data-testid="stVerticalBlockBorderWrapper"] {
            margin-bottom: 12px;
        }
 
    </style>
""", unsafe_allow_html=True)

# 3. Load Lottie Animation


# 4. Tabs
tab1, tab2 = st.tabs(['🏠 Trang chính', '📊 Trực quan dữ liệu'])

with tab1:
    st.title("🧪 Dự đoán tử vong do Covid-19 bằng Logistic Regression")

    st.markdown("Điền thông tin bên dưới:")

    col1, col2 = st.columns(2)
    with col1:
        USMER = st.selectbox('USMER', [1, 2], index=1)
        MEDICAL_UNIT = st.slider('Đơn vị y tế (1-13)', 1, 13, 1)
        PATIENT_TYPE = st.selectbox('Loại bệnh nhân', [1, 2])
        PNEUMONIA = st.selectbox('Viêm phổi', [1, 2])

    with col2:
        DIABETES = st.selectbox('Tiểu đường', [1, 2])
        with st.container():
            HIPERTENSION = st.selectbox('Tăng huyết áp', [1, 2])
        RENAL_CHRONIC = st.selectbox('Suy thận mạn', [1, 2])
        CLASIFFICATION_FINAL = st.selectbox('Phân loại cuối', [1, 2])
        
    AGE = st.slider('Tuổi bệnh nhân', 0, 120, 65)

    def predict_covid(
            USMER, MEDICAL_UNIT, PATIENT_TYPE, PNEUMONIA,
            AGE, DIABETES, HIPERTENSION, RENAL_CHRONIC, CLASIFFICATION_FINAL):

        new_sample = pd.DataFrame({
            "USMER": [USMER],
            "MEDICAL_UNIT": [MEDICAL_UNIT],
            "PATIENT_TYPE": [PATIENT_TYPE],
            "PNEUMONIA": [PNEUMONIA],
            "AGE": [AGE],
            "DIABETES": [DIABETES],
            "HIPERTENSION": [HIPERTENSION],
            "RENAL_CHRONIC": [RENAL_CHRONIC],
            "CLASIFFICATION_FINAL": [CLASIFFICATION_FINAL],
        })

        with open("../Artifacts/log_reg_covid1.sav", 'rb') as f:
            model = joblib.load(f)

        prediction = model.predict(new_sample)[0]
        return prediction

    if st.button('🚀 Dự đoán ngay', use_container_width=True):
        result = predict_covid(
            USMER, MEDICAL_UNIT, PATIENT_TYPE, PNEUMONIA,
            AGE, DIABETES, HIPERTENSION, RENAL_CHRONIC, CLASIFFICATION_FINAL
        )
        if result == 1:
            st.error("☠️ Kết quả: Tỷ lệ tử vong **CAO**")
        else:
            st.success("💪 Kết quả: Tỷ lệ tử vong **THẤP**")

with tab2:
    st.subheader("📊 Trực quan hóa dữ liệu bệnh nhân")

    df = pd.read_csv("Covid Data.csv")

    # Làm sạch dữ liệu
    cols = ['PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION',
            'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']
    for i in cols:
        df = df[(df[i] == 1) | (df[i] == 2)]

    df['DEATH'] = df['DATE_DIED'].apply(lambda x: 2 if x == "9999-99-99" else 1)
    df.drop(columns=["DATE_DIED"], inplace=True)
    df['PREGNANT'] = df['PREGNANT'].replace(97, 2)
    df = df[df['PREGNANT'].isin([1, 2])]
    df['CLASIFFICATION_FINAL'] = df['CLASIFFICATION_FINAL'].replace([1, 2, 3], 1)
    df['CLASIFFICATION_FINAL'] = df['CLASIFFICATION_FINAL'].replace([4, 5, 6, 7], 2)

    st.markdown("Dữ liệu bên dưới đã được xử lý và sẵn sàng cho việc khám phá:")

    pyg_app = StreamlitRenderer(df, spec='../Artifacts/streamlit.json')
    pyg_app.explorer()
