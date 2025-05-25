import os
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import joblib
import pandas as pd
import plotly.express as px

# Khởi tạo Flask app với đường dẫn đúng cho thư mục static
app = Flask(__name__, static_folder='static')

# Đảm bảo cấu hình phục vụ file tĩnh
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Hàm dự đoán Covid-19
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

    with open("../Artifacts/grand_boost_covid.sav", 'rb') as f:
        model = joblib.load(f)

    prediction = model.predict(new_sample)[0]
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    # Thiết lập giá trị mặc định
    form_data = {
        'USMER': 1,
        'MEDICAL_UNIT': 1,
        'PATIENT_TYPE': 1,
        'PNEUMONIA': 1,
        'AGE': 55,
        'DIABETES': 1,
        'HIPERTENSION': 1,
        'RENAL_CHRONIC': 1,
        'CLASIFFICATION_FINAL': 1
    }
    
    result = None
    prediction_text = ""
    
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        form_data['USMER'] = int(request.form.get('USMER'))
        form_data['MEDICAL_UNIT'] = int(request.form.get('MEDICAL_UNIT'))
        form_data['PATIENT_TYPE'] = int(request.form.get('PATIENT_TYPE'))
        form_data['PNEUMONIA'] = int(request.form.get('PNEUMONIA'))
        form_data['AGE'] = int(request.form.get('AGE'))
        form_data['DIABETES'] = int(request.form.get('DIABETES'))
        form_data['HIPERTENSION'] = int(request.form.get('HIPERTENSION'))
        form_data['RENAL_CHRONIC'] = int(request.form.get('RENAL_CHRONIC'))
        form_data['CLASIFFICATION_FINAL'] = int(request.form.get('CLASIFFICATION_FINAL'))
        
        # Dự đoán
        result = predict_covid(
            form_data['USMER'], form_data['MEDICAL_UNIT'], form_data['PATIENT_TYPE'], 
            form_data['PNEUMONIA'], form_data['AGE'], form_data['DIABETES'], 
            form_data['HIPERTENSION'], form_data['RENAL_CHRONIC'], form_data['CLASIFFICATION_FINAL']
        )
        
        if result == 1:
            prediction_text = "Result: High mortality rate"
        else:
            prediction_text = "Result: Low mortality rate"
    
    return render_template('index.html', result=result, prediction_text=prediction_text, form_data=form_data)

@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    df = pd.read_csv("Covid Data.csv")

    # Làm sạch như trước
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
    
    selected_var = request.form.get('variable') if request.method == 'POST' else 'AGE'
    
    import plotly.express as px
    fig = px.histogram(df, x=selected_var, color="DEATH", barmode="group")
    chart_html = fig.to_html(full_html=False)
    
    # Chuyển DataFrame thành dạng JSON để hiển thị
    data_json = df.head(100).to_json(orient='records')
    
    return render_template('visualization.html', data_json=data_json, chart_html=chart_html, selected_var=selected_var)

if __name__ == '__main__':
    app.run(debug=True)
