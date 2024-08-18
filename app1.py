from flask import Flask, request, jsonify, render_template
import pandas as pd
from joblib import load
from datetime import datetime, timedelta

app = Flask(__name__)
model_wrapper = load('random_forest_model.joblib')

feature_columns = [
    'CycleWithPeakorNot', 'ReproductiveCategory', 'LengthofCycle', 'EstimatedDayofOvulation', 'LengthofLutealPhase',
    'FirstDayofHigh', 'TotalHighPostPeak', 'TotalNumberofPeakDays', 'LengthofMenses', 'TotalMensesScore',
    'MeanBleedingIntensity', 'IntercourseInFertileWindow', 'UnusualBleeding', 'Age', 'Height', 'Weight', 
    'Numberpreg', 'Miscarriages', 'Medvits', 'Gynosurgeries', 'Urosurgeries'
]


counter = 1

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    global counter
    try:
        
        form_data = request.json
        print("Received data:", form_data) 

       
        input_df = pd.DataFrame([form_data])
        for col in feature_columns:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        
        if 'LengthofLutealPhase' not in input_df.columns or pd.isna(input_df['LengthofLutealPhase']).any():
            input_df['LengthofLutealPhase'] = input_df['LengthofCycle'] - input_df['EstimatedDayofOvulation']
        if 'TotalHighPostPeak' not in input_df.columns or pd.isna(input_df['TotalHighPostPeak']).any():
            input_df['TotalHighPostPeak'] = input_df['TotalNumberofPeakDays'] - (input_df['LengthofLutealPhase'] / input_df['LengthofMenses'])
        if 'TotalMensesScore' not in input_df.columns or pd.isna(input_df['TotalMensesScore']).any():
            input_df['TotalMensesScore'] = input_df['LengthofMenses'] * 2
        if 'MeanBleedingIntensity' not in input_df.columns or pd.isna(input_df['MeanBleedingIntensity']).any():
            input_df['MeanBleedingIntensity'] = input_df['EstimatedDayofOvulation'] / input_df['TotalNumberofPeakDays']

        
        missing_cols = set(feature_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0

        input_df = input_df[feature_columns]

       
        input_df = input_df.astype({
            'CycleWithPeakorNot': int, 'ReproductiveCategory': int, 'LengthofCycle': int, 'EstimatedDayofOvulation': int, 
            'LengthofLutealPhase': int, 'FirstDayofHigh': int, 'TotalHighPostPeak': int, 'TotalNumberofPeakDays': int, 
            'LengthofMenses': int, 'TotalMensesScore': int, 'MeanBleedingIntensity': float, 'IntercourseInFertileWindow': int, 
            'UnusualBleeding': int, 'Age': int, 'Height': int, 'Weight': int, 'Numberpreg': int, 'Miscarriages': int, 
            'Medvits': int, 'Gynosurgeries': int, 'Urosurgeries': int
        })

       
        pred = model_wrapper.predict(input_df)

        
        start_date = datetime.strptime(form_data['start_date'], "%Y-%m-%d")

        
        next_month_start_date = (start_date + timedelta(days=int(input_df['LengthofCycle'].iloc[0]))).strftime("%Y-%m-%d")
        present_month_end_date = (start_date + timedelta(days=int(pred[0]) - 8)).strftime("%Y-%m-%d")
        average_cycle_length = (int(input_df['LengthofCycle'].iloc[0]) + int(pred[0]) - 8) // 2
        fertility_rate = (input_df['TotalHighPostPeak'].iloc[0] + input_df['TotalNumberofPeakDays'].iloc[0]) / input_df['FirstDayofHigh'].iloc[0]
        disease = 1 if (input_df['Gynosurgeries'].iloc[0] + input_df['Urosurgeries'].iloc[0] + input_df['Miscarriages'].iloc[0] + input_df['Medvits'].iloc[0]) >= 3 else 0
        date_of_high_fertility = (start_date + timedelta(days=int(input_df['FirstDayofHigh'].iloc[0]))).strftime("%Y-%m-%d")
        total_days_of_pain = int(input_df['LengthofMenses'].iloc[0]) - int(input_df['TotalNumberofPeakDays'].iloc[0])

        response = {
            'ID': counter,
            'nextMonthStartDate': next_month_start_date,
            'presentMonthEndDate': present_month_end_date,
            'averageCycleLength': average_cycle_length,
            'fertilityRate': fertility_rate,
            'disease': disease,
            'dateofHighFertility': date_of_high_fertility,
            'totalDaysofPain': total_days_of_pain
        }

        
        counter += 1

        return jsonify(response)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
