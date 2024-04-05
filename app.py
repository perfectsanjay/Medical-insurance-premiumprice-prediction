from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('Medicalpremium.csv')
pipe = pickle.load(open('RandomForestRegressor.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    AGE = request.form.get('age')
    Diabetes = request.form.get('diabetes')
    BloodPressureProblems =request.form.get('blood_pressure_problems')
    AnyTransplants =request.form.get('any_transplants')
    AnyChronicDiseases =request.form.get('any_chronic_diseases')
    Height =request.form.get('height')
    Weight =request.form.get('weight')
    KnownAllergies =request.form.get('known_allergies')
    HistoryOfCancerInFamily =request.form.get('history_of_cancer_in_family')
    NumberOfMajorSurgeries =request.form.get('number_of_major_surgeries')
    
    print(AGE, Diabetes, BloodPressureProblems,AnyTransplants,AnyChronicDiseases,Height,Weight,KnownAllergies,HistoryOfCancerInFamily,NumberOfMajorSurgeries)
    input = pd.DataFrame([[AGE, Diabetes, BloodPressureProblems,AnyTransplants,AnyChronicDiseases,Height,Weight,KnownAllergies,HistoryOfCancerInFamily,NumberOfMajorSurgeries]], columns = ['Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants',
       'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies',
       'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries'])
    prediction = pipe.predict(input)[0]


    return str(prediction)

