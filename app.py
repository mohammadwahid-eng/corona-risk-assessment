from flask import Flask, jsonify, request
import os.path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

@app.route("/")
def home():
	return jsonify({
        "model_creation": {
            "url": request.base_url + "model-creation",
            "method": "GET"
        },
        "prediction": {
            "url": request.base_url + "prediction",
            "method": "GET",
            "parameters": "fever, tiredness, dry_cough, difficulty_in_breathing, sore_throat, pains, nasal_congestion, runny_nose, diarrhea, contact_patient, age, gender"
        },
    }), 200

@app.route("/model-creation")
def model_creation():
    data = pd.read_csv("dataset.csv")

    ###Data Preprocessing Start###

    #Remove unnecessary columns
    data.drop("None_Experiencing", axis=1, inplace=True)
    data.drop("None_Sympton", axis=1, inplace=True)
    data.drop("Country", axis=1, inplace=True)

    #Columns name normalization
    data.rename(columns={'Fever': 'fever', 'Tiredness': 'tiredness', 'Dry-Cough': 'dry_cough', 'Difficulty-in-Breathing': 'difficulty_in_breathing', 'Sore-Throat': 'sore_throat', 'Pains': 'pains', 'Nasal-Congestion': 'nasal_congestion', 'Runny-Nose': 'runny_nose', 'Diarrhea': 'diarrhea'}, inplace=True)

    #Severity
    severity_columns = data.filter(like='Severity_').columns
    data.loc[ data['Severity_Mild'] == 1 , 'severity_level'] = 1
    data.loc[ data['Severity_Moderate'] == 1 , 'severity_level'] = 2
    data.loc[ data['Severity_None'] == 1 , 'severity_level'] = 3
    data.loc[ data['Severity_Severe'] == 1 , 'severity_level'] = 4
    data['severity_level'] = data['severity_level'].astype("int64")
    data.drop(severity_columns, axis=1, inplace=True)

    #Contact
    contact_columns = data.filter(like='Contact_').columns
    data.loc[ data['Contact_Dont-Know'] == 1 , 'contact_patient'] = 1
    data.loc[ data['Contact_Yes'] == 1 , 'contact_patient'] = 2
    data.loc[ data['Contact_No'] == 1 , 'contact_patient'] = 3
    data['contact_patient'] = data['contact_patient'].astype("int64")
    data.drop(contact_columns, axis=1, inplace=True)

    #Age
    age_columns = data.filter(like='Age_').columns
    data.loc[ data['Age_0-9'] == 1 , 'age'] = 1
    data.loc[ data['Age_10-19'] == 1 , 'age'] = 2
    data.loc[ data['Age_20-24'] == 1 , 'age'] = 3
    data.loc[ data['Age_25-59'] == 1 , 'age'] = 4
    data.loc[ data['Age_60+'] == 1 , 'age'] = 5
    data['age'] = data['age'].astype("int64")
    data.drop(age_columns, axis=1, inplace=True)

    #Gender
    gender_columns = data.filter(like='Gender_').columns
    data.loc[ data['Gender_Female'] == 1 , 'gender'] = 1
    data.loc[ data['Gender_Male'] == 1 , 'gender'] = 2
    data.loc[ data['Gender_Transgender'] == 1 , 'gender'] = 3
    data['gender'] = data['gender'].astype("int64")
    data.drop(gender_columns, axis=1, inplace=True)

    ###Data Preprocessing End###

    #Model Creation
    x = data.drop('severity_level', axis=1)
    y = data['severity_level']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.25)
    model = GaussianNB()
    model.fit(x_train, y_train)
    pickle.dump(model, open("model.pkl", "wb"))

    return jsonify({
        "classifier": "Gaussian Naive Bayes",
        "status": "Model has created successfully",
        "score": model.score(x, y)
    }), 200


@app.route("/prediction", methods=['GET'])
def prediction():
    if(os.path.isfile('model.pkl') == False):
        model_creation()

    fever = int(request.args.get('fever'))
    tiredness = int(request.args.get('tiredness'))
    dry_cough = int(request.args.get('dry_cough'))
    difficulty_in_breathing = int(request.args.get('difficulty_in_breathing'))
    sore_throat = int(request.args.get('sore_throat'))
    pains = int(request.args.get('pains'))
    nasal_congestion = int(request.args.get('nasal_congestion'))
    runny_nose = int(request.args.get('runny_nose'))
    diarrhea = int(request.args.get('diarrhea'))
    contact_patient = int(request.args.get('contact_patient'))
    age = int(request.args.get('age'))
    gender = int(request.args.get('gender'))

    args = np.array([fever, tiredness, dry_cough, difficulty_in_breathing, sore_throat, pains, nasal_congestion, runny_nose, diarrhea, contact_patient, age, gender])
    model = pickle.load(open("model.pkl", "rb"))
    predict = model.predict([args])

    if(predict[0]==1):
        result = "Mild"
    elif(predict[0]==2):
        result = "Moderate"
    elif(predict[0]==3):
        result = "None"
    else:
        result = "Severe"

    return jsonify({"result": result}), 200

if __name__ == '__main__':
    app.run(debug=True)