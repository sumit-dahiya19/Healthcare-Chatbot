import pickle
import pandas as pd
import numpy as np
import sys
import os
import re
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
# import tensorflow
from flask import Flask, render_template, url_for, flash, redirect, request, send_from_directory
from sklearn.preprocessing import StandardScaler

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from werkzeug.utils import secure_filename

app = Flask(__name__)

# model_heartdisease = pickle.load(open('heartdisease.pkl', 'rb'))
# model_liverdisease = pickle.load(open('liverdisease.pkl', 'rb'))
# model_cancer = pickle.load(open('breastcancer.pkl', 'rb'))

df=pd.read_csv('Dataset/df_pivoted_final, dataset1.csv',encoding='utf-8')
df1=pd.read_csv('Dataset/Training.csv')
df1=df1.groupby(['prognosis']).sum()
df1=pd.DataFrame(df1)
lst=[]
for i in df.columns:
    lst.append(i)
for i in df1.columns:
    lst.append(i)
lst.remove('Unnamed: 0')
lst.remove('Source')
#lst.index('itching') 2nd dataset
df1['Source']=df1.index
df1=df1.reset_index()
df1.drop(['prognosis'],axis=1,inplace=True)
df.reset_index()
df.drop(['Unnamed: 0'],axis=1,inplace=True)
final=pd.merge(df,df1,how="outer")
final=final.fillna(0)

y=final['Source']
X=final.drop(['Source'],axis=1)
cols=X.columns

mnb = MultinomialNB()
mnb = mnb.fit(X, y)
feature_dict = {}
for i,f in enumerate(cols):
    feature_dict[f] = i

lst = lst[1:]

saved_model=pickle.dumps(mnb)
mnb_from_pickle = pickle.loads(saved_model)

@app.route('/')
def index():
	return render_template("index.html",List = lst)

@app.route("/predict", methods = ['POST'])
def predict():
    op1 = request.form['op1']
    op2 = request.form['op2']
    op3 = request.form['op3']
    s = []
    s.append(str(op1))
    s.append(str(op2))
    s.append(str(op3))
    a=[]
    for i in s:
        a.append(feature_dict[i])
    sample_x=[]
    for i in range(len(cols)):
        if (i==a[0] or i==a[1] or i==a[2] ):
            sample_x.append(1)
        else:
            sample_x.append(0)
    sample_x = np.array(sample_x).reshape(1,len(sample_x))
    ans = str(mnb.predict(sample_x))
    ans = ans[2:]
    ans = ans[:-2]
    return render_template("result.html",Disease = ans)


# Heart Disease

df=pd.read_csv('Dataset/heart_disease.csv')
X=df.drop(['target', 'fbs', 'chol'], axis=1)
#X=df.drop('target', axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.ensemble import BaggingClassifier
bagg = BaggingClassifier(n_estimators=100, random_state=42)
bagg.fit(X_train, y_train)

saved_model = pickle.dumps(bagg)
bagg_from_pickle = pickle.loads(saved_model)

@app.route("/heartdisease", methods = ['GET', 'POST'])
def heartdisease():
    if(request.method == 'GET'):
        return render_template("Heart_Disease.html")
    else:
        Age=int(request.form['age'])
        Gender=int(request.form['sex'])
        ChestPain= int(request.form['cp'])
        BloodPressure= int(request.form['trestbps'])
        ElectrocardiographicResults= int(request.form['restecg'])
        MaxHeartRate= int(request.form['thalach'])
        ExerciseInducedAngina= int(request.form['exang'])
        STdepression= float(request.form['oldpeak'])
        ExercisePeakSlope= int(request.form['slope'])
        MajorVesselsNo= int(request.form['ca'])
        Thalassemia=int(request.form['thal'])
        
        prediction = bagg.predict([[Age, Gender, ChestPain, BloodPressure, ElectrocardiographicResults, MaxHeartRate, ExerciseInducedAngina, STdepression, ExercisePeakSlope, MajorVesselsNo, Thalassemia]])

        if prediction==1:
            return render_template('Heart_Disease_Prediction.html', prediction_text="Oops! You seem to have a Heart Disease.")
        else:
            return render_template('Heart_Disease_Prediction.html', prediction_text="Great! You don't have any Heart Disease.")



# Liver Disease

df=pd.read_csv('Dataset/liver_disease.csv')
df['Disease']=df['Disease'].apply(lambda x: 0 if x==2 else 1)
df['Gender']=df['Gender'].apply(lambda x: 1 if x=='Female' else 0)
X=df.iloc[:, :-1]
y=df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
rf= RandomForestClassifier(n_estimators=50,random_state=42)
rf.fit(X_train, y_train)

saved_model = pickle.dumps(rf)
rf_from_pickle = pickle.loads(saved_model)

@app.route("/liverdisease", methods = ['GET', 'POST'])
def liverdisease():
    if(request.method == 'GET'):
        return render_template("Liver_Disease.html")
    else:
        Age=int(request.form['Age'])
        Gender=int(request.form['Gender'])
        Total_Bilirubin= float(request.form['Total_Bilirubin'])
        Direct_Bilirubin= float(request.form['Direct_Bilirubin'])
        Alkaline_Phosphotase= int(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase= int(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase= int(request.form['Aspartate_Aminotransferase'])
        Total_Protiens= float(request.form['Total_Protiens'])
        Albumin= float(request.form['Albumin'])
        Albumin_and_Globulin_Ratio= float(request.form['Albumin_and_Globulin_Ratio'])

        prediction = rf.predict([[Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]])
        
        if prediction==1:
            return render_template('Liver_Disease_Prediction.html', prediction_text="Oops! You seem to have Liver Disease.")
        else:
            return render_template('Liver_Disease_Prediction.html', prediction_text="Great! You don't have any Liver Disease.")
        



# Breast Cancer

df=pd.read_csv('Dataset/BreastCancer_data.csv')
df=pd.get_dummies(data=df, columns=['diagnosis'], drop_first=True)
df=df.drop(['id'], axis=1)
X=df.drop(['diagnosis_M', 'fractal_dimension_mean', 'texture_se', 'smoothness_se', 'symmetry_se', 'fractal_dimension_se', 'radius_mean', 'area_mean', 'radius_worst',
       'perimeter_worst', 'area_worst', 'perimeter_se', 'area_se'], axis=1)
y=df['diagnosis_M']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import BaggingClassifier
bagcl = BaggingClassifier(n_estimators=100, random_state=42)
bagcl.fit(X_train, y_train)
saved_model = pickle.dumps(bagcl)
bagg_from_pickle = pickle.loads(saved_model)

@app.route("/breastcancer", methods = ['GET', 'POST'])
def breastcancer():
    if(request.method == 'GET'):
        return render_template("Breast_Cancer.html")
    else:
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        radius_se = float(request.form['radius_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave_points_se'])
        texture_worst = float(request.form['texture_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

        prediction = bagcl.predict([[texture_mean, perimeter_mean, smoothness_mean, compactness_mean,
           concavity_mean, concave_points_mean, symmetry_mean, radius_se,
           compactness_se, concavity_se, concave_points_se, texture_worst,
           smoothness_worst, compactness_worst, concavity_worst,
           concave_points_worst, symmetry_worst, fractal_dimension_worst]])
        if prediction==1:
            return render_template('Breast_Cancer_Prediction.html', prediction_text="Oops! The tumor is malignant.")
        else:
            return render_template('Breast_Cancer_Prediction.html', prediction_text="Great! The tumor is benign.")



if __name__ == "__main__":
	app.run()
