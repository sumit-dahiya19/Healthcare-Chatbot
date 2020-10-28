import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template,request
app = Flask(__name__)

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


if __name__ == "__main__":
	app.run()
