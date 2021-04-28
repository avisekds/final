from flask import Flask,render_template,request
from flask_material import Material

import numpy as np

import joblib
import os

app = Flask(__name__)
Material(app)

picsFolder = os.path.join('static','assets')
app.config['UPLOAD_FOLDER'] = picsFolder

pic1 = os.path.join(app.config['UPLOAD_FOLDER'],'undraw_different_love_a3rg.svg')
cancerpic = os.path.join(app.config['UPLOAD_FOLDER'],'cancer.jpg')
heartpic = os.path.join(app.config['UPLOAD_FOLDER'],'heart.jpg')
kidneypic = os.path.join(app.config['UPLOAD_FOLDER'],'kidney.jpg')
diabetespic = os.path.join(app.config['UPLOAD_FOLDER'],'diabetes.jpg')

aboutcancer = os.path.join(app.config['UPLOAD_FOLDER'],'cancer.jpeg')
aboutkidney = os.path.join(app.config['UPLOAD_FOLDER'],'kidney.jpeg')
aboutheart = os.path.join(app.config['UPLOAD_FOLDER'],'heart.jpeg')
aboutdiabetes = os.path.join(app.config['UPLOAD_FOLDER'],'diabetes.jpeg')

@app.route('/')
def index():
    return render_template('index.html',cancerpic=cancerpic,heartpic=heartpic,kidneypic=kidneypic,diabetespic=diabetespic)

@app.route('/about')
def about():
    return render_template('about.html',aboutcancer=aboutcancer,aboutheart=aboutheart,aboutkidney=aboutkidney,aboutdiabetes=aboutdiabetes)


@app.route('/cancer',methods=["POST","GET"])
def cancer():
    if request.method == 'POST':
        ct = request.form['ct']
        csz = request.form['csz']
        csp = request.form['csp']
        cad = request.form['cad']
        es = request.form['esz']
        bn = request.form['bn']
        bc = request.form['bc']
        nn = request.form['nn']


        sample_data = [ct,csz,csp,cad,es,bn,bc,nn]
        clean_data = [float(i) for i in sample_data]
        
        ex1 = np.array(clean_data).reshape(1,-1)
        load_model = joblib.load('data/cancer_model.pkl')
        prediction = load_model.predict(ex1)
        pred = load_model.predict([clean_data])
        if pred[0] == 0:
            state = "Negetive"
        else:
            state = "Positive"
        return render_template('cancer.html',user_img = pic1,state=state)
    else:
        return render_template('cancer.html',user_img = pic1)

@app.route('/dibetes',methods=["POST","GET"])
def dibetes():
    if request.method == 'POST':
        preg = request.form['preg']
        glc = request.form['glc']
        bp = request.form['bp']
        st = request.form['st']
        ins = request.form['ins']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        sample_data = [preg,glc,bp,st,ins,bmi,dpf,age]
        clean_data = [float(i) for i in sample_data]
        
        ex1 = np.array(clean_data).reshape(1,-1)
        load_model = joblib.load('data/dibetes_model.pkl')
        pred = load_model.predict(ex1)
        print(pred)
        if pred[0] == 0:
            state = "Negetive"
        else:
            state = "Positive"
        return render_template('dibetes.html',user_img = pic1,state=state)
    else:
        return render_template('dibetes.html',user_img = pic1)


@app.route('/kidney',methods = ["POST","GET"])
def kidney():
    if request.method == 'POST':
        pcv = request.form['pcv']
        sc = request.form['sc']
        sg = request.form['sg']
        su = request.form['su']
        age = request.form['age']
        pot = request.form['pot']
        wc = request.form['wc']
        rc = request.form['rc']
        al = request.form['al']
        bgr = request.form['bgr']
        dm = request.form['dm']

        sample_data = [pcv,sc,sg,su,age,pot,wc,rc,al,bgr,dm]
        clean_data = [float(i) for i in sample_data]
        
        ex1 = np.array(clean_data).reshape(1,-1)
        load_model = joblib.load('data/kidney_model.pkl')
        prediction = load_model.predict(ex1)
        pred = load_model.predict([clean_data])
        if pred[0] == 0:
            state = "Negetive"
        else:
            state = "Positive"
        return render_template('kidney.html',user_img = pic1,state=state)
    else:
        return render_template('kidney.html',user_img = pic1)


@app.route('/heart',methods=["POST","GET"])
def heart():    
    if request.method == 'POST':
        print("hiii")
        age = request.form['age']
        sex = request.form['sex']
        cigspDay = request.form['cigspDay']
        bpmeds = request.form['bpmeds']
        totChol = request.form['totChol']
        sysbp = request.form['sysbp']
        gls = request.form['gls']

        sample_data = [age,sex,cigspDay,bpmeds,totChol,sysbp,gls]
        clean_data = [eval(i) for i in sample_data]
        
        print(type(clean_data))
        ex1 = np.array(clean_data).reshape(1,-1)
        load_model = joblib.load('data/heart_model.pkl')
        prediction = load_model.predict(ex1)
        pred = load_model.predict([clean_data])
        if pred[0] == 0:
            state = "Negetive"
        else:
            state = "Positive"
        return render_template('heart.html',user_img = pic1,state=state)
    else:
        return render_template('heart.html',user_img = pic1)

if __name__ == "__main__":
    app.run(port=7010)