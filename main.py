from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

file = open('model.pkl','rb')
LR =pickle.load(file)
file.close()
@app.route('/', methods = ["GET","POST"])
def predict():
    if request.method == "POST":
        values = request.form
        fiver = int(values['fiver'])
        breath = int(values['breath'])
        coughing = int(values['cough'])
        cold = int(values['cold'])
        discomfort = int(values['disc'])
        age = int(values['age'])
        sex = int(values['sex'])
        pre = int(values['pre'])
        froeign = int(values['froeign'])
        inputfeatures=[fiver,breath,coughing,cold,discomfort,age,sex]
        prob = LR.predict_proba([inputfeatures])[0][1]
        prob = round(prob*100)
        if (pre == 1):
            prob = prob + 20
        if(froeign == 1 or prob > 45):
            return render_template('danger.html',ip=prob)
        else :
            return render_template('safe.html',ip=prob)
    return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True)

    