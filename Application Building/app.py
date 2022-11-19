import numpy as np
from flask import Flask , render_template,request
import tensorflow as tf
from tensorflow.keras.models import load_model
tf.get_logger().setLevel('ERROR')
app=Flask(__name__,template_folder='template')
model=load_model('crude_oil.h5')
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/')
def home1():
    return render_template("web.html")
@app.route('/predict')
def home2():
    return render_template("web1.html")
@app.route('/login' ,methods = ['POST'])
def login():
    x_input=str(request.form['year'])
    x_input=x_input.split(',')
    n=len(x_input) + 1
    for i in range(0, len(x_input)): 
        x_input[i] = float(x_input[i]) 
    x_input=np.array(x_input).reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=10
    i=0
    while(i<1):
        if(len(temp_input)>10):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    return render_template("web1.html",showcase='The predicted value is: '+" "+str(lst_output[0][0]))
if __name__ == '__main__':
    app.run(debug = True,port=5000)