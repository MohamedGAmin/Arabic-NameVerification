

import numpy as np
from flask import Flask, request, render_template
from tensorflow import keras
from data_preprocessing import prepare_X
import time
#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model.
model_fake = keras.models.load_model('models/best_fake.h5')
model_gender = keras.models.load_model('models/best_gender.h5')
fake_threshold=0.55
gender_threshold=0.585

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    start_time=time.time()
    try:
        input = [x for x in request.form.values()][0]
        names = input.split() 
        if len(names)==3:
            names=prepare_X(names)
            fake_output=model_fake.predict(names)
            output=[1 if i >= fake_threshold else 0 for i in fake_output[:, -1]]
            if sum(output)<=1:
                gender_output=model_gender.predict(names)
                output=[1 if i >= gender_threshold else 0 for i in gender_output[:, -1]]
                if output[1]!=1 and output[2]!=1:
                    return render_template('index.html', prediction_text=f'{input} is a Real Name with high confidence',time_spent=f'Execution time {round(time.time()-start_time,2)} seconds')
                else:
                    return render_template('index.html', prediction_text=f'{input} is a Real Name with low confidence',time_spent=f'Execution time {round(time.time()-start_time,2)} seconds')
            elif sum(output)==2:
                return render_template('index.html', prediction_text=f'{input} is a Real Name with low confidence',time_spent=f'Execution time {round(time.time()-start_time,2)} seconds')

            return render_template('index.html', prediction_text=f'{input} is a Fake Name',time_spent=f'Execution time {round(time.time()-start_time,2)} seconds')
        else:
            return render_template('index.html', prediction_text=f"Please enter three names",time_spent=f'Execution time {round(time.time()-start_time,2)} seconds')
    except:
        return render_template('index.html', prediction_text="ERROR, Try Again",time_spent=f'Execution time {round(time.time()-start_time,2)} seconds')
    
        


if __name__ == "__main__":
    app.run()