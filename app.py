from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np

# Load the trained model
filename = 'heart-disease-prediction-stacked-rf+knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/blog1')
def understanding_heart_diseases():
    return render_template('blog1.html')

@app.route('/blog2')
def healthy_diet_for_heart():
    return render_template('blog2.html')

@app.route('/blog3')
def exercise_heart_health():
    return render_template('blog3.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            # Get all form data with proper type conversion and default values
            age = int(request.form.get('age', 0))
            sex = request.form.get('sex', '0')  # Assuming '0' is default
            cp = request.form.get('cp', '0')    # Assuming '0' is default
            trestbps = int(request.form.get('trestbps', 0))
            chol = int(request.form.get('chol', 0))
            fbs = request.form.get('fbs', '0')  # Assuming '0' is default
            restecg = int(request.form.get('restecg', 0))
            thalach = int(request.form.get('thalach', 0))
            exang = request.form.get('exang', '0')  # Assuming '0' is default
            oldpeak = float(request.form.get('oldpeak', 0.0))
            slope = request.form.get('slope', '0')  # Assuming '0' is default
            ca = int(request.form.get('ca', 0))
            thal = request.form.get('thal', '0')  # Assuming '0' is default

            # Validate that no critical fields are missing
            if any(value is None for value in [age, sex, cp, trestbps, chol, fbs, 
                                              restecg, thalach, exang, oldpeak, slope, ca, thal]):
                flash('Please fill in all required fields', 'error')
                return redirect(url_for('prediction'))

            # Prepare input data - ensure all values are converted to numbers
            data = np.array([
                [
                    age, 
                    int(sex), 
                    int(cp), 
                    trestbps, 
                    chol, 
                    int(fbs), 
                    restecg, 
                    thalach, 
                    int(exang), 
                    oldpeak, 
                    int(slope), 
                    ca, 
                    int(thal)
                ]
            ])
            
            my_prediction = model.predict(data)
            my_probability = model.predict_proba(data)
            
            # Convert probability to a regular Python float
            probability_value = float(my_probability[0][1])  # Get the probability of class 1
            
            return render_template('result.html', 
                                 prediction=int(my_prediction[0]), 
                                 probability=probability_value)
            
        except ValueError as e:
            flash(f'Invalid input: {str(e)}', 'error')
            return redirect(url_for('prediction'))
            
    else:
        return render_template("prediction.html")

if __name__ == '__main__':
    app.run(debug=True)