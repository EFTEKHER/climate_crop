from flask import Flask, request, render_template
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load the model and the label encoder using joblib
models, label_encoder, results_df = joblib.load('models.pkl')

# Mapping from encoded item to original item names
item_mapping = {i: item for item, i in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

def predict_yield_all_models(item, year, avg_rainfall, pesticides, avg_temp, models):
    input_data = pd.DataFrame([[item, year, avg_rainfall, pesticides, avg_temp]], 
                              columns=['Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'])
    
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(input_data)
        predictions[model_name] = prediction[0]
    
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    item = int(request.form['Item'])
    year = int(request.form['Year'])
    avg_rainfall = float(request.form['average_rain_fall_mm_per_year'])
    pesticides = float(request.form['pesticides_tonnes'])
    avg_temp = float(request.form['avg_temp'])

    predictions = predict_yield_all_models(item, year, avg_rainfall, pesticides, avg_temp, models)
    
    return render_template('index.html', predictions=predictions, item=item, year=year, avg_rainfall=avg_rainfall, pesticides=pesticides, avg_temp=avg_temp)

@app.route('/results')
def results():
    # Generate and save the performance graph
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='R2 Score', data=results_df, palette='viridis', hue='Model', legend=False)
    plt.title('R2 Score for Different Models')
    plt.savefig('static/images/r2_score.png')
    plt.clf()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='MAE', data=results_df, palette='viridis', hue='Model', legend=False)
    plt.title('Mean Absolute Error for Different Models')
    plt.savefig('static/images/mae.png')
    plt.clf()

    return render_template('results.html', r2_image='static/images/r2_score.png', mae_image='static/images/mae.png', results_df=results_df)

if __name__ == '__main__':
    app.run(debug=True)
