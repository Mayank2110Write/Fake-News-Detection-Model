from flask import Flask, request, jsonify
import joblib
import scipy.sparse

app = Flask(__name__)

# Load the trained model
model = joblib.load('fake_news_detection_model.pkl')

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json['data']
    
    # Transform the input data into TF-IDF features
    tfidf_features = tfidf_vectorizer.transform([data])
    
    # Make predictions
    prediction = model.predict(tfidf_features)[0]
    
    # Return the prediction as JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
