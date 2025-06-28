from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load your CNN model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

emotion_suggestions = {
    "sad": "Try a 5-minute yoga session.",
    "angry": "Play a relaxing game or breathe deeply.",
    "fearful": "Watch a motivational video.",
    "disgusted": "Take a short break or walk.",
    "happy": "Keep learning, you're doing great!",
    "surprised": "Explore something new today!",
    "neutral": "Stay focused and take short breaks."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/emotion', methods=['POST'])
def detect_emotion():
    data = request.get_json()
    image_data = np.array(data['imageData']).reshape(48, 48, 4)
    
    # Convert to grayscale
    gray_image = np.mean(image_data[:, :, :3], axis=2)
    gray_image = gray_image.reshape((1, 48, 48, 1))
    
    # Normalize pixel values
    gray_image = gray_image.astype('float32') / 255.0
    
    # Make prediction
    prediction = model.predict(gray_image)
    emotion = np.argmax(prediction[0])
    
    # Map numeric emotion to string
    emotions = ["sad", "angry", "fearful", "disgusted", "happy", "surprised", "neutral"]
    detected_emotion = emotions[emotion]
    
    suggestion = emotion_suggestions.get(detected_emotion, 
                                      "Stay positive and keep learning.")
    
    return jsonify({
        'emotion': detected_emotion,
        'suggestion': suggestion
    })

if __name__ == "__main__":
    app.run(debug=True)