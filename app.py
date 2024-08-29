from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('mnist_cnn_model.h5')

# Initialize the camera
camera = cv2.VideoCapture(0)

def preprocess_frame(frame):
    """Preprocess the camera frame for prediction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    processed_frame = resized.reshape(1, 28, 28, 1) / 255.0
    return processed_frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)
            predicted_digit = np.argmax(prediction)
            cv2.putText(frame, f'Digit: {predicted_digit}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def preprocess_image(image_data):
    image_data = image_data.split(",")[1]
    image = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image)).convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.form['image_data']
    processed_image = preprocess_image(image_data)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    return jsonify({'digit': int(predicted_digit)})

@app.route('/capture', methods=['POST'])
def capture():
    # Capture the current frame from the camera
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Failed to capture image'}), 500

    # Convert frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

    # Return the image as base64
    return jsonify({'image_data': f'data:image/jpeg;base64,{frame_base64}'})

if __name__ == "__main__":
    app.run(debug=True)
