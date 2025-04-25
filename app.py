from flask import Flask, request, jsonify
from flask_cors import CORS
import dlib
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import google.generativeai as poki
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize dlib's face detector and shape predictor
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print(f"Error loading dlib models: {e}")
    exit(1)

def get_face_shape(landmarks):
    try:
        jaw = landmarks[0:17]
        jaw_width = np.linalg.norm(jaw[0] - jaw[-1])
        jaw_height = np.linalg.norm(jaw[8] - (jaw[0] + jaw[-1]) / 2)

        if jaw_width > jaw_height * 1.5:
            return "round"
        elif jaw_height > jaw_width * 1.5:
            return "long"
        elif abs(jaw[4][0] - jaw[12][0]) > abs(jaw[0][1] - jaw[-1][1]):
            return "square"
        else:
            return "oval"
    except Exception as e:
        return f"Error in face shape detection: {str(e)}"

@app.route('/detect_face_shape', methods=['POST'])
def detect_face_shape():
    try:
        data = request.json
        img_data = data.get("image")
        if not img_data:
            return jsonify({"status": "error", "error": "No image received"}), 400

        # Decode base64 image
        image_bytes = base64.b64decode(img_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            return jsonify({"status": "error", "error": "No face detected"}), 400

        # Get landmarks
        shape = predictor(gray, faces[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        face_shape = get_face_shape(landmarks)

        return jsonify({
            "status": "success",
            "face_shape": face_shape,
            "hair_type": "Curly",  # Placeholder (replace with actual logic if needed)
            "recommendation": "Short Crop"  # Placeholder (replace with actual logic if needed)
        })
    except Exception as e:
        return jsonify({"status": "error", "error": f"Processing error: {str(e)}"}), 500

poki.configure(api_key="AIzaSyD9xH034GyER34QNDV9t_r5t9SpXdtrj34")  # Replace with your actual API key
model = poki.GenerativeModel("models/gemini-1.5-pro-latest")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_prompt = data.get('message')

    if not user_prompt:
        return jsonify({'error': 'No message provided'}), 400

    try:
        response = model.generate_content(user_prompt)
        # Handle the response based on its structure
        reply = response.text if hasattr(response, "text") else response.candidates[0].content.parts[0].text
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_html():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
