
from flask import Flask, request, jsonify, send_file
import google.generativeai as poki
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure Gemini model
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
    app.run(host="0.0.0.0", port=5001, debug=True)