from src.infer import main
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test():
    # Try to get reqId from the form data
    try:
        result=main()
    except:
        return jsonify({
            'status': 'error',
            'message': 'No reqId found in form data.'
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)