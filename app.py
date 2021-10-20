from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)
app.debug = True

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.files['upload-file']
        data = pd.read_csv(file)
        return data.to_json(orient='records')


if __name__ == '__main__':
    app.run(debug=True)
