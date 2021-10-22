import json

from flask_cors import CORS
from flask import Flask, render_template, request, jsonify

import pandas as pd


import mysql
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session,jsonify
import MySQLdb.cursors
from flask_mysqldb import MySQL
import os
import re

app = Flask(__name__)
app.debug = True
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root1234'
app.config['MYSQL_DB'] = 'canadadb'

mysql = MySQL(app)

CORS(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.files['upload-file']
        ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
        file.save(os.path.join(ROOT_PATH, file.filename))
        data = pd.read_csv(file.filename)
        return data.to_json(orient='records')


@app.route('/login', methods =['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = % s AND password = % s', (username, password, ))
        account = cursor.fetchone()
        if account:
            #session['loggedin'] = True
            #session['id'] = account['id']
            #session['username'] = account['username']
            msg = 'Logged in successfully !'

            id=account["id"]
        else:
            msg = 'Incorrect username / password !'
            id = -1
    return json.dumps({"id":id,"message":msg})

@app.route('/signup', methods =['GET', 'POST'])
def signup():
    return render_template('register.html')

@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = % s', (username,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'name must contain only characters and numbers !'
        else:
            cursor.execute('INSERT INTO users (username,password) VALUES (% s, % s)', (username, password,))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return msg



if __name__ == '__main__':
    app.run(debug=True)
