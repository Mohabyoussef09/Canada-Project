import os
import json

from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
import analyzeUtilis as au

import mysql
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import MySQLdb.cursors
from flask_mysqldb import MySQL
import os
import re

import constants
from binning_calculations import calculate_data_variables,calculate_bin_variables,apply_binning
from analyze_calculations import calculate_distribution_variables,calculate_volatility_variables,calculate_correlation_variables
from dataset import Dataset
from main_window import MainWindow
from analyze_display import AnalyzeDisplay
import pandas as pd
from binning_settings import (
    BinAsMode,
    BinningNumericMode,
    BinningSettings,
)
import config
import numpy as np


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

#upload file
@app.route('/api/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.files['upload-file']
        ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
        file.save(os.path.join(ROOT_PATH, file.filename))
        data = pd.read_csv(file.filename)
        return data.to_json(orient='records')


#send list of dictioanries to me
''''
[{"default":1,"x1":1,"x2":null,"x3":10.0,"x4":-0.227743901,"x5":-0.096194976,"x6":0.647261887,"x7":0,"x8":1,"x9":10,"x10":0.847598004,"x11":"Y","x12":null,"x13":"29-Feb-20"},{"default":1,"x1":0,"x2":1.867784437,"x3":0.066157442,"x4":1.666478059,"x5":-0.096194976,"x6":0.259899026,"x7":0,"x8":1,"x9":10,"x10":0.833230712,"x11":"N","x12":"High","x13":"5-Mar-19"},{"default":1,"x1":1,"x2":1.867784437,"x3":0.066157442,"x4":-0.227743901,"x5":-0.096194976,"x6":0.230768773,"x7":0,"x8":4,"x9":10,"x10":0.975528418,"x11":"N","x12":"High","x13":"31-Jan-18"},{"default":1,"x1":1,"x2":1.867784437,"x3":4.313933758,"x4":-0.490906623,"x5":-0.477077425,"x6":0.406273195,"x7":0,"x8":1,"x9":10,"x10":0.580412251,"x11":null,"x12":"Medium","x13":"29-Feb-20"},{"default":1,"x1":1,"x2":-1.16726471,"x3":0.066157442,"x4":-0.227743901,"x5":-0.096194976,"x6":0.259899026,"x7":0,"x8":1,"x9":10,"x10":0.533771841,"x11":"Y","x12":"High","x13":"5-Mar-19"},{"default":1,"x1":0,"x2":-1.16726471,"x3":0.066157442,"x4":-0.227743901,"x5":-0.096194976,"x6":0.647261887,"x7":1,"x8":1,"x9":10,"x10":null,"x11":"N","x12":"High","x13":"31-Jan-18"}]
'''
@app.route('/calculateBinVariables', methods=['GET', 'POST'])
def calculateBinVariables():
    b=BinningSettings()
    data= request.get_json()
    data= pd.DataFrame(data)
    binVariables=calculate_data_variables(data,b)
    return binVariables.to_json(orient='records')

@app.route('/calculateBinAnalysis', methods=['GET', 'POST'])
def calculateBinAnalysis():
    dataRequest = request.get_json()
    target = dataRequest["target"]
    data=dataRequest["data"]
    b = BinningSettings()
    data = pd.DataFrame(data)
    binVariables=calculate_data_variables(data,b)
    binAnalysis= calculate_bin_variables(data, b, target, binVariables)
    return binAnalysis.to_json(orient='records')


#json with binAnalysis and data
@app.route('/binVariablesOkButton', methods=['GET', 'POST'])
def binVariablesOkButton():
    dataRequest = request.get_json()
    bin_vars=pd.DataFrame(dataRequest["binAnalysis"])
    data = pd.DataFrame(dataRequest["data"])
    parent_list = list(bin_vars["variable_parent"].unique())
    binned_list = list(bin_vars["variable"].unique())
    df_binned = apply_binning(data[parent_list], bin_vars)[binned_list]

    # merge data dataframe with df_binned dataframe when user press ok
    datasetObject = Dataset()
    finalBinningData = datasetObject.add_binned_feature_vars(data, df_binned)

    return finalBinningData.to_json(orient='records')


@app.route('/variableAnalyze', methods=['GET', 'POST'])
def variableAnalyze():
    dataRequest = request.get_json()
    data = pd.DataFrame(dataRequest["data"])
    targetLabel = dataRequest["target"]
    probabilityLabel=constants.PROBABILITY_LABEL

    analyze=AnalyzeDisplay()
    invalid_vars= list(range(config.DEFAULT_COL_COUNT))

    invalid_vars=au.update_invalid_vars_list(data,probabilityLabel)
    non_event_labels=au.non_event_labels(data,probabilityLabel,targetLabel)
    modelQualifiedVarLabels=au.model_qualified_var_labels(data,non_event_labels,invalid_vars)

    distributionVariables=calculate_distribution_variables(data[modelQualifiedVarLabels])
    volatilityVariables=calculate_volatility_variables(data[modelQualifiedVarLabels])
    correlatioVariables=calculate_correlation_variables(data[modelQualifiedVarLabels])

    return {"distributionVariable":distributionVariables.to_json(orient='records'),
            "volatileVariable":volatilityVariables.to_json(orient='records'),
            "correlationVariable":correlatioVariables.to_json(orient='records')}


@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ""
    id=0
    username=""
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = % s AND password = % s', (username, password, ))
        account = cursor.fetchone()
        if account:
            msg = 'Logged in successfully !'
            id = account["id"]
            username=account["username"]
        else:
            msg = 'Incorrect username / password !'
            id = -1
            username="NA"
    return json.dumps({"id": id, "username": username,"msg":msg})


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template('register.html')


@app.route('/register', methods=['GET', 'POST'])
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
            cursor.execute(
                'INSERT INTO users (username,password) VALUES (% s, % s)', (username, password,))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return msg


if __name__ == '__main__':
    app.run(debug=True)
