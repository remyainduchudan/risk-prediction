from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

@app.route('/losspredict', methods=['POST'])
def losspredict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            print query
            query = query.reindex(columns=autolossmodel_columns, fill_value=0)
            print query
            prediction = list(lr.predict(query))
            

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("autolossmodel.pkl") # Load "model.pkl"
    print ('Model loaded')
    autolossmodel_columns = joblib.load("autolossmodel_columns.pkl") # Load "autolossmodel_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)
