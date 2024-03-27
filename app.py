from flask import Flask, request
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
api = Api(app)

#prediction api call
class prediction(Resource):
    def get(self, text):
        text = [text]
        df = pd.DataFrame(text, columns=['text'])
        model = pickle.load(open('texcat.pkl', 'rb'))
        prediction = model.predict(df)
        return prediction

api.add_resource(prediction, '/prediction/<str:text>')

if __name__ == '__main__':
    app.run(debug=True)
