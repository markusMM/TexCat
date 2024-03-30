from flask import Flask
from flask_restful import Resource, Api
from flask_cors import CORS
from src import predictor


app = Flask(__name__)
CORS(app)
api = Api(app)


# prediction api call
class prediction(Resource):
    def get(self, text):
        prediction = predictor.categorize(text)
        return prediction


api.add_resource(prediction, "/prediction/<string:category>")

if __name__ == "__main__":
    app.run(debug=True)
