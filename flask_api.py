from flask import Flask, request, jsonify
import pickle 

app = Flask(__name__)

with open("D:\\mlops\\picklefile\\iris_model.pkl","rb") as fileobj:
   iris_model = pickle.load(fileobj)    #Load the pre-trained model

@app.route(rule="/", methods=["GET"])
def home():
  return "Welcome to Shivani's Page"

@app.route("/get_sqaure", methods=["POST"]) #this is controller
# this is view function


def get_square(): 
    
    data = request.get_json()
    number = data.get("number")
    return jsonify({"square": number ** 2})

@app.route("/iris_prediction", methods=["POST"]) #this is controller

def iris_prediction():
   
   data = request.get_json()
   sepal_length= data.get("sl")
   sepal_width= data.get("sw")
   petal_length= data.get("pl")
   petal_width= data.get("pw")

   
   flower_type = iris_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
   return jsonify({"predcited_flower_type": flower_type[0]})

#  running main file
#dunder name == dunder main
if __name__ == "__main__":
    app.run()
