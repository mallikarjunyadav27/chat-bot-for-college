from flask import Flask,render_template,request
import random
import json
import torch

from model  import NeuralNet
from nltk1 import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my=open('C:/Users/sai kiran Reddy/Downloads/malli intents.json','r')
jsondata=my.read()
intents=json.loads(jsondata)
print(intents)


FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
app = Flask(__name__,template_folder='templates')
@app.route('/',methods=['GET','POST'])
   # return render_template("index.html")
#@app.route("/get")
#def get_bot_response():
 #bot_name = "Sam"
 #print("Let's chat! (type 'quit' to exit)")
  #while True:
    # sentence = "do you use credit cards?"
   # sentence = input("You: ")
    #if sentence == "Thankyou":
       # break
def samplefunction():
   if request.method == 'GET':
       return render_template('index.html')
   if request.method == 'POST':
       sentence  = request.form['human']

       sentence = tokenize(sentence)
       X = bag_of_words(sentence, all_words)
       X = X.reshape(1, X.shape[0])
       X = torch.from_numpy(X).to(device)

       output = model(X)
       _, predicted = torch.max(output, dim=1)

       tag = tags[predicted.item()]

       probs = torch.softmax(output, dim=1)
       prob = probs[0][predicted.item()]
       if prob.item() > 0.75:
          for intent in intents['intents']:
            if tag == intent["tag"]:
                bot =random.choice(intent['responses'])
                return render_template('index.html',bot=bot)
       else:
                bot='sorry.no idea!'
                return render_template('index.html',bot=bot)

if __name__ == "__main__":
   app.run(debug=True)
            