import numpy as np
from flask import Flask, request,jsonify,  render_template
from annoy import AnnoyIndex
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# Create flask app
app = Flask(__name__)
index = AnnoyIndex(384, 'angular')
index.load("annoy_index.ann")
data = pd.read_csv('data.csv')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
@app.route("/")
def Home(): 
    return render_template("chat.html")

@app.route("/ask", methods = ["POST"])
def get_response():
    message = [request.form['messageText']]
    t = tokenizer(message, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    sentence_embeddings = embeddings[0].cpu().numpy()
    closest_index = index.get_nns_by_vector(sentence_embeddings, 1)[0]  # Ищем ближайший вектор
    answer = data["chandler's phrase"][closest_index].replace('\xa0', ' ')
    return jsonify({'status':'OK','answer':answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5555)
  
