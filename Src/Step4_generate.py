from transformers import pipeline

generator = pipeline('text-generation', model='distilgpt2')

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import pandas as pd

pc = Pinecone(api_key='YOUR_API_KEY_HERE')
index = pc.Index('ecommerce-data')
model = SentenceTransformer('all-MiniLM-L6-v2')

data = pd.read_csv('cleaned_data.csv')
data['text'] = data.apply(lambda row: f"CustomerID: {row['CustomerID']}, Quantity: {row['Quantity']}, UnitPrice: {row['UnitPrice']}, Country: {row['Country']}", axis=1)
sample_data = data['text'].head(100).tolist()  

query = "CustomerID: 17850, Quantity: 6, UnitPrice: 2.55, Country: United Kingdom"
query_embedding = model.encode([query])[0].tolist()
result = index.query(vector=query_embedding, top_k=3)

retrieved_examples = [sample_data[int(match['id'])] for match in result['matches']]
print("Retrieved examples:", retrieved_examples)

prompt = "Generate 5 synthetic e-commerce transaction records similar to these examples:\n"
for i, example in enumerate(retrieved_examples, 1):
    prompt += f"{i}. {example}\n"
prompt += "Each record should include CustomerID (a 5-digit number), Quantity (1-20), UnitPrice (0.5-50.0), Country (a valid country name), and follow a similar format."
print("Prompt:", prompt)

synthetic_output = generator(prompt, max_length=300, num_return_sequences=1, truncation=True)[0]['generated_text']
print("Synthetic output:", synthetic_output)

import re

pattern = r"CustomerID: (\d+), Quantity: (\d+), UnitPrice: ([\d.]+), Country: (\w+)"
synthetic_records = re.findall(pattern, synthetic_output)
synthetic_data = [
    {"CustomerID": int(cid), "Quantity": int(qty), "UnitPrice": float(up), "Country": country}
    for cid, qty, up, country in synthetic_records
]
print("Synthetic records:", synthetic_data)








