import pandas as pd
from sentence_transformers import SentenceTransformer

data = pd.read_csv('cleaned_data.csv')

data['text'] = data.apply(lambda row: f"CustomerID: {row['CustomerID']}, Quantity: {row['Quantity']}, UnitPrice: {row['UnitPrice']}, Country: {row['Country']}", axis=1)

sample_data = data['text'].head(100).tolist()


model = SentenceTransformer('all-MiniLM-L6-v2')  

embeddings = model.encode(sample_data, show_progress_bar=True)

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key='YOUR_API_KEY_HERE')

index_name = 'ecommerce-data'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1') 
    )

index = pc.Index(index_name)

ids = [str(i) for i in range(len(sample_data))]
vectors = [(id_, emb.tolist()) for id_, emb in zip(ids, embeddings)]

index.upsert(vectors)
print("Vectors uploaded to Pinecone index!")