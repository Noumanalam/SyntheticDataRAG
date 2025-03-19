# Synthetic Data Generation with Retrieval-Augmented Generation (RAG)

This project creates synthetic e-commerce data to solve issues with real data scarcity or privacy concerns.

## Overview
- **Purpose**: Generate synthetic transaction records (CustomerID, Quantity, UnitPrice, Country).
- **Dataset**: E-commerce data from Kaggle.
- **Tools**: Pinecone, Transformers, Pandas, Scikit-learn, Matplotlib.

## Steps
1. **Setup**: Installed necessary tools (Pinecone, Transformers, etc.).
2. **Data**: Cleaned the dataset and saved it as cleaned_data.csv.
3. **Retrieve**: Built a vector index in Pinecone for retrieval.
4. **Generate**: Used DistilGPT2 to create synthetic records.
5. **Evaluate**: Compared real and synthetic data.

## Results
- **KS Test**: Quantity (0.82, p=0.073) shows good similarity, UnitPrice (0.97, p=0.0039) has differences.
- **Accuracy**: Real data achieved 1.0, synthetic got 0.0 (needs tuning).
- **Visuals**: See graphs in `results/` folder.

## Future Work
- Improve prompts to better align UnitPrice.
- Generate more synthetic records.

## How to Run
1. Clone repo: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python src/step1_setup.py` (for each step).

## Skills Demonstrated
- Data cleaning, AI generation, ML evaluation, visualization.
