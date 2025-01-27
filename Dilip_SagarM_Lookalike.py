import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

customers_df = pd.read_csv('Customers.csv')
transactions_df = pd.read_csv('Transactions.csv')

merged_df = pd.merge(transactions_df, customers_df, on='CustomerID')

customer_profile = merged_df.groupby('CustomerID').agg(
    total_transactions=('TotalValue', 'sum'),
    total_quantity=('Quantity', 'sum'),
    total_spent=('TotalValue', 'sum'),
    num_transactions=('TransactionID', 'count')
).reset_index()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_profile.drop(columns=['CustomerID']))

similarity_matrix = cosine_similarity(scaled_features)

similarity_df = pd.DataFrame(similarity_matrix, index=customer_profile['CustomerID'], columns=customer_profile['CustomerID'])

lookalike_dict = {}

for customer_id in customer_profile['CustomerID']:

    similar_customers = similarity_df[customer_id].sort_values(ascending=False).drop(customer_id).head(3)

    lookalike_dict[customer_id] = [(similar_customers.index[i], round(similar_customers.iloc[i], 2)) 
                                   for i in range(len(similar_customers))]

lookalike_list = []

for cust_id, similar_customers in lookalike_dict.items():
    lookalike_list.append({
        'CustomerID (C0001 - C0020)': cust_id,
        'Top 3 Lookalikes and Similarity Scores': f"[{', '.join([f'{i[0]}, {i[1]}' for i in similar_customers])}]"
    })

lookalike_df = pd.DataFrame(lookalike_list)

lookalike_df.to_csv('Dilip_SagarM_Lookalike.csv', index=False, header=True)

print("Lookalike model has been successfully created and saved to 'Dilip_SagarM_Lookalike.csv'")

