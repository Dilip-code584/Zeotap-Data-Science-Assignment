import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

merged_data = pd.merge(customers, transactions, on='CustomerID')

print(merged_data.head())

numeric_columns = merged_data.select_dtypes(include=['number']).columns
merged_data[numeric_columns] = merged_data[numeric_columns].fillna(merged_data[numeric_columns].mean())


agg_data = merged_data.groupby(['CustomerID', 'Region']).agg(
    total_transactions=('TotalValue', 'sum'),
    avg_price=('Price', 'mean'),
    transaction_count=('TransactionID', 'count')
).reset_index()

agg_data = pd.get_dummies(agg_data, columns=['Region'], drop_first=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(agg_data.drop('CustomerID', axis=1))

kmeans = KMeans(n_clusters=5, random_state=42)
agg_data['Cluster'] = kmeans.fit_predict(scaled_data)

db_index = davies_bouldin_score(scaled_data, agg_data['Cluster'])
print(f"Davies-Bouldin Index: {db_index}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=agg_data['total_transactions'], y=agg_data['avg_price'], hue=agg_data['Cluster'], palette='Set2')
plt.title('Customer Segmentation Clusters')
plt.xlabel('Total Transactions')
plt.ylabel('Average Price')

plt.savefig('customer_segmentation_clusters.png', format='png')

agg_data.to_csv('Customer_Segmentation_Results.csv', index=False)

print(agg_data.head())

