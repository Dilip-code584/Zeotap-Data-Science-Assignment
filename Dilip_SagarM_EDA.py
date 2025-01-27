import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

customers_file = "/home/dilip/Downloads/Zeotap/Customers.csv"
products_file = "/home/dilip/Downloads/Zeotap/Products.csv"
transactions_file = "/home/dilip/Downloads/Zeotap/Transactions.csv"

customers = pd.read_csv(customers_file)
products = pd.read_csv(products_file)
transactions = pd.read_csv(transactions_file)

print("Checking missing values:")
print("Customers:")
print(customers.isnull().sum())
print("Products:")
print(products.isnull().sum())
print("Transactions:")
print(transactions.isnull().sum())

print("\nChecking duplicates:")
print(f"Customers: {customers.duplicated().sum()} duplicates")
print(f"Products: {products.duplicated().sum()} duplicates")
print(f"Transactions: {transactions.duplicated().sum()} duplicates")

region_counts = customers['Region'].value_counts()
print("\nUnique customers by region:")
print(region_counts)

sns.barplot(x=region_counts.index, y=region_counts.values)
plt.title("Number of Customers by Region")
plt.xlabel("Region")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig('/home/dilip/Downloads/region_counts.png', bbox_inches='tight')
plt.close()

category_counts = products['Category'].value_counts()
print("\nTop 5 products by category:")
print(category_counts)

sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title("Top Categories by Product Count")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig('/home/dilip/Downloads/category_counts.png', bbox_inches='tight')
plt.close()

transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
transactions['YearMonth'] = transactions['TransactionDate'].dt.to_period('M').astype(str)

monthly_trends = transactions.groupby('YearMonth')['TotalValue'].sum()
print("\nMonthly transaction trends:")
print(monthly_trends)

monthly_trends.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Monthly Transaction Trends')
plt.xlabel('Year-Month')
plt.ylabel('Total Transaction Value')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig('/home/dilip/Downloads/monthly_trends.png', bbox_inches='tight')
plt.close()

print("Plots have been saved successfully.")
