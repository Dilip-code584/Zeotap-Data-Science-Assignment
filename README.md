Data Science Assignment: eCommerce Transactions Dataset
(Please Run Via Virtual Environment And Ive plotted images via stroring them in png form)
Overview
This repository contains the solution for an eCommerce Data Science assignment, where the task was to analyze an eCommerce transactions dataset consisting of three files: Customers.csv, Products.csv, and Transactions.csv. The assignment involved performing exploratory data analysis (EDA), building a Lookalike Model, and performing customer segmentation using clustering techniques.

Setup and Installation
To run this project, follow these steps:

1. Set up the virtual environment
It's recommended to run this project in a virtual environment to manage dependencies effectively. Follow the steps below to create and activate a virtual environment.
Also please note im saving all the images in png form instead of plotting it directly
python3 -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate the virtual environment
Once the virtual environment is activated, you'll see (venv) at the beginning of your command prompt.

2. Install required modules
You will need to install the following Python modules:

pandas
seaborn
matplotlib
scikit-learn
numpy
You can install these using the following command:

pip install pandas seaborn matplotlib scikit-learn numpy
3. Run the script
Run the Python scripts to execute the tasks:

Task 1: EDA and Business Insights
Run the Python script Dilip_SagarM_EDA.py.
Please note im saving all the images in png form instead of plotting it directly
Expected Output -



![Dilip_SagarM_EDA_Output1](https://github.com/user-attachments/assets/c8dc33fd-4246-474e-8e4c-70509b1564b0)

![Dilip_SagarM_EDA_Output2](https://github.com/user-attachments/assets/db7628ea-a437-4b3c-af85-37d76fc3c168)

Region_Counts


![region_counts](https://github.com/user-attachments/assets/c066cd10-6201-4a60-9c3d-119b616bcf3c)

Monthly_Trends



![monthly_trends](https://github.com/user-attachments/assets/42ed9242-85f4-4ce7-995b-9b0021f4c73c)




Category_Counts


![category_counts](https://github.com/user-attachments/assets/f10a3544-ab0f-4cb2-a146-ec3aecaea77f)











Task 2: Lookalike Model
Run the Python script Dilip_SagarM_Lookalike.py.
Please note im saving all the images in png form instead of plotting it directly
Expected Output -



![Dilip_SagarM_Lookalike_Output](https://github.com/user-attachments/assets/c0edc531-4afd-49b3-ac9a-48d574b63ca7)

And Dilip_SagarM_Lookalike.csv is generate with CustomerID and Similarities


Task 3: Customer Segmentation/Clustering
Run the Python script Dilip_SagarM_Clustering.py.
Expected Output -
Please note im saving all the images in png form instead of plotting it directly

![Dilip_SagarM_Clustering_Output](https://github.com/user-attachments/assets/9b874cd9-25f6-43a3-b863-f6b31cf90437)

![customer_segmentation_clusters](https://github.com/user-attachments/assets/6a8907cf-4143-4762-b764-bfd426395c5b)

And csv file is generated


Task Breakdown
Task 1: Exploratory Data Analysis (EDA) and Business Insights
Perform exploratory data analysis (EDA) on the provided dataset.
Derive at least 5 business insights based on the EDA.
Deliverables: A Python script and a PDF report with business insights.
Task 2: Lookalike Model
Build a Lookalike Model that recommends 3 similar customers based on profile and transaction history.
Deliverables: A CSV file containing the top 3 lookalikes with their similarity scores for the first 20 customers.
Task 3: Customer Segmentation / Clustering
Perform customer segmentation using clustering techniques on both profile and transaction information.
Deliverables: A PDF report and a Python script containing the clustering analysis.
License
This project is open-source and available under the MIT License.
