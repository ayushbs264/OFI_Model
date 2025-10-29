 **OFI Model – Predictive Delivery Optimization**

This project is a machine learning-powered delivery optimization dashboard built using Streamlit.
It analyzes delivery data and predicts potential delays to help improve operational efficiency and customer satisfaction.


**Project Review**

The model provides:

- Insights into delivery performance and quality issues

- Predictive delay analysis for new delivery orders

- Visualization of logistics trends based on multiple datasets

It combines multiple logistics-related datasets such as delivery performance, vehicle fleet, customer feedback, and route distances to train and visualize predictive insights.

 **Key Features**

-Interactive dashboard built with Streamlit

-Predictive model trained using scikit-learn

-Visual analytics for delivery delays, quality issues, and costs

-Multiple data sources integrated:

  1)Delivery performance

  2)Customer feedback

  3)Vehicle fleet

  4)Route distances

  5)Warehouse inventory

  6)Cost breakdown

 **Technologies Used**

Programming Language ---	Python 3
Web Framework ---       	Streamlit
Machine Learning ---    	Scikit-learn
Data Handling ---	        Pandas, NumPy
Visualization ---	        Plotly, Matplotlib
Model Storage ---	        Joblib


** Repository Structure**
OFI_Model/
│
├── app.py                    # Streamlit web app
├── model.joblib              # Trained ML model
├── requirements.txt          # Required dependencies
│
├── cost_breakdown.csv        # Delivery cost-related data
├── customer_feedback.csv     # Customer rating and feedback data
├── delivery_performance.csv  # Core dataset used for prediction
├── orders.csv                # Order information
├── routes_distance.csv       # Route and distance data
├── vehicle_fleet.csv         # Fleet and vehicle details
└── warehouse_inventory.csv   # Warehouse and stock information

** Installation and Setup**
1.) Clone the Repository
git clone https://github.com/ayushbs264/OFI_Model.git
cd OFI_Model

2️.) Install Dependencies
pip install -r requirements.txt

3️.) Run the Streamlit App
streamlit run app.py

4️.) Open in Browser

Once started, open:
👉 http://localhost:8501

 **Sample Dashboard Views**
-Delay by Quality Issue

Displays delivery delays categorized by quality issues such as minor damage, wrong item, or incomplete delivery.

-Predict Delay for a New Order

Users can input:

1)Carrier

2)Promised Delivery Days

3)Estimated Delivery Cost (INR)

4)Expected Customer Rating
    and get a predicted delay risk instantly.

** Model Information**

-Model Type: Logistic Regression

-Training Data: delivery_performance.csv

-Target Variable: Delivery Delay (binary)

Features Used:

1)Promised Delivery Days

2)Delivery Cost

3)Customer Rating

4)Carrier

  The trained model is stored as model.joblib and loaded dynamically by the Streamlit app.


** Author**

Ayush Bhushan Singh
GitHub: ayushbs264
