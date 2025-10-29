 **OFI Model – Predictive Delivery Optimization**

  This project is a machine learning-powered delivery optimization dashboard built using Streamlit.
  It analyzes delivery data and predicts potential delays to help improve operational efficiency and customer satisfaction.


**(i) Project Review**

  The model provides:

  - Insights into delivery performance and quality issues

- Predictive delay analysis for new delivery orders

- Visualization of logistics trends based on multiple datasets

It combines multiple logistics-related datasets such as delivery performance, vehicle fleet, customer feedback, and route distances to train and visualize predictive insights.

 **(ii) Key Features**

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



**(iii) Installation and Setup**

1.) Clone the Repository
git clone https://github.com/ayushbs264/OFI_Model.git
cd OFI_Model

2️.) Install Dependencies
pip install -r requirements.txt

3️.) Run the Streamlit App
streamlit run app.py

4️.) Open in Browser

Once started, open:
 http://localhost:8501

 **(iv) Sample Dashboard Views**
-Delay by Quality Issue

Displays delivery delays categorized by quality issues such as minor damage, wrong item, or incomplete delivery.

-Predict Delay for a New Order

Users can input:

1)Carrier

2)Promised Delivery Days

3)Estimated Delivery Cost (INR)

4)Expected Customer Rating
    and get a predicted delay risk instantly.

**(v) Model Information**

-Model Type: Logistic Regression

-Training Data: delivery_performance.csv

-Target Variable: Delivery Delay (binary)

Features Used:

1)Promised Delivery Days

2)Delivery Cost

3)Customer Rating

4)Carrier

  The trained model is stored as model.joblib and loaded dynamically by the Streamlit app.


**(vi) Author**

Ayush Bhushan Singh


