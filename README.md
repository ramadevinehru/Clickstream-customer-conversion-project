# Customer Conversion Analysis for Online Shopping Using Clickstream Data

### Overview:

This project is a **Customer Conversion Analysis Application** built using **Python, Streamlit, and Scikit-learn**. It analyzes online shopping clickstream data to predict customer conversion, estimate potential revenue, and segment users into distinct customer groups. The app helps businesses gain actionable insights to improve engagement and boost sales.

### Features:

* **File Upload**: Users can upload a CSV file.
* **Real-time Predictions**:

  * **Conversion Prediction**: Classifies whether a customer will convert or not.
  * **Revenue Estimation**: Predicts expected revenue from a customer.
* **Customer Segmentation**: Uses clustering techniques to group customers based on behavior.
* **Data Visualization**: Offers interactive plots including:

  * Price distribution
  * Clustered customer segments
* **Preprocessing Pipeline**: Handles missing values, feature scaling, one-hot encoding, and consistent feature alignment.

### Technologies Used:

* Python
* Streamlit
* pandas
* scikit-learn
* matplotlib
* seaborn
* pickle (for model serialization)

### Prerequisites:

Ensure you have the following installed:

* Python (>=3.13)
* Required Python libraries (`pip install -r requirements.txt`)

### Usage:

1. Launch the app using the command:

   ```bash
   streamlit run streamlit_app.py
   ```
2. Upload a CSV file.
3. Click the appropriate button to:

   * Predict conversion
   * Estimate revenue
   * Display customer segments
   * View data visualizations

### Input Data Format:

Expected columns include:

* `session_length`
* `price` (or `price_2` based on model training)
* Any relevant categorical or numerical features used during model training.

### Models:

* **Classification**: Predicts conversion using models like Logistic Regression, Random Forest, or Neural Networks.
* **Regression**: Estimates revenue using Linear Regression or Gradient Boosting models.
* **Clustering**: Segments customers using algorithms like K-Means or DBSCAN.

Models are pre-trained and stored as:

* `models/classification.pkl`
* `models/regression.pkl`
* `models/clustering.pkl`

### Contribution:

Feel free to contribute by forking the repository and submitting pull requests.

### License:

This project is licensed under the **MIT License**.

### Author:

**Ramadevi N**

