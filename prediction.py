import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://t3.ftcdn.net/jpg/03/04/07/34/360_F_304073455_ZMlki5v2LTzgXjm8FZlgdhsXApcA9RUZ.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🏠 House Price Prediction")

# Load dataset
df = pd.read_csv("C:/wanted stuff/COE cource/LinearRegression_houseprice_data/house-prices.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# Graph 1
st.subheader("Square Feet vs Price")

fig, ax = plt.subplots()
sns.scatterplot(x=df['SqFt'], y=df['Price'], ax=ax)
st.pyplot(fig)

# Graph 3
st.subheader("Bathrooms vs Price")

fig, ax = plt.subplots()
sns.boxplot(x=df['Bathrooms'], y=df['Price'], ax=ax)
st.pyplot(fig)

# Train model
X = df[['SqFt','Bedrooms','Bathrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)

# Show model score
y_pred = model.predict(X_test)
score = r2_score(y_test,y_pred)
st.subheader("Model Accuracy")
st.write(score)

# Inputs after graphs
st.subheader("Enter House Details")

sqft = st.number_input("Square Feet",500,5000,1500)
bedrooms = st.number_input("Bedrooms",1,10,3)
bathrooms = st.number_input("Bathrooms",1,5,2)

# Prediction button
if st.button("Predict Price"):
    
    prediction = model.predict([[sqft,bedrooms,bathrooms]])
    
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")