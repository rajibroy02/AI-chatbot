import streamlit as st
from pymongo import MongoClient

# Access the MongoDB URI from secrets.toml
MONGO_URI = st.secrets["MONGO_URI"]
# MONGO_URI = "mongodb+srv://.../ai_chatbot?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["ai_chatbot"]
    collection = db["user_queries"]
    client.server_info()  # Test connection
    st.success("Connected to MongoDB!")
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {str(e)}")
    st.stop()