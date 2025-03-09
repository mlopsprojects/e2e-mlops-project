import streamlit as st
import requests

# FastAPI backend URL (update the host if necessary)
FASTAPI_URL = "http://fastapi:8000/data"  # Use service name in Kubernetes

st.title("Streamlit + FastAPI App")

st.write("Fetching data from FastAPI...")

# Fetch data from FastAPI
try:
    response = requests.get(FASTAPI_URL)
    if response.status_code == 200:
        data = response.json()
        st.write("Data from FastAPI:", data)
    else:
        st.error("Failed to fetch data from FastAPI")
except Exception as e:
    st.error(f"Error connecting to FastAPI: {e}")
