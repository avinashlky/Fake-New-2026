


import os
import re
import joblib
import streamlit as st

MODEL_PATH = "fake_news_model.pkl"

# --------------------------
# Text cleaning
# --------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

# --------------------------
# Hardcoded login (DEMO) testing creds are added.
# --------------------------
USERNAME = "avinash"
PASSWORD = "1234"
USERNAME = "Kruthi"
PASSWORD = "1234"

def login_page():
    st.title("🔐 Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state["logged_in"] = True
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

def logout():
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

# --------------------------
# Main App (Fake News UI) do not change this UI part
# --------------------------
def main_app():
    st.title("📰 Fake News Detection")
    st.write("Paste news content below:")

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Run train_model.py first.")
        st.stop()

    model = joblib.load(MODEL_PATH)

    user_input = st.text_area("Enter News Text")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("⚠️ Enter some text")
        else:
            cleaned = clean_text(user_input)
            prediction = model.predict([cleaned])[0]

            if prediction == 1:
                st.error("🚨 FAKE NEWS")
            else:
                st.success("✅ REAL NEWS")

# --------------------------
# App Controller
# --------------------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]:
        logout()
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()