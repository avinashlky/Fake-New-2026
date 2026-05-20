# Fake-New-2026




Fake news  by Avinash-2026
Step 1 — make sure python version and virt envs are set
cd /path/to/fake_news

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install streamlit pandas scikit-learn joblib

python -m streamlit run app.py


Step 1 — Confirm your dataset file
In your project folder (/Users/avinashrathod/Downloads/fake_news) run:

ls
Step 2 — Create a training script train_model.py
Create a new file train_model.py in the same folder as app.py.
Step 3 — Run training inside your virtual environment

source .venv/bin/activate
python train_model.py


After success, run:
ls

any issues reach me on 9900225993
