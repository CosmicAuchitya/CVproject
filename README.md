# Beta Product Recommendation Project

A **content-based product recommendation** web app built with **Streamlit**.

The app lets users:

- Search products by name
- Filter by **main category**, **sub category**, **rating ranges**, and **price range**
- Select any product from the filtered table and get **similar product recommendations**
- Re-rank recommendations using **ratings**, **number of ratings**, and **discount percentage**

---

## 1. Tech Stack

- **Python 3.10+**
- **Streamlit** – web UI
- **pandas / numpy** – data handling
- **scikit-learn** – TF–IDF vectorizer and cosine similarity

---

## 2. Project Structure

Typical layout:

```text
project_root/
├─ PRapp.py                 # Streamlit app 
├─ sample_cleaned_products.csv   # Cleaned products dataset
├─ bg_image.png           # Background image for the UI
├─ requirements.txt
└─ README.md
