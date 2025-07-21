```markdown
# House Price Prediction Flask App

This project provides a simple web application for predicting house prices using a pre-trained machine learning model.

## Project Structure

```

your\_house\_predictor/
├── app.py                  \# Flask backend
├── house\_price\_model.pkl   \# Pre-trained ML model
├── requirements.txt        \# Python dependencies
└── templates/
└── index.html          \# Frontend (HTML/CSS/JS)

````

## Setup & Run

1.  **Place `house_price_model.pkl`:** Ensure this file (from your Jupyter Notebook) is in the `your_house_predictor/` directory. If using Google Colab, download it after saving.
2.  **Install Dependencies:**
    ```bash
    cd your_house_predictor
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` doesn't exist, create it with `Flask`, `pandas`, `scikit-learn`, `numpy` listed, or run `pip freeze > requirements.txt` after installing them.)*
3.  **Run Application:**
    ```bash
    python app.py
    ```
4.  **Access:** Open your browser to `http://127.0.0.1:5000/`.

## Usage

Enter house features in the form and click "Predict Price" to get an instant prediction.

## Model Details

The app uses a scikit-learn `LinearRegression` model within a `Pipeline`. Numerical features are scaled, and `furnishingstatus` is one-hot encoded.

---
**Developed by:** Your Name/Organization
````
