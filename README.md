# JobShield-AI
JobShield-AI is a machine-learning web application that detects potentially fraudulent job postings. Enter the details of a job posting and get a fraud-risk score, a risk level and the words and fields that influenced the result.
 
**Live demo:** https://jobshield-ai-tan.vercel.app/

---
 
## Features
 
- Fraud-risk score (0 to 100%) with a **Low / Medium / High** risk level
- Explanation of the features that pushed the score toward fraud or toward legitimate
- Example buttons (real-looking and scam-looking) for quick testing
- Input validation with clear error messages
- Note when the company profile is empty, since this strongly affects the score
- No user data is stored
## Tech stack
 
| Area | Tools |
|---|---|
| Language | Python |
| Machine learning | scikit-learn (TF-IDF, Logistic Regression), pandas, NumPy, SciPy |
| Web app | Flask, HTML/CSS |
| Deployment | Vercel |
 
## Dataset
 
The project uses `fake_job_postings.csv`, a public dataset of 17,880 job postings labelled as real or fraudulent (about 4.8% fraudulent). After removing the `job_id` column and 281 duplicate rows, 17,599 postings remain. They are split once (stratified 70 / 15 / 15, `random_state=42`):
 
| Split | Postings | Fraudulent |
|---|---|---|
| Train | 12,319 | 599 |
| Validation | 2,640 | 129 |
| Test | 2,640 | 128 |
 
## Model
 
- **Models compared:** Logistic Regression, Linear SVM and Random Forest.
- **Final model:** Logistic Regression (`C=10`, class-weighted) with a decision threshold of **0.45**.
- **Class imbalance:** handled with class weights, a stratified split and threshold tuning.
 
### Performance
 
| Data | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Validation | 0.851 | 0.884 | 0.867 | 0.941 |
| Test | 0.856 | 0.883 | 0.869 | 0.948 |
 
On the test set the model caught 113 of 128 fraudulent postings and wrongly flagged 19 genuine ones. Full results are in the `reports/` folder.
 
## Installation
 
Requires Python 3.11 or newer.
 
```bash
# Clone the repository
git clone https://github.com/deepaklal-io/JobShield-AI.git
cd JobShield-AI
 
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # macOS / Linux: source .venv/bin/activate
 
# Install dependencies
pip install -r requirements.txt
```
 
## Usage
 
### Run the web app
 
```bash
python app/app.py
```
 
Open http://127.0.0.1:5000 in your browser, fill in the form and click **Check posting**. Only the job title and description are required.
 
### Use the model in Python
 
```python
from src.utils import load_model, predict_posting
 
pipeline, threshold = load_model("models/job_fraud_pipeline.joblib")
 
result = predict_posting(
    {"title": "Data Analyst", "description": "SQL and Python, 2 years experience."},
    pipeline,
    threshold,
)
print(result)   # {'fraud_probability': ..., 'is_fraud': ..., 'threshold': 0.45}
```
 
### Run the evaluation on the test set
 
```bash
pip install -r requirements-dev.txt
python -m src.evaluate --model models/job_fraud_pipeline.joblib
```
 
### Run the notebooks
 
```bash
pip install -r requirements-dev.txt
jupyter notebook
```
 
Run them in order: `01_eda_and_cleaning`, `02_feature_engineering`, `03_model_development`.
 
## Project structure
 
```
JobShield-AI/
├── app/
│   ├── app.py                      # Flask application
│   └── templates/index.html        # Input form and result page
├── data/raw/fake_job_postings.csv  # Dataset
├── models/job_fraud_pipeline.joblib
├── notebooks/                      # EDA, feature engineering, model development
├── reports/                        # Test metrics, confusion matrix, error analysis
├── src/
│   ├── preprocessing.py            # Cleaning and feature pipeline
│   ├── utils.py                    # load_model() and predict_posting()
│   ├── explain.py                  # Feature explanations
│   └── evaluate.py                 # Test-set evaluation
├── requirements.txt
├── requirements-dev.txt
└── README.md
```
 
## Limitations
 
- A genuine posting with no company profile can score around 50%, because an empty profile is a strong fraud signal in the training data.
- The model was trained on English postings only.
- Some of the highest-weighted features are very specific (certain locations or companies), so the model partly learns individual fraud campaigns.
- scikit-learn must be version 1.8.0 to load the saved model. Retraining on another machine can change metrics by about ±0.01.
## Team
 
| Name | Role |
|---|---|
| Deepak | Dataset, EDA, cleaning and data split |
| Ritika | Model development and prediction pipeline |
| Nainsy | Flask app, explanations and integration |
 

 