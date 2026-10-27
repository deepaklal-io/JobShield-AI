import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEXT_FEATURES = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORICAL_FEATURES = ["location", "department", "employment_type",
                        "required_experience", "required_education", "industry", "function"]
BINARY_FEATURES = ["telecommuting", "has_company_logo", "has_questions"]
SALARY_FEATURES = ["salary_min", "salary_max", "salary_missing", "salary_unparsed"]

# The 16 raw columns the model needs as input (the "data contract")
RAW_INPUT_COLUMNS = (TEXT_FEATURES + CATEGORICAL_FEATURES
                     + BINARY_FEATURES + ["salary_range"])


def engineer_salary_features(data):
    data = data.copy()
    salary = data["salary_range"].astype("string").str.strip()
    data["salary_missing"] = salary.isna().astype("int64")
    extracted = salary.str.extract(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
    data["salary_min"] = pd.to_numeric(extracted[0], errors="coerce")
    data["salary_max"] = pd.to_numeric(extracted[1], errors="coerce")
    data["salary_unparsed"] = (salary.notna() & extracted[0].isna()).astype("int64")
    return data.drop(columns=["salary_range"])


def prepare_input_types(data):
    data = data.copy()
    for col in TEXT_FEATURES:
        data[col] = data[col].fillna("").astype(str)
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype(object).where(data[col].notna(), "Unknown")
    for col in SALARY_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype("float64")
    for col in BINARY_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="raise")
    return data


def clean_input(data):
    return prepare_input_types(engineer_salary_features(data))


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            *[(f"tfidf_{c}", TfidfVectorizer(max_features=10000, ngram_range=(1, 2)), c)
              for c in TEXT_FEATURES],
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("salary", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler())]), SALARY_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )