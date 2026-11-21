import sys
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so "src" can be imported

from flask import Flask, render_template, request
from src.explain import explain_posting
from src.preprocessing import RAW_INPUT_COLUMNS, BINARY_FEATURES
from src.utils import load_model, predict_posting

MODEL_PATH = ROOT / "models" / "job_fraud_pipeline.joblib"

TEXTAREAS = [
    ("company_profile", "Company profile"),
    ("description", "Description *"),
    ("requirements", "Requirements"),
    ("benefits", "Benefits"),
]
SHORT_FIELDS = [
    ("location", "Location (e.g. US, CA, San Francisco)"),
    ("department", "Department"),
    ("industry", "Industry"),
    ("function", "Function"),
    ("salary_range", "Salary range (e.g. 40000-60000)"),
]
DROPDOWNS = [
    ("employment_type", "Employment type",
     ["Full-time", "Part-time", "Contract", "Temporary", "Other"]),
    ("required_experience", "Required experience",
     ["Internship", "Entry level", "Associate", "Mid-Senior level",
      "Director", "Executive", "Not Applicable"]),
    ("required_education", "Required education",
     ["High School or equivalent", "Some High School Coursework",
      "Some College Coursework Completed", "Vocational",
      "Vocational - HS Diploma", "Vocational - Degree", "Certification",
      "Associate Degree", "Bachelor's Degree", "Master's Degree",
      "Doctorate", "Professional", "Unspecified"]),
]
CHECKBOXES = [
    ("telecommuting", "Remote job"),
    ("has_company_logo", "Has company logo"),
    ("has_questions", "Has screening questions"),
]
SALARY_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?")

MAX_LENGTHS = {
    "title": 200, "company_profile": 10000, "description": 20000,
    "requirements": 10000, "benefits": 10000,
    "location": 100, "department": 100, "industry": 100, "function": 100,
}
ALLOWED = {name: set(options) for name, _, options in DROPDOWNS}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1_000_000  # reject requests over about 1 MB
try:
    pipeline, threshold = load_model(MODEL_PATH)
    LOAD_ERROR = None
except Exception as exc:
    pipeline, threshold, LOAD_ERROR = None, None, str(exc)


def risk_level(p):
    if p >= 0.70:
        return "High"
    if p >= threshold:
        return "Medium"
    return "Low"

def validate(posting):
    """Return an error message, or None if the input is fine."""
    if not posting["title"] or not posting["description"]:
        return "Title and description are required."
    if len(posting["title"]) < 3:
        return "Title is too short (at least 3 characters)."
    if len(posting["description"]) < 20:
        return "Description is too short (at least 20 characters)."

    for col, limit in MAX_LENGTHS.items():
        if len(posting[col]) > limit:
            return f"{col.replace('_', ' ').capitalize()} is too long (maximum {limit} characters)."

    for col, allowed in ALLOWED.items():
        if posting[col] and posting[col] not in allowed:
            return f"Invalid value for {col.replace('_', ' ')}."

    salary = posting["salary_range"]
    if salary:
        if not SALARY_PATTERN.fullmatch(salary):
            return "Salary range must look like 40000-60000 (numbers only), or leave it blank."
        low, high = (float(x) for x in re.split(r"\s*-\s*", salary))
        if low > high:
            return "Salary range: the first number must not be larger than the second."
    return None
def page(form=None, result=None, error=None):
    return render_template(
        "index.html", form=form or {}, result=result, error=error,
        textareas=TEXTAREAS, short_fields=SHORT_FIELDS,
        dropdowns=DROPDOWNS, checkboxes=CHECKBOXES)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return page(error=LOAD_ERROR)

    form = request.form.to_dict()
    if LOAD_ERROR:
        return page(form, error=LOAD_ERROR)

    posting = {}
    for col in RAW_INPUT_COLUMNS:
        if col in BINARY_FEATURES:
            posting[col] = "1" if form.get(col) else "0"
        else:
            posting[col] = form.get(col, "").strip()

    problem = validate(posting)
    if problem:
        return page(form, error=problem)

    try:
        result = predict_posting(posting, pipeline, threshold)
    except Exception as exc:
        return page(form, error=f"Could not score this posting: {exc}")

    result["percent"] = round(result["fraud_probability"] * 100, 1)
    result["risk_level"] = risk_level(result["fraud_probability"])
    result["notes"] = []

    if not posting["company_profile"]:
        result["notes"].append(
            "No company profile provided. In our training data, about 17.5% of "
            "postings without one were fraudulent, versus 2.0% of postings with one, "
            "so adding it can change the score a lot.")

    try:
        result["explanation"] = explain_posting(posting, pipeline)

        # Display only the features supporting the final prediction.
        if result["is_fraud"]:
            result["relevant_features"] = (
                result["explanation"]["toward_fraud"]
            )
            result["explanation_heading"] = "Features Supporting Fraud Prediction"
        else:
            result["relevant_features"] = (
                result["explanation"]["toward_legit"]
            )
            result["explanation_heading"] = "Features Supporting Legitimate Prediction"

        fraud_features = result["explanation"]["toward_fraud"]
        legit_features = result["explanation"]["toward_legit"]

        result["result_label"] = (
            "Likely Fraudulent" if result["is_fraud"]
            else "Looks Legitimate"
        )

        # Build an explanation using actual model contributions.
        supporting_features = (
            fraud_features if result["is_fraud"] else legit_features
        )

        if supporting_features:
            reasons = ", ".join(
                item["feature"] for item in supporting_features[:3]
            )
            result["why_result"] = (
                f"The model reached this result based on several influential "
                f"features in the posting, including {reasons}."
            )
        else:
            result["why_result"] = (
                "The model produced this result, but no strong individual "
                "feature contributions were available."
            )

        # Extract actual text terms that pushed the prediction toward fraud.
        import re

        result["suspicious_keywords"] = []

        text_fields = {
            "title",
            "description",
            "requirements",
            "benefits",
            "company_profile"
        }

        for item in fraud_features:
            match = re.match(
                r'^(' + '|'.join(text_fields) + r'): "(.*)"$',
                item["feature"]
            )

            if match:
                keyword = match.group(2)

                if keyword not in result["suspicious_keywords"]:
                    result["suspicious_keywords"].append(keyword)

        # Keep only the top 8 text signals.
        result["suspicious_keywords"] = result["suspicious_keywords"][:8]

        # Remove duplicates while preserving order.
        result["suspicious_keywords"] = list(
            dict.fromkeys(result["suspicious_keywords"])
        )[:8]

    except Exception:
        app.logger.exception("Explanation failed")

        result["explanation"] = {
            "method": "unavailable",
            "toward_fraud": [],
            "toward_legit": []
        }

        result["result_label"] = (
            "Likely Fraudulent" if result["is_fraud"]
            else "Looks Legitimate"
        )

        result["why_result"] = "A detailed explanation is currently unavailable."
        result["suspicious_keywords"] = []

    return page(form, result=result)


if __name__ == "__main__":
    app.run(debug=False)