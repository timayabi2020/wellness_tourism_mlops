import json
from pathlib import Path

import pandas as pd
import skops.io as sio
import streamlit as st

from huggingface_hub import hf_hub_download
from jsonschema import Draft202012Validator


st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Pin the Hugging Face repository and artifact used for inference.
MODEL_REPO = "motidev/wellness-tourism-model"
MODEL_FILE = "wellness_tourism_model.skops"
SCHEMA_FILE = Path(__file__).parents[1] / "schema" / "inference_schema.v1.json"

with SCHEMA_FILE.open(encoding="utf-8") as schema_file:
    INFERENCE_SCHEMA = json.load(schema_file)

Draft202012Validator.check_schema(INFERENCE_SCHEMA)
SCHEMA_VALIDATOR = Draft202012Validator(INFERENCE_SCHEMA)
SCHEMA_VERSION = INFERENCE_SCHEMA["x-schema-version"]


@st.cache_resource
def load_model():
    # Download once per Streamlit process and reuse the cached fitted pipeline.
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

    # Inspect the artifact before explicitly trusting the types required by skops.
    unknown_types = sio.get_untrusted_types(file=model_path)
    return sio.load(model_path, trusted=unknown_types)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,600&display=swap');

    :root {
        --ink: #18332d;
        --muted: #66756f;
        --forest: #176b55;
        --forest-deep: #0f4d3e;
        --coral: #e66a4e;
        --line: #dce6e1;
        --paper: #fbfcfa;
    }

    html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
        color: var(--ink);
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--paper);
        background-image:
            linear-gradient(rgba(23, 107, 85, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(23, 107, 85, 0.035) 1px, transparent 1px);
        background-size: 32px 32px;
    }

    [data-testid="stHeader"] { background: transparent; }

    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }

    .brand-mark {
        color: var(--forest);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin-bottom: 0.9rem;
        text-transform: uppercase;
    }

    .hero-title {
        font-family: "Newsreader", Georgia, serif;
        font-size: clamp(2.8rem, 5vw, 4.8rem);
        font-weight: 600;
        line-height: 0.96;
        margin: 0;
        max-width: 8ch;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.65;
        margin: 1.25rem 0 1.5rem;
        max-width: 34rem;
    }

    .hero-photo {
        background-image: linear-gradient(180deg, rgba(15, 77, 62, 0.02), rgba(15, 77, 62, 0.38)),
            url("https://images.unsplash.com/photo-1540555700478-4be289fbecef?auto=format&fit=crop&w=1200&q=85");
        background-position: center;
        background-size: cover;
        border-radius: 6px;
        height: 280px;
        margin: 1.6rem 0 1.4rem;
        position: relative;
    }

    .hero-photo span {
        background: rgba(251, 252, 250, 0.94);
        bottom: 1rem;
        color: var(--forest-deep);
        font-size: 0.78rem;
        font-weight: 700;
        left: 1rem;
        padding: 0.5rem 0.7rem;
        position: absolute;
    }

    .trust-row {
        border-top: 1px solid var(--line);
        display: grid;
        gap: 0.8rem;
        grid-template-columns: repeat(3, 1fr);
        padding-top: 1rem;
    }

    .trust-row strong {
        color: var(--forest-deep);
        display: block;
        font-size: 1.05rem;
    }

    .trust-row span { color: var(--muted); font-size: 0.72rem; }

    .form-intro {
        align-items: end;
        border-bottom: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        margin-bottom: 1.35rem;
        padding-bottom: 1rem;
    }

    .form-heading {
        font-family: "Newsreader", Georgia, serif;
        font-size: 2rem;
        font-weight: 600;
        line-height: 1.1;
        margin: 0;
    }

    .form-intro span { color: var(--muted); font-size: 0.8rem; }

    .section-label {
        color: var(--forest);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin: 0.6rem 0 0.8rem;
        text-transform: uppercase;
    }

    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid var(--line);
        border-radius: 6px;
        box-shadow: 0 18px 50px rgba(20, 61, 51, 0.08);
        padding: 1.5rem 1.65rem 1.65rem;
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #334b45;
        font-size: 0.82rem;
        font-weight: 600;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: #f5f8f6;
        border-color: #dbe6e1;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: var(--forest);
        border: 0;
        border-radius: 4px;
        color: white;
        font-weight: 700;
        min-height: 3.1rem;
        width: 100% !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--forest-deep);
        color: white;
    }

    .result-panel {
        background: #ffffff;
        border-left: 5px solid var(--forest);
        box-shadow: 0 12px 35px rgba(20, 61, 51, 0.08);
        margin-top: 1.2rem;
        padding: 1.25rem 1.4rem;
    }

    .result-panel.negative { border-left-color: var(--coral); }

    .result-kicker {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .result-heading {
        font-family: "Newsreader", Georgia, serif;
        font-size: 1.8rem;
        margin: 0.25rem 0;
    }

    .result-score {
        color: var(--forest-deep);
        font-size: 2rem;
        font-weight: 700;
    }

    .result-track {
        background: #e7eeeb;
        height: 8px;
        margin: 0.8rem 0 0.65rem;
        overflow: hidden;
    }

    .result-fill { background: var(--forest); height: 100%; }
    .negative .result-fill { background: var(--coral); }

    .result-copy {
        color: var(--muted);
        font-size: 0.86rem;
        margin: 0;
    }

    @media (max-width: 900px) {
        .block-container { padding-top: 1.2rem; }
        .hero-title { font-size: 3rem; max-width: none; }
        .hero-photo { height: 220px; }
        .trust-row { margin-bottom: 1.5rem; }
    }

    @media (max-width: 640px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero-title { font-size: 2.55rem; }
        .hero-copy { font-size: 0.94rem; }
        .form-intro { align-items: start; flex-direction: column; gap: 0.25rem; }
        div[data-testid="stForm"] { padding: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model()

if list(model.feature_names_in_) != INFERENCE_SCHEMA["required"]:
    st.error("The deployed model is incompatible with the inference schema.")
    st.stop()

brand_column, form_column = st.columns([0.82, 1.18], gap="large")

with brand_column:
    st.markdown(
        """
        <div class="brand-mark">Verdant Journeys / Decision Studio</div>
        <h1 class="hero-title">Find the travelers ready to say yes.</h1>
        <p class="hero-copy">
            Turn a customer profile and sales conversation into a clear purchase
            propensity signal for your wellness package team.
        </p>
        <div class="hero-photo">
            <span>WELLNESS TOURISM / PURCHASE INTENT</span>
        </div>
        <div class="trust-row">
            <div><strong>18</strong><span>customer signals</span></div>
            <div><strong>1 click</strong><span>to score intent</span></div>
            <div><strong>v{SCHEMA_VERSION}</strong><span>inference schema</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with form_column:
    st.markdown(
        """
        <div class="form-intro">
            <div class="form-heading">Customer profile</div>
            <span>All fields required</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown('<div class="section-label">01 / Traveler</div>', unsafe_allow_html=True)
        traveler_left, traveler_right = st.columns(2, gap="medium")

        with traveler_left:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            occupation = st.selectbox(
                "Occupation",
                ["Salaried", "Small Business", "Free Lancer", "Large Business"],
            )
            marital_status = st.selectbox(
                "Marital status",
                ["Single", "Married", "Divorced", "Unmarried"],
            )
            designation = st.selectbox(
                "Designation",
                ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
            )

        with traveler_right:
            gender = st.selectbox("Gender", ["Male", "Female"])
            monthly_income = st.number_input(
                "Monthly income", min_value=0.0, value=25000.0, step=1000.0
            )
            passport = st.selectbox(
                "Passport", [0, 1], format_func=lambda value: "Yes" if value else "No"
            )
            own_car = st.selectbox(
                "Owns a car", [0, 1], format_func=lambda value: "Yes" if value else "No"
            )

        st.markdown('<div class="section-label">02 / Trip context</div>', unsafe_allow_html=True)
        trip_left, trip_right = st.columns(2, gap="medium")

        with trip_left:
            city_tier = st.selectbox(
                "City tier", [1, 2, 3], format_func=lambda value: f"Tier {value}"
            )
            number_of_persons = st.number_input("Travelers", min_value=1, value=2)
            number_of_children = st.number_input("Children traveling", min_value=0, value=0)

        with trip_right:
            number_of_trips = st.number_input("Trips per year", min_value=0, value=2)
            preferred_property_star = st.selectbox(
                "Preferred property", [3, 4, 5], format_func=lambda value: f"{value}-star"
            )
            product_pitched = st.selectbox(
                "Package pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
            )

        st.markdown('<div class="section-label">03 / Sales interaction</div>', unsafe_allow_html=True)
        sales_left, sales_right = st.columns(2, gap="medium")

        with sales_left:
            type_of_contact = st.selectbox(
                "Type of contact", ["Self Enquiry", "Company Invited"]
            )
            duration_of_pitch = st.number_input(
                "Pitch duration (minutes)", min_value=0.0, value=15.0, step=1.0
            )

        with sales_right:
            number_of_followups = st.number_input("Follow-ups", min_value=0, value=3)
            pitch_satisfaction_score = st.selectbox(
                "Pitch satisfaction",
                [1, 2, 3, 4, 5],
                index=2,
                format_func=lambda value: f"{value} / 5",
            )

        submitted = st.form_submit_button(
            "Calculate purchase likelihood",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        payload = {
            "Age": age,
            "TypeofContact": type_of_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": number_of_persons,
            "NumberOfFollowups": number_of_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_of_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_satisfaction_score,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": number_of_children,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }

        errors = sorted(SCHEMA_VALIDATOR.iter_errors(payload), key=lambda error: list(error.path))
        if errors:
            st.error(f"Input does not satisfy schema v{SCHEMA_VERSION}: {errors[0].message}")
            st.stop()

        input_data = pd.DataFrame([payload], columns=INFERENCE_SCHEMA["required"])

        # Return both the binary decision and positive-class purchase probability.
        prediction = model.predict(input_data)[0]
        probability = float(model.predict_proba(input_data)[0][1])
        result_class = "positive" if prediction == 1 else "negative"
        result_heading = (
            "Strong purchase potential" if prediction == 1 else "Needs further nurturing"
        )
        result_copy = (
            "This profile is likely to convert. Prioritize a timely, personalized follow-up."
            if prediction == 1
            else "This profile is less likely to convert right now. Consider refining the offer or timing."
        )

        st.markdown(
            f"""
            <div class="result-panel {result_class}">
                <div class="result-kicker">Prediction result</div>
                <div class="result-heading">{result_heading}</div>
                <div class="result-score">{probability:.0%}</div>
                <div class="result-track">
                    <div class="result-fill" style="width: {probability:.1%}"></div>
                </div>
                <p class="result-copy">{result_copy}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
