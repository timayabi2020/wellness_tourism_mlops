import pandas as pd
import streamlit as st
import skops.io as sio

from huggingface_hub import hf_hub_download


MODEL_REPO = "motidev/wellness-tourism-model"
MODEL_FILE = "wellness_tourism_model.skops"


@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    unknown_types = sio.get_untrusted_types(
        file=model_path
    )

    model = sio.load(
        model_path,
        trusted=unknown_types
    )

    return model


model = load_model()

st.title("Wellness Tourism Package Predictor")

st.write(
    "Enter customer information below to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

st.subheader("Customer Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=0.0,
    value=15.0
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Free Lancer",
        "Large Business"
    ]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_persons = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    value=2
)

number_of_followups = st.number_input(
    "Number of Followups",
    min_value=0,
    value=3
)

product_pitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced",
        "Unmarried"
    ]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    value=2
)

passport = st.selectbox(
    "Has Passport?",
    [0, 1]
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Owns Car?",
    [0, 1]
)

number_of_children = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    value=0
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=25000.0
)

input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "DurationOfPitch": [duration_of_pitch],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_persons],
    "NumberOfFollowups": [number_of_followups],
    "ProductPitched": [product_pitched],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income]
})

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "The customer is likely to purchase the Wellness Tourism Package."
        )
    else:
        st.warning(
            "The customer is unlikely to purchase the Wellness Tourism Package."
        )

    st.write(
        f"Purchase probability: **{probability:.2%}**"
    )

    model.prepare(input_data)