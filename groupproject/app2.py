import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
import dash
from dash import dcc, html, Input, Output, State

# Loading data
df_loans = pd.read_csv("train.csv")

# Define features set
X = df_loans.drop("Credit_History", axis=1)
y = df_loans["Credit_History"]

# Handle missing values in the target vector by using the mode of y
y_mode = y.mode()[0]
y = y.fillna(y_mode)

# Oversample the minority class (Credit_History = 0) using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Select numeric and categorical features
numeric_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
categorical_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

# Create the preprocessing pipelines for both numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Preprocess the entire dataset using the preprocessor
X_preprocessed = preprocessor.fit_transform(X_resampled)

# Create the decision tree classifier instance
model = DecisionTreeClassifier()

# Fit the model with the preprocessed data and target labels
model.fit(X_preprocessed, y_resampled)

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Loan Eligibility Prediction"),
    html.Div([
        html.Label("Applicant Income:"),
        dcc.Input(id="applicant-income", type="number" ),
    ]),
    html.Div([
        html.Label("Coapplicant Income:"),
        dcc.Input(id="coapplicant-income", type="number", value=0),
    ]),
    html.Div([
        html.Label("Loan Amount:"),
        dcc.Input(id="loan-amount", type="number"),
    ]),
    html.Div([
        html.Label("Loan Amount Term:"),
        dcc.Input(id="loan-amount-term", type="number", value=360),
    ]),
    html.Div([
        html.Label("Gender:"),
        dcc.Dropdown(
            id="gender",
            options=[
                {"label": "Male", "value": "Male"},
                {"label": "Female", "value": "Female"},
            ],
            value="Male",
        ),
    ]),
    html.Div([
        html.Label("Married:"),
        dcc.Dropdown(
            id="married",
            options=[
                {"label": "Yes", "value": "Yes"},
                {"label": "No", "value": "No"},
            ],
            value="Yes",
        ),
    ]),
    html.Div([
        html.Label("Dependents:"),
        dcc.Dropdown(
            id="dependents",
            options=[
                {"label": "0", "value": "0"},
                {"label": "1", "value": "1"},
                {"label": "2", "value": "2"},
                {"label": "3+", "value": "3+"},
            ],
            value="0",
        ),
    ]),
    html.Div([
        html.Label("Education:"),
        dcc.Dropdown(
            id="education",
            options=[
                {"label": "Graduate", "value": "Graduate"},
                {"label": "Not Graduate", "value": "Not Graduate"},
            ],
            value="Graduate",
        ),
    ]),
    html.Div([
        html.Label("Self Employed:"),
        dcc.Dropdown(
            id="self-employed",
            options=[
                {"label": "Yes", "value": "Yes"},
                {"label": "No", "value": "No"},
            ],
            value="No",
        ),
    ]),
    html.Div([
        html.Label("Property Area:"),
        dcc.Dropdown(
            id="property-area",
            options=[
                {"label": "Rural", "value": "Rural"},
                {"label": "Semiurban", "value": "Semiurban"},
                {"label": "Urban", "value": "Urban"},
            ],
            value="Rural",
        ),
    ]),
    
    html.Button("Check Eligibility", id="check-eligibility"),
    html.Div(id="eligibility-result"),
])


@app.callback(
    Output("eligibility-result", "children"),
    Input("check-eligibility", "n_clicks"),
    State("applicant-income", "value"),
    State("coapplicant-income", "value"),
    State("loan-amount", "value"),
    State("loan-amount-term", "value"),
    State("gender", "value"),
    State("married", "value"),
    State("dependents", "value"),
    State("education", "value"),
    State("self-employed", "value"),
    State("property-area", "value"),
)
def check_loan_eligibility(n_clicks, applicant_income, coapplicant_income, loan_amount, loan_amount_term, gender,
                           married, dependents, education, self_employed, property_area):
    # Create a dictionary to hold the input values
    input_data = {
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_amount_term],
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "Property_Area": [property_area],
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data)

    # Preprocess the input data using the preprocessor
    input_preprocessed = preprocessor.transform(input_df)

    # Make predictions using the trained model
    eligibility = model.predict(input_preprocessed)

    if eligibility[0] == 1:
        return "Congratulations! You are eligible for a loan."
    else:
        return "Sorry, you are not eligible for a loan."


if __name__ == "__main__":
    app.run_server(debug=True)
