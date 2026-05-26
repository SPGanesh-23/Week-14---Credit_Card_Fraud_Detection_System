import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load Credit Card Fraud dataset from Kaggle
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df = pd.read_csv('creditcard.csv')

print('Shape:', df.shape)
print('\nColumn Names:')
print(df.columns.tolist())
df.head()

# Select relevant columns for analysis
# V1-V28 are PCA-transformed anonymized features
cols_to_keep = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']

df = df[cols_to_keep]

print('Selected Columns:', df.columns.tolist())
df.head()

# Check for missing values
print('Missing Values per Column:')
print(df.isnull().sum())

# Fill any missing values with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Drop duplicates
df = df.drop_duplicates()

print('\nData types:')
print(df.dtypes)

print('\nCleaned dataset shape:', df.shape)

from sklearn.preprocessing import StandardScaler

# Normalize the Amount column (Time and V1-V28 are already scaled)
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])

print('Amount_Scaled added. Sample values:')
print(df['Amount_Scaled'].head())

print('Average Transaction Amount')
print(round(df['Amount'].mean(), 2))

print('Total Fraudulent Transactions')
print(df['Class'].sum())

print('\nTotal Normal Transactions')
print((df['Class'] == 0).sum())

print('\nFraud Percentage')
print(round(df['Class'].mean() * 100, 4), '%')

# Summary statistics split by class
print('Descriptive Statistics by Class:')
df.groupby('Class')[['Amount', 'Time']].describe()

# Fraud vs Normal transaction count
class_counts = df['Class'].value_counts().reset_index()
class_counts.columns = ['Class', 'Count']
class_counts['Class'] = class_counts['Class'].map({0: 'Normal', 1: 'Fraud'})

fig = px.bar(class_counts, x='Class', y='Count',
             title='Normal vs Fraudulent Transactions',
             text='Count', color='Class',
             color_discrete_map={'Normal': 'green', 'Fraud': 'red'})
fig.show()

# Average transaction amount: Normal vs Fraud
amount_comparison = df.groupby('Class')['Amount'].mean().reset_index()
amount_comparison['Class'] = amount_comparison['Class'].map({0: 'Normal', 1: 'Fraud'})

fig = px.bar(amount_comparison, x='Class', y='Amount',
             title='Average Transaction Amount: Normal vs Fraud',
             text='Amount', color='Class',
             labels={'Amount': 'Avg Transaction Amount ($)', 'Class': 'Transaction Type'},
             color_discrete_map={'Normal': 'blue', 'Fraud': 'red'})
fig.show()

# Fraud rate by transaction amount range
df['Amount_Range'] = pd.cut(df['Amount'],
                            bins=[0, 10, 50, 200, 500, 1000, 5000, 30000],
                            labels=['0-10', '10-50', '50-200', '200-500', '500-1K', '1K-5K', '5K+'])

amount_fraud = df.groupby('Amount_Range')['Class'].mean().reset_index()
amount_fraud.columns = ['Amount Range', 'Fraud Rate']

fig = px.bar(amount_fraud, x='Amount Range', y='Fraud Rate',
             title='Fraud Rate by Transaction Amount Range',
             text='Fraud Rate', color='Fraud Rate',
             color_continuous_scale='Reds')
fig.show()

# Fraud rate by hour of day (Time is seconds elapsed; convert to hours)
df['Hour'] = (df['Time'] // 3600) % 24

time_fraud = df.groupby('Hour')['Class'].mean().reset_index()
time_fraud.columns = ['Hour of Day', 'Fraud Rate']

fig = px.line(time_fraud, x='Hour of Day', y='Fraud Rate',
              title='Fraud Rate by Hour of Day',
              labels={'Fraud Rate': 'Fraud Rate', 'Hour of Day': 'Hour (0-23)'},
              markers=True, template='plotly_white')
fig.show()

# Scatter: Time vs Amount colored by Class
sample_df = df.sample(n=5000, random_state=42)

fig = px.scatter(sample_df, x='Time', y='Amount',
                 color=sample_df['Class'].map({0: 'Normal', 1: 'Fraud'}),
                 title='Time vs Transaction Amount (Sample of 5000)',
                 labels={'Time': 'Time (seconds)', 'Amount': 'Amount ($)', 'color': 'Transaction Type'},
                 template='plotly_white')
fig.show()

# Amount distribution by fraud status
fig = px.histogram(df, x='Amount',
                   color=df['Class'].map({0: 'Normal', 1: 'Fraud'}),
                   title='Transaction Amount Distribution by Class',
                   labels={'Amount': 'Amount ($)', 'color': 'Transaction Type'},
                   barmode='overlay', template='plotly_white', nbins=60)
fig.show()

# Correlation heatmap with key features + Class
key_cols = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V10', 'V12', 'V14', 'V17', 'Class']
corr = df[key_cols].corr()

fig = px.imshow(corr, text_auto='.2f',
                title='Correlation Heatmap (Key Features vs Class)',
                color_continuous_scale='RdBu_r',
                width=800, height=700)
fig.show()

# Transaction Amount Distribution (log scale for clarity)
fig = px.histogram(df[df['Amount'] > 0], x='Amount',
                   title='Transaction Amount Distribution',
                   labels={'Amount': 'Transaction Amount ($)'},
                   nbins=60, template='plotly_white', log_y=True)
fig.show()

# Box plot: Amount by fraud class
fig = px.box(df, x=df['Class'].map({0: 'Normal', 1: 'Fraud'}),
             y='Amount',
             title='Transaction Amount by Class (Box Plot)',
             labels={'x': 'Transaction Type', 'Amount': 'Amount ($)'},
             color=df['Class'].map({0: 'Normal', 1: 'Fraud'}),
             color_discrete_map={'Normal': 'blue', 'Fraud': 'red'})
fig.show()

# Distribution of top correlated PCA features (V14, V12, V10, V4)
top_features = ['V14', 'V12', 'V10', 'V4']
fraud_df = df[df['Class'] == 1]
normal_df = df[df['Class'] == 0].sample(n=len(fraud_df)*5, random_state=42)
viz_df = pd.concat([fraud_df, normal_df])

for feat in top_features:
    fig = px.histogram(viz_df, x=feat,
                       color=viz_df['Class'].map({0: 'Normal', 1: 'Fraud'}),
                       title=f'{feat} Distribution: Normal vs Fraud',
                       barmode='overlay', template='plotly_white')
    fig.show()

# Classify transactions into Low, Medium, High Risk based on Amount
def classify_risk(amount):
    if amount < 50:
        return 'Low Risk'
    elif amount < 500:
        return 'Medium Risk'
    else:
        return 'High Risk'

df['Risk_Level'] = df['Amount'].apply(classify_risk)

risk_counts = df['Risk_Level'].value_counts().reset_index()
risk_counts.columns = ['Risk Level', 'Count']

fig = px.bar(risk_counts, x='Risk Level', y='Count',
             title='Transaction Risk Classification',
             text='Count', color='Risk Level',
             color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'})
fig.show()

# Fraud Rate per Risk Level
risk_fraud = df.groupby('Risk_Level')['Class'].mean().reset_index()
risk_fraud.columns = ['Risk Level', 'Fraud Rate']

fig = px.bar(risk_fraud, x='Risk Level', y='Fraud Rate',
             title='Fraud Rate by Risk Level',
             text='Fraud Rate', color='Risk Level',
             color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'})
fig.show()

# Independent variables: Time, Amount
# Dependent variable: Class (fraud probability proxy)
X = df[['Time', 'Amount']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training set size:', X_train.shape)
print('Testing set size:', X_test.shape)

# Build and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print('Model trained successfully.')
print('Coefficients (Time, Amount):', model.coef_)
print('Intercept:', model.intercept_)

# Predict fraud likelihood
y_pred = model.predict(X_test)

print('Sample Predictions:', y_pred[:5])
print('Actual Values:     ', y_test.values[:5])

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('R2 Score:', round(r2, 4))
print('MSE     :', round(mse, 6))
print('RMSE    :', round(rmse, 6))

# Residual Plot
residuals = y_test - y_pred

fig = px.scatter(x=y_pred, y=residuals,
                 title='Residual Plot (Predicted vs Residuals)',
                 labels={'x': 'Predicted Fraud Score', 'y': 'Residuals'},
                 template='plotly_white')
fig.add_hline(y=0, line_dash='dash', line_color='red')
fig.show()

from dash import Dash, dcc, html, Input, Output

# Initialize Dash app
app = Dash(__name__)

app.layout = html.Div([

    html.H1('Credit Card Fraud Detection Dashboard', style={'textAlign': 'center'}),

    html.Br(),

    # Risk Level Filter
    html.Div([
        html.Label('Select Risk Level'),
        dcc.Dropdown(
            id='risk-dropdown',
            options=[{'label': r, 'value': r} for r in sorted(df['Risk_Level'].unique())],
            value='High Risk',
            clearable=False
        )
    ], style={'width': '40%', 'margin': 'auto'}),

    html.Br(),

    # Amount Range Slider
    html.Div([
        html.Label('Select Amount Range ($)'),
        dcc.RangeSlider(
            id='amount-slider',
            min=0, max=int(df['Amount'].max()),
            step=50,
            value=[0, 2000],
            marks={0: '$0', 500: '$500', 1000: '$1K', 2000: '$2K', 5000: '$5K'}
        )
    ], style={'width': '60%', 'margin': 'auto'}),

    html.Br(),

    # Charts
    dcc.Graph(id='scatter-chart'),
    dcc.Graph(id='bar-chart'),
    dcc.Graph(id='pie-chart'),
    dcc.Graph(id='hist-chart')

])

@app.callback(
    [Output('scatter-chart', 'figure'),
     Output('bar-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('hist-chart', 'figure')],

    [Input('risk-dropdown', 'value'),
     Input('amount-slider', 'value')]
)
def update_dashboard(selected_risk, amount_range):

    filtered_df = df[
        (df['Risk_Level'] == selected_risk) &
        (df['Amount'] >= amount_range[0]) &
        (df['Amount'] <= amount_range[1])
    ].copy()

    filtered_df['Transaction Type'] = filtered_df['Class'].map({0: 'Normal', 1: 'Fraud'})

    # 1. Scatter: Time vs Amount
    scatter_fig = px.scatter(
        filtered_df.sample(min(1000, len(filtered_df)), random_state=42),
        x='Time', y='Amount',
        color='Transaction Type',
        color_discrete_map={'Normal': 'green', 'Fraud': 'red'},
        title='Time vs Transaction Amount',
        template='plotly_dark'
    )

    # 2. Bar: Fraud count by Hour
    hour_group = filtered_df.groupby('Hour')['Class'].sum().reset_index()
    hour_group.columns = ['Hour', 'Fraud Count']

    bar_fig = px.bar(
        hour_group,
        x='Hour', y='Fraud Count',
        color='Fraud Count',
        title='Fraud Count by Hour of Day',
        template='plotly_dark',
        color_continuous_scale='Reds'
    )

    # 3. Pie: Normal vs Fraud distribution
    pie_data = filtered_df['Transaction Type'].value_counts().reset_index()
    pie_data.columns = ['Transaction Type', 'Count']

    pie_fig = px.pie(
        pie_data,
        names='Transaction Type', values='Count',
        title='Normal vs Fraud Distribution',
        color='Transaction Type',
        color_discrete_map={'Normal': 'green', 'Fraud': 'red'},
        template='plotly_dark'
    )

    # 4. Histogram: Amount distribution by class
    hist_fig = px.histogram(
        filtered_df, x='Amount',
        color='Transaction Type',
        color_discrete_map={'Normal': 'green', 'Fraud': 'red'},
        title='Transaction Amount Distribution',
        labels={'Amount': 'Amount ($)'},
        barmode='overlay',
        template='plotly_dark'
    )

    return scatter_fig, bar_fig, pie_fig, hist_fig


if __name__ == '__main__':
    app.run(debug=True)

print('=== Key Insights ===')
print()
print('1. Overall Fraud Rate:')
print('  ', round(df['Class'].mean() * 100, 4), '%')
print()
print('2. Avg Transaction Amount - Normal vs Fraud:')
print(df.groupby('Class')['Amount'].mean().rename({0: 'Normal', 1: 'Fraud'}))
print()
print('3. Fraud Rate by Risk Level:')
print(df.groupby('Risk_Level')['Class'].mean().sort_values(ascending=False))
print()
print('4. Top 3 Hours with Highest Fraud Rate:')
print(df.groupby('Hour')['Class'].mean().sort_values(ascending=False).head(3))
print()
print('5. Linear Regression R2 Score:', round(r2, 4))
print()
print('Conclusion:')
print('Fraudulent transactions tend to have lower amounts but appear at specific hours.')
print('PCA features V14, V12, V10, and V4 are most negatively correlated with fraud.')
print('Risk classification and dashboard filters help isolate suspicious transaction clusters.')
print('Linear Regression provides a baseline trend; ensemble methods (e.g. Random Forest) are recommended for production.')