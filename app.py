import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
d=pd.read_csv('/content/data.csv')
d.head()
z=d.drop(columns=['Unnamed: 32'],axis=1)
z

target = 'area_mean'
X = z.drop(columns=['id', 'diagnosis'])
y = z[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X)
print(y)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)
poly_regressor = LinearRegression()
poly_regressor.fit(X_train_poly, y_train_poly)

y_pred_poly = poly_regressor.predict(X_test_poly)

mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
r2_poly = r2_score(y_test_poly, y_pred_poly)
mse_poly, r2_poly

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (Degree=2)": poly_regressor,
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=100),
    "Support Vector Regressor": SVR(kernel='rbf'),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    if name == "Polynomial Regression (Degree=2)":
        
        model.fit(X_train_poly, y_train_poly)
        y_pred = model.predict(X_test_poly)
    else:
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}

results

import numpy as np

best_model = RandomForestRegressor(random_state=42, n_estimators=100)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

prediction_stats = {
    "Mean Prediction": np.mean(y_pred),
    "Max Prediction": np.max(y_pred),
    "Min Prediction": np.min(y_pred),
    "Standard Deviation": np.std(y_pred)
}

mse_best = mean_squared_error(y_test, y_pred)
r2_best = r2_score(y_test, y_pred)

model_performance = {
    "MSE": mse_best,
    "R2": r2_best,
    "Prediction Stats": prediction_stats
}

model_performance

if best_model == "Random Forest":
  best_model = RandomForestRegressor()
elif best_model == "Linear Regression":
  best_model = LinearRegression()
elif best_model == "Decision Tree":
  best_model = DecisionTreeRegressor()
elif best_model == "Support Vector Machine":
  best_model = SVR()
elif best_model == "Multiple Regression":
  best_model = LinearRegression()
elif best_model == "Polynomial Regression":
  best_model = LinearRegression()

best_model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

random_forest = RandomForestRegressor(random_state=42, n_estimators=100)
gradient_boosting = GradientBoostingRegressor(random_state=42)

random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

gradient_boosting.fit(X_train, y_train)
y_pred_gb = gradient_boosting.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

if r2_rf > r2_gb:
    best_model = "Random Forest Regressor"
    best_model_instance = random_forest
    predictions = y_pred_rf
else:
    best_model = "Gradient Boosting Regressor"
    best_model_instance = gradient_boosting
    predictions = y_pred_gb

input_output_df = pd.DataFrame(X_test, columns=X.columns)
input_output_df['Actual'] = y_test.values
input_output_df['Predicted'] = predictions

print(f"The best model is: {best_model}")
print(input_output_df.head())
