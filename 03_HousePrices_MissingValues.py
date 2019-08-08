import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Read the data
X_full = pd.read_csv(r'C:\Data\Kaggle\HousePricesAdvancedRegressionTechniques\train.csv', index_col='Id')
X_test_full = pd.read_csv(r'C:\Data\Kaggle\HousePricesAdvancedRegressionTechniques\test.csv', index_col='Id')


# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
print("--->Train and test split")


print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# Fill in the lines below: imputation
my_imputer = SimpleImputer() # Your code here
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns


# Imputation
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

# Imputation removed column names; put them back
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

# Preprocessed training and validation features
final_X_train = imputed_X_train
final_X_valid = imputed_X_valid


# Define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid))

# Imputation
final_imputer = SimpleImputer(strategy='median')

# Fill in the line below: preprocess test data
final_X_test = pd.DataFrame(final_imputer.fit_transform(X_test))

# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)
# Preprocess test data
final_X_test = pd.DataFrame(final_imputer.transform(X_test))
# Get test predictions
preds_test = model.predict(final_X_test)


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv(r'C:\Data\Kaggle\HousePricesAdvancedRegressionTechniques\submission_missing.csv', index=False)

