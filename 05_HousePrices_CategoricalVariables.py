import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer



# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# Read the data
X = pd.read_csv(r'C:\Data\Kaggle\HousePricesAdvancedRegressionTechniques\train.csv', index_col='Id') 
X_test = pd.read_csv(r'C:\Data\Kaggle\HousePricesAdvancedRegressionTechniques\test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)
#X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)
print("---> Train and test split")



# Fill in the lines below: drop columns in training and validation data
drop_X_train =  X_train.select_dtypes(exclude=['object'])
drop_X_valid =  X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())


# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(X_train[col]) == set(X_valid[col])]

same_test_cols=[col for col in object_cols if 
                   set(X_train[col]) == set(X_test[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))

bad_testlabel_cols = list(set(object_cols)-set(same_test_cols))

total_bad_label_cols=list((set(same_test_cols)| set(good_label_cols)))

        
print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
print('\nCategorical columns that will be dropped from the test dataset:', same_test_cols)
print('\nCategorical columns that will be dropped from the total dataset:', total_bad_label_cols)



# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply label encoder 
label_encoder = LabelEncoder()
for col in good_label_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
    

print("MAE from Approach 2 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))


# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
print(sorted(d.items(), key=lambda x: x[1]))


# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)




# Use as many lines of code as you need!
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

#OH_X_train = ____ # Your code here
#OH_X_valid = ____ # Your code here

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

#-------------------------------------------------------------------------------
# Test Data
# Define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(OH_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(OH_X_valid)
print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid))

# Imputation


print("1----->",X_test.info(verbose=True))


X_test_nonobj = X_test.select_dtypes(exclude=['object'])
#print("2----->", X_test_nonobj.head())

testobj_cols = list(set(X_test.columns)-set(X_test_nonobj.columns))
#print("3----->",testobj_cols)
X_test_obj=X_test[testobj_cols]



final_imputer = SimpleImputer(strategy='median')
tmp_X_test = pd.DataFrame(final_imputer.fit_transform(X_test_nonobj))
tmp_X_test.columns=X_test_nonobj.columns
tmp_X_test.index=X_test.index
#print("3b----->",tmp_X_test.head())


# OH
# Number of missing values in each column of testing data
missing_val_count_by_column = (X_test_obj.isnull().sum())
print("4------>", missing_val_count_by_column[missing_val_count_by_column > 0])
X_test_obj=X_test_obj.fillna("NA")
#print("4b------>",X_test_obj.head())

OH_cols_test = pd.DataFrame(OH_encoder.fit_transform(X_test_obj[low_cardinality_cols]))
OH_cols_test.index = X_test.index
#print("6b------>",OH_cols_test.head())




#num_X_test = OH_cols_test2.drop(object_cols, axis=1)
OH_X_test = pd.concat( [tmp_X_test, OH_cols_test], axis=1)
print("6c------>",OH_X_test.info(verbose=True))
print("6d------>", object_cols)

#['GarageYrBlt', 'MasVnrArea', 'LotFrontage']
label_OH_X_test = OH_X_test.drop(['GarageYrBlt', 'MasVnrArea', 'LotFrontage'], axis=1)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(OH_X_train[label_OH_X_test.columns], y_train)


preds_test = model.predict(label_OH_X_test)

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv(r'C:\Data\Kaggle\HousePricesAdvancedRegressionTechniques\submission_OH.csv', index=False)
