import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.impute import SimpleImputer


df_bengaluru = pd.read_csv('bengaluru_house_data_with_has_hall_kitchen_and_bedrooms.csv')

bengaluru_col = ['bath','balcony','Has Hall','Has Kitchen','Number of Bedrooms','price']
df_bengaluru = df_bengaluru[bengaluru_col]

X_bengaluru = df_bengaluru[['bath','balcony','Has Hall','Has Kitchen','Number of Bedrooms']]
y_bengaluru = df_bengaluru['price']


X_train_bengaluru, X_test_bengaluru, y_train_bengaluru, y_test_bengaluru = train_test_split(X_bengaluru, y_bengaluru, test_size=0.25,random_state=42)


imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform both the training and testing data
X_train_bengaluru_imputed = imputer.fit_transform(X_train_bengaluru)
X_test_bengaluru_imputed = imputer.transform(X_test_bengaluru)
lr_bengaluru = LinearRegression()
lr_bengaluru.fit(X_train_bengaluru_imputed, y_train_bengaluru)

pickle.dump(lr_bengaluru, open('model_bengaluru.pkl', 'wb'))








df_mumbai = pd.read_csv('Mumbai1.csv')

mumbai_col = ['sqrt','bhk','Gymnasium','Lift Available','CarParking','Maintenance Staff','24x7 Security','Clubhouse','Intercom','Landscaped Gardens','Indoor Games','Gas Connection','Jogging Track','Swimming Pool','price']
df_mumbai = df_mumbai[mumbai_col]

X_mumbai = df_mumbai[['sqrt','bhk','Gymnasium','Lift Available','CarParking','Maintenance Staff','24x7 Security','Clubhouse','Intercom','Landscaped Gardens','Indoor Games','Gas Connection','Jogging Track','Swimming Pool']]
y_mumbai = df_mumbai['price']


X_train_mumbai, X_test_mumbai, y_train_mumbai, y_test_mumbai = train_test_split(X_mumbai, y_mumbai, test_size=0.25,random_state=42)

lr_mumbai = LinearRegression()
lr_mumbai.fit(X_train_mumbai, y_train_mumbai)
#print(X.shape)

pickle.dump(lr_mumbai, open('model_mumbai.pkl', 'wb'))



df_delhi = pd.read_csv('MagicBricks.csv')

delhi_col = ['Area','BHK','Bathroom','Parking','Price']
df_delhi = df_delhi[delhi_col]

X_delhi = df_delhi[['Area','BHK','Bathroom','Parking']]
y_delhi = df_delhi['Price']


X_train_delhi, X_test_delhi, y_train_delhi, y_test_delhi = train_test_split(X_delhi, y_delhi, test_size=0.25,random_state=42)
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform both the training and testing data
X_train_delhi_imputed = imputer.fit_transform(X_train_delhi)
X_test_delhi_imputed = imputer.transform(X_test_delhi)
lr_delhi = LinearRegression()
lr_delhi.fit(X_train_delhi_imputed, y_train_delhi)


#print(X.shape)

pickle.dump(lr_delhi, open('model_delhi.pkl', 'wb'))















