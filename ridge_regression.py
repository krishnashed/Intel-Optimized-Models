from timeit import default_timer as timer
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import psycopg2
from datetime import datetime
from IPython.display import HTML
warnings.filterwarnings('ignore')

DB_NAME = "aiml_optimizations"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_HOST = "15.207.20.67"
DB_PORT = "5432"

try:
	conn = psycopg2.connect(database=DB_NAME,
							user=DB_USER,
							password=DB_PASS,
							host=DB_HOST,
							port=DB_PORT)
	print("Database connected successfully")
except:
	print("Database not connected successfully")

dataset = 'Airlines_DepDelay_10M'
x, y = fetch_openml(name=dataset, return_X_y=True)


for col in ['UniqueCarrier', 'Origin', 'Dest']:
    le = LabelEncoder().fit(x[col])
    x[col] = le.transform(x[col])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape



from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler_x = MinMaxScaler()
scaler_y = StandardScaler()



y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)

scaler_x.fit(x_train)
x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)

scaler_y.fit(y_train)
y_train = scaler_y.transform(y_train).ravel()
y_test = scaler_y.transform(y_test).ravel()



from sklearnex import patch_sklearn
patch_sklearn()



from sklearn.linear_model import Ridge

params = {
    "alpha": 0.3,    
    "fit_intercept": False,
    "random_state": 0,
    "copy_X": False,
}

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
model = Ridge(random_state=0).fit(x_train, y_train)
train_patched = timer() - start
print(f"Intel® extension for Scikit-learn time: {train_patched:.2f} s")

cur = conn.cursor()
query = f'''
	INSERT INTO ridge_regression( 
dataset_name,
datetime,
model_name,
total_time_taken
) VALUES('{dataset}','{dt_string}','optimized ridge_regression', {train_patched}) 
on conflict (model_name) do nothing;


update ridge_regression set datetime = '{dt_string}', total_time_taken = {train_patched} where model_name = 'optimized ridge_regression' returning *;
'''

cur.execute(query)
conn.commit()

y_predict = model.predict(x_test)
mse_metric_opt = metrics.mean_squared_error(y_test, y_predict)
print(f'Patched Scikit-learn MSE: {mse_metric_opt}')



from sklearnex import unpatch_sklearn
unpatch_sklearn()



from sklearn.linear_model import Ridge

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
model = Ridge(random_state=0).fit(x_train, y_train)
train_unpatched = timer() - start
print(f"Original Scikit-learn time: {train_unpatched:.2f} s")

query = f'''
	INSERT INTO ridge_regression( 
dataset_name,
datetime,
model_name,
total_time_taken
) VALUES('{dataset}','{dt_string}','unoptimized ridge_regression', {train_unpatched}) 
on conflict (model_name) do nothing;


update ridge_regression set datetime = '{dt_string}', total_time_taken = {train_unpatched} where model_name = 'unoptimized ridge_regression' returning *;
'''

cur.execute(query)
conn.commit()



y_predict = model.predict(x_test)
mse_metric_original = metrics.mean_squared_error(y_test, y_predict)
print(f'Original Scikit-learn MSE: {mse_metric_original}')



print(f'Get speedup in {(train_unpatched/train_patched):.1f} times.')