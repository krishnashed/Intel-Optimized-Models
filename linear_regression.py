from timeit import default_timer as timer
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import requests
import warnings
import psycopg2
from datetime import datetime
from IPython.display import HTML
warnings.filterwarnings('ignore')

DB_NAME = "aiml_optimizations"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_HOST = "192.168.122.173"
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

dataset_dir = 'data'
dataset_name = 'year_prediction_msd'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'

os.makedirs(dataset_dir, exist_ok=True)
local_url = os.path.join(dataset_dir, os.path.basename(url))

if not os.path.isfile(local_url):
    response = requests.get(url, stream=True)
    with open(local_url, 'wb+') as file:
        for data in response.iter_content(8192):
            file.write(data)
    
year = pd.read_csv(local_url, header=None)
x = year.iloc[:, 1:].to_numpy(dtype=np.float32)
y = year.iloc[:, 0].to_numpy(dtype=np.float32)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler_x = MinMaxScaler()
scaler_y = StandardScaler()


scaler_x.fit(x_train)
x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)

scaler_y.fit(y_train.reshape(-1, 1))
y_train = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()


from sklearnex import patch_sklearn
patch_sklearn()


from sklearn.linear_model import LinearRegression

params = {
    "n_jobs": -1,
    "copy_X": False
}

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
model = LinearRegression(**params).fit(x_train, y_train)
train_patched = timer() - start
print(f"Intel® extension for Scikit-learn time: {train_patched:.2f} s")

cur = conn.cursor()
create_table_query = f'''
create table if not exists linear_regression(
	dataset_name varchar(255),
	datetime varchar(255),
	model_name varchar(255) primary key,
	total_time_taken real
)
'''
cur.execute(create_table_query)


query = f'''
	INSERT INTO linear_regression( 
dataset_name,
datetime,
model_name,
total_time_taken
) VALUES('{dataset_name}','{dt_string}','optimized linear_regression', {train_patched}) 
on conflict (model_name) do nothing;


update linear_regression set datetime = '{dt_string}', total_time_taken = {train_patched} where model_name = 'optimized linear_regression' returning *;
'''

cur.execute(query)
conn.commit()

y_predict = model.predict(x_test)
mse_metric_opt = metrics.mean_squared_error(y_test, y_predict)
print(f'Patched Scikit-learn MSE: {mse_metric_opt}')


from sklearnex import unpatch_sklearn
unpatch_sklearn()



from sklearn.linear_model import LinearRegression

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
model = LinearRegression(**params).fit(x_train, y_train)
train_unpatched = timer() - start
print(f"Original Scikit-learn time: {train_unpatched:.2f} s")

query = f'''
	INSERT INTO linear_regression( 
dataset_name,
datetime,
model_name,
total_time_taken
) VALUES('{dataset_name}','{dt_string}','unoptimized linear_regression', {train_unpatched}) 
on conflict (model_name) do nothing;


update linear_regression set datetime = '{dt_string}', total_time_taken = {train_unpatched} where model_name = 'unoptimized linear_regression' returning *;
'''

cur.execute(query)
conn.commit()


y_predict = model.predict(x_test)
mse_metric_original = metrics.mean_squared_error(y_test, y_predict)
print(f'Original Scikit-learn MSE: {mse_metric_original}')

print(f'Get speedup in {(train_unpatched/train_patched):.1f} times')
