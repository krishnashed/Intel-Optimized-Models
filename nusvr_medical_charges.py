from timeit import default_timer as timer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import psycopg2
from datetime import datetime
from IPython.display import HTML
import warnings
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

dataset = 'medical_charges_nominal'
x, y = fetch_openml(name=dataset, return_X_y=True)

cat_columns = x.select_dtypes(['category']).columns
x[cat_columns] = x[cat_columns].apply(lambda x: x.cat.codes)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.svm import NuSVR

params = {
    'nu': 0.4,
    'C': y_train.mean(),
    'degree': 2,
    'kernel': 'poly',
}

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
nusvr = NuSVR(**params).fit(x_train, y_train)
train_patched = timer() - start
print(f"Intel® extension for Scikit-learn time: {train_patched:.2f} s")

cur = conn.cursor()
query = f'''
	INSERT INTO nusvr( 
dataset_name,
datetime,
model_name,
total_time_taken
) VALUES('{dataset}','{dt_string}','optimized nusvr', {train_patched}) 
on conflict (model_name) do nothing;


update nusvr set datetime = '{dt_string}', total_time_taken = {train_patched} where model_name = 'optimized nusvr' returning *;
'''

cur.execute(query)
conn.commit()

score_opt = nusvr.score(x_test, y_test)
print(f'Intel® extension for Scikit-learn R2 score: {score_opt}')


from sklearnex import unpatch_sklearn
unpatch_sklearn()

from sklearn.svm import NuSVR

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
nusvr = NuSVR(**params).fit(x_train, y_train)
train_unpatched = timer() - start
print(f"Original Scikit-learn time: {train_unpatched:.2f} s")

query = f'''
	INSERT INTO nusvr( 
dataset_name,
datetime,
model_name,
total_time_taken
) VALUES('{dataset}','{dt_string}','unoptimized nusvr', {train_unpatched}) 
on conflict (model_name) do nothing;


update nusvr set datetime = '{dt_string}', total_time_taken = {train_unpatched} where model_name = 'unoptimized nusvr' returning *;
'''

cur.execute(query)
conn.commit()


score_original = nusvr.score(x_test, y_test)
print(f'Original Scikit-learn R2 score: {score_original}')

print(f'Get speedup in {(train_unpatched/train_patched):.1f} times.')