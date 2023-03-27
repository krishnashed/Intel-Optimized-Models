from timeit import default_timer as timer
from IPython.display import HTML
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import psycopg2
from datetime import datetime

DB_NAME = "aiml_optimizations"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_HOST = "192.168.122.172"
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

dataset = 'a9a'
x, y = fetch_openml(name=dataset, return_X_y=True)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



from sklearnex import patch_sklearn
patch_sklearn()



from sklearn.svm import SVC

params = {
    'C': 100.0,
    'kernel': 'rbf',
    'gamma': 'scale'
}

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
classifier = SVC(**params).fit(x_train, y_train)
train_patched = timer() - start
print(f"Intel® extension for Scikit-learn time: {train_patched:.2f} s")

cur = conn.cursor()
create_table_query = f'''
create table if not exists svc(
	dataset_name varchar(255),
	datetime varchar(255),
	model_name varchar(255) primary key,
	total_time_taken real
)
'''
cur.execute(create_table_query)

query = f'''
	INSERT INTO svc( 
dataset_name,
datetime,
model_name,
total_time_taken
) VALUES('{dataset}','{dt_string}','optimized svc', {train_patched}) 
on conflict (model_name) do nothing;


update svc set datetime = '{dt_string}', total_time_taken = {train_patched} where model_name = 'optimized svc' returning *;
'''

cur.execute(query)
conn.commit()


predicted = classifier.predict(x_test)
report = metrics.classification_report(y_test, predicted)
print(f"Classification report for Intel® extension for Scikit-learn SVC:\n{report}\n")




from sklearnex import unpatch_sklearn
unpatch_sklearn()




from sklearn.svm import SVC

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
classifier = SVC(**params).fit(x_train, y_train)
train_unpatched = timer() - start
print(f"Original Scikit-learn time: {train_unpatched:.2f} s")

query = f'''
	INSERT INTO svc( 
dataset_name,
datetime,
model_name,
total_time_taken
) VALUES('{dataset}','{dt_string}','unoptimized svc', {train_unpatched}) 
on conflict (model_name) do nothing;


update svc set datetime = '{dt_string}', total_time_taken = {train_unpatched} where model_name = 'unoptimized svc' returning *;
'''

cur.execute(query)
conn.commit()



predicted = classifier.predict(x_test)
report = metrics.classification_report(y_test, predicted)
print(f"Classification report for original Scikit-learn SVC:\n{report}\n")
print(f'Get speedup in {(train_unpatched/train_patched):.1f} times.')