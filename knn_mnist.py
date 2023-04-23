from timeit import default_timer as timer
from IPython.display import HTML
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from scipy.io import arff
import pandas as pd
# import psycopg2
from datetime import datetime

# DB_NAME = "aiml_optimizations"
# DB_USER = "postgres"
# DB_PASS = "postgres"
# DB_HOST = "192.168.122.172"
# DB_PORT = "5432"

# try:
# 	conn = psycopg2.connect(database=DB_NAME,
# 							user=DB_USER,
# 							password=DB_PASS,
# 							host=DB_HOST,
# 							port=DB_PORT)
# 	print("Database connected successfully")
# except:
# 	print("Database not connected successfully")

# data = arff.loadarff('./data/mnist_784.arff')
# df = pd.DataFrame(data[0])
# x = df.iloc[:,:-1]
# y = df.iloc[:, -1]
dataset = 'mnist_784'
x, y = fetch_openml(name=dataset, return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=72)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


from sklearnex import patch_sklearn
patch_sklearn()


from sklearn.neighbors import KNeighborsClassifier

params = {
    'n_neighbors': 40,
    'weights': 'distance',
    'n_jobs': -1
}

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
knn = KNeighborsClassifier(**params).fit(x_train, y_train)
predicted = knn.predict(x_test)
time_opt = timer() - start
f"Intel® extension for Scikit-learn time: {time_opt:.2f} s"

# cur = conn.cursor()
# create_table_query = f'''
# create table if not exists knn_mnist(
# 	dataset_name varchar(255),
# 	datetime varchar(255),
# 	model_name varchar(255) primary key,
# 	total_time_taken real
# )
# '''
# cur.execute(create_table_query)

# query = f'''
# 	INSERT INTO knn_mnist( 
# dataset_name,
# datetime,
# model_name,
# total_time_taken
# ) VALUES('{dataset}','{dt_string}','optimized knn_mnist', {time_opt}) 
# on conflict (model_name) do nothing;


# update knn_mnist set datetime = '{dt_string}', total_time_taken = {time_opt} where model_name = 'optimized knn_mnist' returning *;
# '''

# cur.execute(query)
# conn.commit()


report = metrics.classification_report(y_test, predicted)
print(f"Classification report for Intel® extension for Scikit-learn KNN:\n{report}\n")


from sklearnex import unpatch_sklearn
unpatch_sklearn()


from sklearn.neighbors import KNeighborsClassifier

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
knn = KNeighborsClassifier(**params).fit(x_train, y_train)
predicted = knn.predict(x_test)
time_original = timer() - start
f"Original Scikit-learn time: {time_original:.2f} s"


# query = f'''
# 	INSERT INTO knn_mnist( 
# dataset_name,
# datetime,
# model_name,
# total_time_taken
# ) VALUES('{dataset}','{dt_string}','unoptimized knn_mnist', {time_original}) 
# on conflict (model_name) do nothing;


# update knn_mnist set datetime = '{dt_string}', total_time_taken = {time_original} where model_name = 'unoptimized knn_mnist' returning *;
# '''

# cur.execute(query)
# conn.commit()

report = metrics.classification_report(y_test, predicted)
print(f"Classification report for original Scikit-learn KNN:\n{report}\n")


print(f'Get speedup in {(time_original/time_opt):.1f} times.')