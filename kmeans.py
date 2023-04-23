from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from IPython.display import HTML
import warnings
# import psycopg2
from datetime import datetime
warnings.filterwarnings('ignore')

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

dataset = 'spoken-arabic-digit'
x, y = fetch_openml(name=dataset, return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()


scaler_x.fit(x_train)
x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)


from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import KMeans

params = {
    "n_clusters": 128,    
    "random_state": 123,
    "copy_x": False,
}
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
model = KMeans(**params).fit(x_train, y_train)
train_patched = timer() - start
print(f"Intel® extension for Scikit-learn time: {train_patched:.2f} s")

# cur = conn.cursor()
# create_table_query = f'''
# create table if not exists kmeans(
# 	dataset_name varchar(255),
# 	datetime varchar(255),
# 	model_name varchar(255) primary key,
# 	total_time_taken real
# )
# '''
# cur.execute(create_table_query)

# query = f'''
# 	INSERT INTO kmeans( 
# dataset_name,
# datetime,
# model_name,
# total_time_taken
# ) VALUES('{dataset}','{dt_string}','optimized kmeans', {train_patched}) 
# on conflict (model_name) do nothing;


# update kmeans set datetime = '{dt_string}', total_time_taken = {train_patched} where model_name = 'optimized kmeans' returning *;
# '''

# cur.execute(query)
# conn.commit()

inertia_opt = model.inertia_
n_iter_opt = model.n_iter_
print(f"Intel® extension for Scikit-learn inertia: {inertia_opt}")
print(f"Intel® extension for Scikit-learn number of iterations: {n_iter_opt}")


from sklearnex import unpatch_sklearn
unpatch_sklearn()

from sklearn.cluster import KMeans

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
start = timer()
model = KMeans(**params).fit(x_train, y_train)
train_unpatched = timer() - start
print(f"Original Scikit-learn time: {train_unpatched:.2f} s")

# query = f'''
# 	INSERT INTO kmeans( 
# dataset_name,
# datetime,
# model_name,
# total_time_taken
# ) VALUES('{dataset}','{dt_string}','unoptimized kmeans', {train_unpatched}) 
# on conflict (model_name) do nothing;


# update kmeans set datetime = '{dt_string}', total_time_taken = {train_unpatched} where model_name = 'unoptimized kmeans' returning *;
# '''

# cur.execute(query)
# conn.commit()


inertia_original = model.inertia_
n_iter_original = model.n_iter_
print(f"Original Scikit-learn inertia: {inertia_original}")
print(f"Original Scikit-learn number of iterations: {n_iter_original}")


print(f'Get speedup in {(train_unpatched/train_patched):.1f} times')