import os.path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder

data_dir = 'data'
data_path = {
    'blood': '/'.join([data_dir, 'blood_transfusion.csv']),
    'bcw': '/'.join([data_dir, 'breast_cancer.csv']),
    'ccrf': '/'.join([data_dir, 'cervical_cancer.csv']),
    'cmc': '/'.join([data_dir, 'cmc.csv']),
    'diabetic': '/'.join([data_dir, 'diabetic_debrecen.csv']),
    'google_reviews': '/'.join([data_dir, 'google_review_ratings.csv']),
    'facebook': '/'.join([data_dir, 'facebook_live_sellers.csv']),
    'frogs': '/'.join([data_dir, 'frogs_anuran_calls.csv']),
    'mm': '/'.join([data_dir, 'mammographic_mass.csv']),
    'spam': '/'.join([data_dir, 'spambase.csv']),
    'ml100k': '/'.join([data_dir, 'ml100k_movie_ratings_genres.csv']),
    'ml1m': '/'.join([data_dir, 'ml1m_adapted_genre_ratings.csv'])
}


def switch(key: str):

    if key == 'blood':
        return load_blood_transfusion
    elif key == 'bcw':
        return load_breast_cancer_wisconsin
    elif key == 'ccrf':
        return load_cervical_cancer
    elif key == 'cmc':
        return load_contraceptive_method
    elif key == 'diabetic':
        return load_diabetic_retinopathy
    elif key == 'facebook':
        pass
    elif key == 'frogs':
        return load_frogs_anuran_calls
    elif key == 'google_reviews':
        return load_google_reviews
    elif key == 'mm':
        return load_mammographic_mass
    elif key == 'spam':
        return load_spambase
    elif key == 'ml100k':
        return load_movielens_100k
    elif key == 'ml1m':
        return load_movielens_1m
    else:
        raise ValueError('A valid dataset name should be used.')


# https://archive-beta.ics.uci.edu/ml/datasets/facebook+live+sellers+in+thailand
# samples: 7051
# no target
def load_facebook_live_sellers(path=None, adj_labels=False):

    if not os.path.exists(str(path)):
        path = data_path.get('facebook')

    df = pd.read_csv(path, sep=',', index_col=0, header=0, engine='c')

    # Convert string datetime to timestamp
    df[['status_published']] = df[['status_published']].applymap(
        lambda x: datetime.timestamp(datetime.strptime(x, "%m/%d/%Y %H:%M")))

    X = df.values[:, 1:-4].astype(np.float32)
    y = np.zeros(df.values.shape[0], dtype=int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29
# samples: 7195
# target: 10 (0-9)
def load_frogs_anuran_calls(path=None, adj_labels=False):

    if not os.path.exists(str(path)):
        path = data_path.get('frogs')

    df = pd.read_csv(path, sep=',', header=0, engine='c')

    labels = df.values[:, -2].astype(str)
    X = df.values[:, :22].astype(np.float32)
    y = LabelEncoder().fit_transform(labels)

    if adj_labels:
        y += 1

    return X, y


def load_movielens_100k(path=None, adj_labels=False):
    
    if path is None:
        path = data_path.get('ml100k')
    
    df = pd.read_csv(path, sep=',', names=list(range(20)))
    
    X = df.values.astype(np.float32)
    y = np.zeros(df.shape[0], dtype=int)
    
    if adj_labels:
        y += 1
    
    return X, y


def load_movielens_1m(path=None, adj_labels=False, missing='mean'):
    
    if not os.path.exists(str(path)):
        path = data_path.get('ml1m')
   
    df = pd.read_csv(path, sep=',', index_col=False, header=None)

    if missing == 'knn':
        X = KNNImputer(n_neighbors=5).fit_transform(df.values)
    else:
        X = SimpleImputer().fit_transform(df.values)

    y = np.zeros(df.shape[0], dtype=int)
    
    if adj_labels:
        y += 1
    
    return X, y
        
        
# https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
def load_blood_transfusion(path, adj_labels=True):

    if not os.path.exists(path):
        path = data_path.get('blood')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')
    df.dropna(how='all', inplace=True)

    X = df.values[:, :-1].astype(np.float32)
    y = df.values[:, -1].astype(int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
def load_breast_cancer_wisconsin(path=None, adj_labels=False):

    if not os.path.exists(path):
        path = data_path.get('bcw')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = df.values[:, 2:].astype(np.float32)
    y = df.values[:, 1].astype(int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29
def load_cervical_cancer(path=None, adj_labels=False):

    if not os.path.exists(str(path)):
        path = data_path.get('ccrf')

    df = pd.read_csv(path, sep=',', index_col=False, engine='c')

    X = df.values[:, :-4].astype(np.float32)
    y = df.values[:, -1].astype(int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
def load_contraceptive_method(path=None, adj_labels=False):

    if (path is None) or (not os.path.exists(path)):
        path = data_path.get('cmc')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = df.values[:, :-1].astype(np.float32)
    y = df.values[:, -1].astype(int)

    if not adj_labels:
        y -= 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
def load_diabetic_retinopathy(path=None, adj_labels=False):

    if not os.path.exists(str(path)):
        path = data_path.get('diabetic')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = df.values[:, :-1].astype(np.float64)
    y = df.values[:, -1].astype(int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Tarvel+Review+Ratings
def load_google_reviews(path=None, adj_labels=False):

    if (path is None) or (not os.path.exists(path)):
        path = data_path.get('google_reviews')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = KNNImputer(missing_values=0.).fit_transform(df.values.astype(np.float32))
    y = np.zeros(shape=(X.shape[0],))

    if adj_labels:
        y += 1

    return X, y


# http://archive.ics.uci.edu/ml/datasets/mammographic+mass
def load_mammographic_mass(path=None, adj_labels=False):

    if not os.path.exists(str(path)):
        path = data_path.get('mm')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = df.values[:, :-1].astype(np.float32)
    y = df.values[:, -1].astype(int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/spambase
def load_spambase(path=None, adj_labels=False):

    if (path is None) or (not os.path.exists(path)):
        path = data_path.get('spam')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = SimpleImputer().fit_transform(df.values[:, :-1].astype(np.float64))
    y = df.values[:, -1].astype(int)

    if adj_labels:
        y += 1

    return X, y

