import datetime
from typing import List
import os
from pymongo import MongoClient
import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import dump
from surprise import accuracy
import random
from bson.objectid import ObjectId
from models.user import UserCreate, UserInDB
from multiprocessing import Pool

dump_file = "data/knn_model.pkl"
load_from_file = True

def art_rating(df):
    counter = 0
    cuenta_total = list()
    cuenta_artista = list()
    artistas_usuario = list()
    for usr in df['user_id'].unique():
        usr_df = df[df['user_id'] == usr]
        n_art = df[df['user_id'] == usr].artname.nunique()
        n = len(usr_df)
        for art in usr_df.artname:
            n_artista = usr_df[usr_df.artname == art].artname.count()
            cuenta_artista.append(n_artista)
            cuenta_total.append(n)
            artistas_usuario.append(n_art)

        if counter % 100 == 0:
            print('User {} done'.format(counter))
        counter += 1
    return cuenta_artista, cuenta_total, artistas_usuario


def rebase_ratings(df):
    mean_ratings = dict()
    for user in df['user_id'].unique():
        df_user = df[df['user_id'] == user]
        mean_rating = df_user.re_based_rating.mean()
        mean_ratings[user] = mean_rating
        del df_user
    return mean_ratings


def rebase_ratings2(df):
    mean_ratings = dict()
    max_ratings = dict()
    min_ratings = dict()
    for user in df['user_id'].unique():
        df_user = df[df['user_id'] == user]
        mean_rating = df_user['rating'].mean()
        max_rating = df_user['rating'].max()
        min_rating = df_user['rating'].min()
        mean_ratings[user] = mean_rating
        max_ratings[user] = max_rating
        min_ratings[user] = min_rating
        del df_user
    return mean_ratings, max_ratings, min_ratings


def func_rating(list_param):
    escala = list_param.describe()
    std = escala[2]
    mean = escala[1]
    c1 = mean - 1 * std
    c2 = mean - 0.5 * std
    c3 = mean + 0.5 * std
    c4 = mean + 1 * std
    ratings = list()
    for fr in list_param:
        if fr < c1:
            r = 1
        elif fr < c2:
            r = 2
        elif fr < c3:
            r = 3
        elif fr < c4:
            r = 4
        else:
            r = 5
        ratings.append(r)
    return ratings


def load_model():
    # Later, to load the model from the file:
    return dump.load(dump_file)[1]


class RecomendationService:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["lastfm"]
        self.collection = self.db["tracks"]

    def load_data(self):
        if load_from_file:
            load_track_artista = pd.read_csv('data/df_track_artistas.csv')
            data = load_track_artista[
                ['#id', 'artname', 'tot_plays_usuario', 'plays_por_artista', 'n_artistas_usuario']]
            data = data.rename(columns={'#id': 'user_id'})
        else:
            # Retrieve all documents from the collection and store them in a list
            documents = list(self.collection.find({}, {"user_id": 1, "artname": 1, "trackname": 1}))

            # Convert the list of documents to a pandas DataFrame
            data = pd.DataFrame(documents, columns=["user_id", "artname", "trackname"])
            data = data.dropna().drop_duplicates()
            plays_por_artista, tot_plays_usuario, n_artistas_usuario = art_rating(data)
            data['tot_plays_usuario'] = tot_plays_usuario
            data['plays_por_artista'] = plays_por_artista
            data['n_artistas_usuario'] = n_artistas_usuario
            data = data[['user_id', 'artname', 'tot_plays_usuario', 'plays_por_artista', 'n_artistas_usuario']]

        data = data.drop_duplicates().dropna()
        data = data[(data.tot_plays_usuario >= 500) & (data.n_artistas_usuario >= 500)]
        data['peso_artista'] = data.plays_por_artista / data.n_artistas_usuario
        data['fuente_rating'] = (np.log10(data.peso_artista / data.tot_plays_usuario))
        data['re_based_rating'] = (data.fuente_rating - data.fuente_rating.min()) / (
                data.fuente_rating.max() - data.fuente_rating.min())
        mean_dict = rebase_ratings(data)
        df_means = pd.DataFrame(mean_dict.items())
        del mean_dict
        df_means.columns = ['user_id', 're_based_rating_mean']
        df_pre_rating = pd.merge(left=data, right=df_means, how='left', on='user_id')
        del df_means
        df_pre_rating['adj_base_rating'] = df_pre_rating.re_based_rating - df_pre_rating.re_based_rating_mean
        df_pre_rating['rating'] = func_rating(df_pre_rating.re_based_rating)

        data = df_pre_rating[['user_id', 'artname', 'rating']]
        del df_pre_rating

        mean_dict2, max_dict, min_dict = rebase_ratings2(data)

        df_means2 = pd.DataFrame(mean_dict2.items())
        df_means2.columns = ['user_id', 'rating_mean']

        del mean_dict2

        df_max = pd.DataFrame(max_dict.items())
        df_max.columns = ['user_id', 'rating_max']

        del max_dict

        df_min = pd.DataFrame(min_dict.items())
        df_min.columns = ['user_id', 'rating_min']

        del min_dict

        dict_df = df_means2.merge(df_max, on='user_id', how='left').merge(df_min, on='user_id', how='left')

        data = pd.merge(left=data, right=dict_df, how='left', on='user_id')

        del dict_df

        data['rating_adj'] = (data.rating - data.rating_mean)

        return data

    def build_model(self):
        # Para garantizar reproducibilidad en resultados
        seed = 10
        random.seed(seed)
        np.random.seed(seed)

        ratings = self.load_data()

        # escala ratings
        rmin = ratings.rating_adj.min()
        rmax = ratings.rating_adj.max()

        # carga reader
        reader = Reader(rating_scale=(rmin, rmax))

        # Se crea el dataset a partir del dataframe
        surprise_dataset = Dataset.load_from_df(ratings[['user_id', 'artname', 'rating_adj']], reader)

        # definimos train y test set
        train_set, test_set = train_test_split(surprise_dataset, test_size=.2)

        # se crea un modelo knnbasic
        sim_options = {'name': 'pearson_baseline',
                       'user_based': True  # calcule similitud usuario-usuario
                       }
        algo = KNNBasic(k=60,
                        min_k=2,
                        sim_options=sim_options)

        # Se le pasa la matriz de utilidad al algoritmo
        algo.fit(trainset=train_set)

        # Save the model to a file
        dump.dump(dump_file, algo=algo)
        return algo

    def make_prediction(self, user_id):
        # Check if model data exists, if not build it
        if not os.path.exists(dump_file):
            self.build_model()

        # Load the saved model from the file
        model = load_model()

        # Get a list of all tracks in the dataset
        pipeline = [
            {"$group": {"_id": "$artname"}},
            {"$project": {"artist": "$_id", "_id": 0}},
        ]
        artists = self.collection.aggregate(pipeline)

        # Make a prediction for each track in the dataset
        predictions = []
        for artist in artists:
            prediction = model.predict(user_id, artist["artist"])
            predictions.append(prediction)

        # Sort the predictions by estimated rating in descending order
        predictions.sort(key=lambda x: x.est, reverse=True)

        predictions_data = []
        for prediction in predictions:
            if prediction.details['was_impossible']:
                predictions_data.append((prediction.iid, "Not available", "Not available", True))
            else:
                predictions_data.append((prediction.iid, prediction.est, prediction.details['actual_k'], False))
        track_ratings = pd.DataFrame(predictions_data, columns=["artist", "estimation", "neighbors", "impossible"])

        return track_ratings

    def top_ten_tracks(self, user_id):

        return self.make_prediction(user_id).head(10)

    def load_new_tracks(self, user_id, new_tracks):
        for track in new_tracks:
            timestamp = datetime.datetime.utcnow().isoformat()
            track_doc = {
                "user_id": user_id,
                "timestamp": timestamp,
                "artid": track["artid"],
                "artname": track["artname"],
                "trackid": track["trackid"],
                "trackname": track["trackname"]
            }
            self.collection.insert_one(track_doc)

        self.build_model()

        return True
