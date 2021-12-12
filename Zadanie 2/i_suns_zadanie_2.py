from typing import List, Set
from pandas import DataFrame, Series

from sklearn import preprocessing

import pandas as pd


def extract_unique_genres(df: DataFrame) -> Set[str]:
    res = [i.replace("'", "")
               .replace("[", "")
               .replace("]", "")
               .replace(" ", "")
               .split(',')
           for i in df['artist_genres'].unique()]
    return set([g for sg in res for g in sg])


def str_to_list(s: str) -> List[str]:
    return s.replace("'", "").replace("[", "").replace("]", "").replace(" ", "").split(',')


def collect_release_data(df: DataFrame) -> Set[str]:
    return set(df['release_date'].unique())


def create_bool_mask(df: DataFrame, genres: Set[str], column: str):
    bool_dict = {}
    for i, item in enumerate(genres):
        bool_dict[item] = df[column].apply(lambda x: item in x)
    return pd.DataFrame(bool_dict)

def create_bool_mask_date(df: DataFrame, genres: Set[str], column: str):
    bool_dict = {}
    for i, item in enumerate(genres):
        bool_dict[item] = df[column].apply(lambda x: item == x)
    return pd.DataFrame(bool_dict)


def create_groups(df: DataFrame, group_by: str) -> List[DataFrame]:
    grouped = df.groupby(group_by)
    return [grouped.get_group(g) for g in grouped.groups]


def filter_only_relevant_song(g: List[DataFrame]) -> List[DataFrame]:
    return [
        sg.head(1)
        if len(sg) <= 1
        else sg.sort_values(by='popularity', ascending=False).head(1)
        for sg in g
    ]

# nacitanie dat a extrakcia unikatnych hodnot ktore sa pojdu maskovat
df_raw: DataFrame = pd.read_csv('sample_data/spotify_train.csv')
df_raw['release_date'] = df_raw['release_date'].apply(lambda d: d[0:4])
genres: Set[str] = extract_unique_genres(df_raw)
release_dates: Set[str] = collect_release_data(df_raw)
df_raw['artist_genres'] = df_raw['artist_genres'].apply(str_to_list)

# pridam index
df_raw['index'] = df_raw.index

# vytvorim bool masku top 50 udajov
genres_mask = create_bool_mask(df_raw, genres, 'artist_genres')
genres_mask['index'] = genres_mask.index
genres_index = genres_mask.index

release_mask = create_bool_mask_date(df_raw, release_dates, 'release_date')
release_mask['index'] = release_mask.index
releases_index = release_mask.index

df_raw['index'] = df_raw.index

genres_mask.replace({False: 0, True: 1}, inplace=True)
release_mask.replace({False: 0, True: 1}, inplace=True)

top_genres = genres_mask.sum(axis=0).drop(labels=['index']).nlargest(50)
top_years = release_mask.sum(axis=0).drop(labels=['index']).nlargest(50)

for genre in top_genres.keys():
  df_raw[genre] = df_raw['artist_genres'].apply(lambda x: 1 if genre in x else 0)
for year in top_years.keys():
  df_raw[year] = df_raw['release_date'].apply(lambda x: 1 if x == year else 0)

# odstanime nepotrebne udaje
df_raw.drop(['url', 'playlist_id', 'playlist_description', 'playlist_name', 'playlist_url', 'query'], axis=1, inplace=True)

# vycistim duplikaty songov
artist_groups = create_groups(df_raw, 'artist')
filtered_artists: List[DataFrame] = []
for artist in artist_groups:
    grouped_by_song: List[DataFrame] = create_groups(artist, 'name')
    filtered_artists.append(pd.concat(filter_only_relevant_song(grouped_by_song))) # songy ktore su uz odfiltrovane zlucim do jedneho DF
df_filtered = pd.concat(filtered_artists)

# hladam null hodnoty
df_filtered = df_filtered.dropna()

# hladame najviac zastupene
genres = top_genres.keys()
genre_percent = { genre: len(df_filtered.loc[df_filtered[genre]==True]) / len(df_filtered) for genre in list(genres)}
s_genres: Series = Series(genre_percent)
s_genres.sort_values(ascending=False, inplace=True)

s_genres.head(15).plot.bar(rot=0, figsize=(20,5))

# hladame najoblubenejsich interpretov
artist_leaderboard = df_filtered[['name', 'artist', 'artist_followers']].groupby(['artist']).first().reset_index().sort_values(by='artist_followers', ascending=False).head(15).plot.bar(x='artist', y='artist_followers', rot=0, figsize=(20,5))

# ako energicky posobila hudba pocas rokov
df_filtered[['release_date', 'energy']].sort_values(by='energy', ascending=False).groupby('release_date').mean().reset_index().plot.line(x='release_date', y='energy', rot=0, figsize=(20,5))

# ako hlasito posobila hudba pocas rokov
df_filtered[['release_date', 'loudness']].sort_values(by='loudness', ascending=False).groupby('release_date').mean().reset_index().plot.line(x='release_date', y='loudness', rot=0, figsize=(20,5))

# uprava dat pre regresor
X = df_filtered.drop(['release_date', 'name', 'artist', 'artist_genres', 'artist_id', 'id', 'explicit', 'index'], axis=1)
X.replace({False: 0, True: 1}, inplace=True)
y = X.pop('loudness')
head_X = list(X.columns.values)

min_max_scaler = preprocessing.MinMaxScaler()
# vyhadzujem kazdy riadok v ktorom akykolvek stlpec ma outlier hodnotu (https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame)

X = min_max_scaler.fit_transform(X)

# SVR regresor
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', gamma='auto')
regressor.fit(X, y.to_numpy())

# nacitanie, spracovanie a normalizacia testovacich dat
df_raw = pd.read_csv('sample_data/spotify_test.csv')
df_raw['release_date'] = df_raw['release_date'].apply(lambda d: d[0:4])
genres: Set[str] = extract_unique_genres(df_raw)
release_dates: Set[str] = collect_release_data(df_raw)
df_raw['artist_genres'] = df_raw['artist_genres'].apply(str_to_list)
to_be_dropped = df_raw.select_dtypes(['object', 'string'])
drpd = df_raw[['artist_followers']].copy()
to_be_dropped = to_be_dropped.join(drpd)
dropped = pd.concat([df_raw.pop(x) for x in to_be_dropped.columns], axis=1)
cols = df_raw.columns


df_raw = df_raw.join(dropped)
df_raw['index'] = df_raw.index
for genre in top_genres.keys():
  df_raw[genre] = None
  df_raw[genre] = df_raw['artist_genres'].apply(lambda x: 1 if genre in x else 0)
for year in top_years.keys():
  df_raw[year] = None
  df_raw[year] = df_raw['release_date'].apply(lambda x: 1 if x == year else 0)
df_raw.drop(['url', 'playlist_id', 'playlist_description', 'playlist_name', 'playlist_url', 'query'], axis=1, inplace=True)
Xt = df_raw.drop(['release_date', 'name', 'artist', 'artist_genres', 'artist_id', 'id', 'explicit','index'], axis=1)
Xt.replace({False: 0, True: 1}, inplace=True)
yt = Xt.pop('loudness')

Xt = min_max_scaler.transform(Xt)

y_pred = regressor.predict(Xt)

import sklearn.metrics as metrics
import numpy as np

mse = metrics.mean_squared_error(yt, y_pred)
r2 = metrics.r2_score(yt,y_pred)

print("Results of sklearn.metrics:")
print("MSE:", mse)
print("R-Squared:", r2)

import sklearn

sorted(sklearn.metrics.SCORERS.keys())

# grid search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
# Set the parameters by cross-validation
tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},    
]

scores = ["r2"]
print("# Tuning hyper-parameters for %s" % score)
print()

clf = GridSearchCV(SVR(), param_grid=tuned_parameters, cv=2)
clf.fit(X, y.to_numpy())

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_["mean_test_score"]
stds = clf.cv_results_["std_test_score"]
for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = yt, clf.predict(Xt)
print()

y.to_numpy()

# bagging
from sklearn.ensemble import BaggingRegressor

bagging_reg = BaggingRegressor(base_estimator=regressor, n_estimators=27, random_state=255)
bagging_reg.fit(X, y.to_numpy())

y_pred_bagging = bagging_reg.predict(Xt)

mse = metrics.mean_squared_error(yt, y_pred_bagging)
r2 = metrics.r2_score(yt,y_pred_bagging)

print("Results of sklearn.metrics:")
print("MSE:", mse)
print("R-Squared:", r2)

# boosting
from sklearn.ensemble import GradientBoostingRegressor

boosting_reg = GradientBoostingRegressor(random_state=255)
boosting_reg.fit(X, y.to_numpy())

y_pred_boosting = boosting_reg.predict(Xt)

mse = metrics.mean_squared_error(yt, y_pred_boosting)
r2 = metrics.r2_score(yt,y_pred_boosting)

print("Results of boosting:")
print("MSE:", mse)
print("R-Squared:", r2)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=0)
rf.fit(X, y)

estimator = rf.estimators_[42]
from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = head_X,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
Image(filename = 'tree.png')