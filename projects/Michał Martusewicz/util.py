import os
import subprocess
from itertools import chain
from pathlib import Path

import pandas as pd
import sklearn
from IPython.display import display
from graphviz import Source
import numpy as np
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import matplotlib.pyplot as plt


def read_files(size="small"):
    if size == "small":
        ratings = pd.read_csv("data/ml-latest-small/ratings.csv", sep=",", header=0, engine='python')
        movies = pd.read_csv("data/ml-latest-small/movies.csv", sep=",", header=0, engine='python')

        # Getting all the available movie genres
        genres = set(chain(*[x.split("|") for x in movies["genres"]]))
        genres.remove("(no genres listed)")

        # "genres vector"
        for v in genres:
            movies[v] = movies['genres'].str.contains(v)

        # movies mean raiting
        movies = movies.merge(ratings.groupby('movieId')['rating'].agg([pd.np.mean]), how='left', on='movieId')
        return ratings, movies, genres


def export_tree(tree, feature_names, export=False, show=False, filename="tree.pdf"):
    if show:
        tmp = sklearn.tree.export_graphviz(tree, feature_names=feature_names)
        display(Source(tmp))
    if export:
        Path("output/").mkdir(parents=True, exist_ok=True)
        with open("output/tmp1.dot", "w") as f:
            sklearn.tree.export_graphviz(tree, out_file=f, feature_names=feature_names)
            cmd = ['dot', '-Tpdf', 'output/tmp1.dot', '-o', "output/" + filename]
            subprocess.call(cmd)
            os.remove("output/tmp1.dot")


def test_classifier(classifier, clf_data, genres, export=False, show_tree=False, verbose=True, plot=True, axs=None,
                    clf_name=None, cv_n_jobs=4, **kwargs):
    plot_data = {}
    for i, genre in enumerate(tqdm(genres)):
        clf = classifier(**kwargs)

        X = clf_data.drop(genre, axis=1)
        y = clf_data[genre]
        clf = clf.fit(X, y)
        score_val = np.sum(cross_val_score(clf, X, y, cv=10, verbose=verbose, n_jobs=cv_n_jobs)) / 10
        if verbose:
            print(f"""{genre}:
            Quantity: {len(y[clf_data[genre] == True])} / {len(y[clf_data[genre] != True])}
            True positives: {clf.score(X[clf_data[genre] == True], y[clf_data[genre] == True])}
            Accuracy: {clf.score(X, y)}
            Cross-val: {score_val}"""
                  )
        plot_data[genre] = [len(y[clf_data[genre] == True]), score_val]
        if export or show_tree:
            export_tree(clf, X.columns, show=show_tree, export=export, filename=f"custom_{genre}.pdf")
    if plot:
        plot_results(plot_data, axs, clf_name=classifier.__name__ if clf_name is None else clf_name)
    return plot_data


def plot_results(plot_data, axs=None, clf_name=None, color=None):
    if axs is None:
        fig, ax = plt.subplots(figsize=(15, 15))
    else:
        fig, ax = axs
    v1, v2 = list(zip(*plot_data.values()))
    # v2 = np.log(v2)
    ax.scatter(v1, v2, label=clf_name, c=color)

    for i, txt in enumerate(plot_data.keys()):
        ax.annotate(txt, (v1[i], v2[i]))