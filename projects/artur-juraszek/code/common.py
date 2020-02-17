import functools

from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.feature_extraction.text
import sklearn.metrics

stemmer = SnowballStemmer('english')


def stemming_preprocessor(data):
    return stemmer.stem(data)


@functools.lru_cache(maxsize=3)
def load_dataset(sampling_method, vectorization, preprocessing):
    filename = f'vectorized/{sampling_method}-{vectorization}-{preprocessing or "none"}'
    filename_train = f'{filename}-TRAIN.npz'
    filename_test = f'{filename}-TEST.npz'
    loaded_from_disk = False
    
    try:
        train_as_vector = scipy.sparse.load_npz(filename_train)
        test_as_vector = scipy.sparse.load_npz(filename_test)
        loaded_from_disk = True
    except:
        print('have to generate new vectorizations')
        
    
    vectorizers = {
        'count': {
            None: sklearn.feature_extraction.text.CountVectorizer(),
            'stop_words': sklearn.feature_extraction.text.CountVectorizer(stop_words='english'),
            'stem': sklearn.feature_extraction.text.CountVectorizer(preprocessor=stemming_preprocessor)
        },
        'binary': {
            None: sklearn.feature_extraction.text.CountVectorizer(binary=True),
            'stop_words': sklearn.feature_extraction.text.CountVectorizer(binary=True, stop_words='english'),
        },
        'tf_idf': {
            None: sklearn.feature_extraction.text.TfidfVectorizer(),
            'stop_words': sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english'),
            'stem': sklearn.feature_extraction.text.TfidfVectorizer(preprocessor=stemming_preprocessor)
        }
    }
    
    vectorizer = vectorizers[vectorization][preprocessing]
    
    filenames = {
        'random_downsampling': ('downsampled_train.csv', 'test.csv'),
        'full': ('full_train.csv', 'test.csv'),
        'oversampled': ('oversampled_train.csv', 'test.csv'),
    }
    
    train_name, test_name = filenames[sampling_method]
    
    train = pd.read_csv(train_name, header=0, index_col=0)
    test = pd.read_csv(test_name, header=0, index_col=0)
    
    # print(train['reviewText'].values)
    # print(train['reviewText'].index)
    # print(train.dtypes)

    if not loaded_from_disk:
        train_as_vector = vectorizer.fit_transform(train['reviewText'].values)
        test_as_vector = vectorizer.transform(test['reviewText'].values)
        print('saving matrices to disk')
        scipy.sparse.save_npz(filename_train, train_as_vector)
        scipy.sparse.save_npz(filename_test, test_as_vector)
    
    return train_as_vector, train['overall'].values, test_as_vector, test['overall'].values


# load_dataset('random_downsampling', 'count', None)


def get_score(classifier, test, test_targets):
    return sklearn.metrics.balanced_accuracy_score(test_targets, classifier.predict(test))


def display_confusion_matrices(classifier, test, test_targets):
    print(sklearn.metrics.confusion_matrix(
        test_targets, classifier.predict(test), normalize='true'))
    sklearn.metrics.plot_confusion_matrix(
        classifier, test, test_targets, normalize='true', cmap='Purples')
    

def display_score(classifier, test, test_targets):
    print(f'SCORE: {get_score(classifier, test, test_targets)}')


def display_classifier_performance(classifier, test, test_targets):
    display_score(classifier, test, test_targets)
    display_confusion_matrices(classifier, test, test_targets)


def order_aware_error(estimator, test_X, test_Y):
    #predictions = estimator.predict(test_X)
    #error_count = sum(predictions != test_Y)
    #return sum(abs(predictions - test_Y)) / error_count
    klasses = {}
    for klass in range(1, 5+1):
        klass_indices = (test_Y == klass)
        klass_predictions = estimator.predict(test_X[klass_indices])
        klass_error_count = sum(klass_predictions != test_Y[klass_indices])
        klasses[f'order_aware_error_{klass}'] = \
            sum(abs(klass_predictions - test_Y[klass_indices])) / klass_error_count
    klasses['order_aware_error_avg'] = sum(klasses.values()) / 5
    return klasses
        

def perf_row(
    classifier, test_as_vec, test_targets, classifier_type, sampling,
    representation, preprocessing, **classifier_specific):
    return {
        'classifier_type': classifier_type,
        'sampling': sampling,
        'representation': representation,
        'preprocessing': preprocessing,
        **classifier_specific,
        'real_world_acc': classifier.score(test_as_vec, test_targets),
        'score': get_score(classifier, test_as_vec, test_targets),
        **order_aware_error(classifier, test_as_vec, test_targets),
    }