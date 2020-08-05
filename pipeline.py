""" Loblaws Solutions Engineering take home problem """
import luigi
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    # TODO...
    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        f = self.output().open('w')
        df = pd.read_csv(self.tweet_file, encoding="ISO-8859-1")
        df = df[df.tweet_coord.notna()]
        df = df[df.tweet_coord.astype(str) != "[0.0, 0.0]"]
        df.to_csv(f, header=True, index=False)
        f.close()


class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')

    # TODO...
    def closest_point(self, point, points):
        """ Find closest point from a list of points. """
        return points[cdist([point], points).argmin()]

    def match_value(self, df, col1, x, col2):
        """ Match value x from col1 row to value in col2. """
        return df[df[col1] == x][col2].values[0]
        
    def requires(self):
        return CleanDataTask(self.tweet_file)
    
    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        f = self.output().open('w')
        inf = self.input()
        df = pd.read_csv(inf.open('r'))
        new = df['tweet_coord'].astype(str).str.split(',', n=1, expand=True)
        df['lan'] = new[0].astype(str).str.slice(start=1)
        df['long'] = new[1].astype(str).str.slice(stop=-1)
        cities_df = pd.read_csv(self.cities_file)
        df['point'] = [(x, y) for x, y in zip(df['lan'], df['long'])]
        cities_df['point'] = [(x, y) for x, y in zip(cities_df['latitude'], cities_df['longitude'])]
        df['closest'] = [self.closest_point(x, list(cities_df['point'])) for x in df['point']]
        df['city'] = [self.match_value(cities_df, 'point', x, 'name') for x in df['closest']]
        train_df = df [['city', 'airline_sentiment']]
        lenc = LabelEncoder()
        train_df['city_name'] = train_df['city']
        train_df['city'] = train_df[['city']].astype(str).apply(lenc.fit_transform)
        train_df['airline_sentiment'] = train_df[['airline_sentiment']].apply(lenc.fit_transform)
        train_df.to_csv(f, header=True, index=False)
        f.close()


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    # TODO...
    def requires(self):
        return TrainingDataTask(self.tweet_file)
    
    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        f = open(self.output().path, 'wb')
        train_df = pd.read_csv(self.input().open('r'))
        
        logreg = LogisticRegression(C=0.0001, class_weight=None, dual=False, fit_intercept=True,
                                    intercept_scaling=1, max_iter=10000, multi_class='ovr', n_jobs=3,
                                    penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                    verbose=0, warm_start=False)
        logreg.fit(train_df[['city']], train_df['airline_sentiment'])
        pickle.dump(logreg, f)
        f.close()


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    # TODO...
    def requires(self):
        return TrainModelTask(self.tweet_file), TrainingDataTask(self.tweet_file)
    
    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        f = self.output().open('w')
        md, td = self.input()
        inf = open(md.path, 'rb')
        df = pd.read_csv(td.open('r'))
        df = df.drop('airline_sentiment', axis=1)
        pickle_model = pickle.load(inf)
        yhat = pickle_model.predict_proba(df[['city']])
        f.write('city_name, neutral, negative, positive \n')
        for x, y in zip(df['city_name'], yhat):
            row = [x] + y.tolist()
            row = [str(x) for x in row]
            f.write('%s\n' % ','.join(row))
        f.close()


if __name__ == "__main__":
    luigi.run()
