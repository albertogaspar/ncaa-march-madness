import pandas as pd
import numpy as np
import logging
from statsmodels.distributions.empirical_distribution import ECDF
from src.evaluation import Metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from src import LoadDatasets

logger = logging.getLogger('root')
ABSOLUTE_PATH = LoadDatasets.get_absolute_path()

class KNN(object):

    def __init__(self, n_neighbors=70, target_class='MOV', metric='euclidean'):
        """
        :param n_neighbors: n_neighbors for KNN
        :param target_class: 'MOV' or 'win_prob'
        :param metric: 'euclidean', 'manhattan', 'jaccard'
        """
        self.n_neighbors=n_neighbors
        self.target_class = target_class
        self.metric = metric

    def predict(self,target, normalize=True):
        """
        :param target: dataframe with the matchup to predict; the resulting probabilities are written here
        :param normalize: normalize the input or not before fedding it to the models
        :return: target dataframes with probabilities.
        """
        self.normalize=normalize
        # normalize data:  If we don't normalize those dimensions,
        # one or the other is likely to be far more important in determining the nearest neighbor
        self.target = target
        self._load_files()

        years = self.target['Season'].unique()
        for year in years:
            logger.info("Generating prediction for year {0}".format(year))

            X_train_rs = self.regular_season_train[self.regular_season_train['season']<=year]
            X_train_t = self.tournament_train[self.tournament_train['season']<year]

            X_test = self.to_predict[self.to_predict['season']==year]

            X_train_rs = X_train_rs.dropna()
            X_train_t = X_train_t.dropna()
            if X_test.dropna().shape[0] == X_test.shape[0]:
                col_to_drop = []
            else: #drop all columns that has missing values
                col_to_drop = X_test.columns[pd.isnull(X_test).any()].tolist()

            Y_train_rs = X_train_rs['Score_diff'].tolist()
            Y_train_t = X_train_t['Score_diff'].tolist()

            #Remove Categorical value (useless, name ofconference team are also coded into ids) + value to predict
            X_train_rs.drop(
                ['Unnamed: 0', 'Score_diff', 'team1_score', 'team2_score', 'team1', 'team2', 'team1win', 'team1_Team',
                 'team2_Team', 'team1_Conference', 'team2_Conference']+col_to_drop, inplace=True, axis=1)
            X_train_t.drop(
                ['Unnamed: 0', 'Score_diff', 'team1_score', 'team2_score', 'team1', 'team2', 'team1win', 'team1_Team',
                 'team2_Team', 'team1_Conference', 'team2_Conference']+col_to_drop, inplace=True, axis=1)
            X_test.drop(
                ['Unnamed: 0', 'team1_score', 'team2_score', 'team1', 'team2', 'team1_Team',
                 'team2_Team', 'team1_Conference', 'team2_Conference']+col_to_drop, inplace=True, axis=1)

            X_train = pd.concat([X_train_rs,X_train_t])
            Y_train = Y_train_rs + Y_train_t

            if self.normalize:
                train = X_train.drop('season', axis=1).values
                test = X_test.drop('season', axis=1).values
                # Scaling -> for each column j : (x - column_min)/(column_max - column_min)
                min_max_scaler = preprocessing.MinMaxScaler()
                train_scaled = min_max_scaler.fit_transform(train)
                test_scaled = min_max_scaler.fit_transform(test)
                X_train = pd.DataFrame(train_scaled, columns=X_train_rs.columns[1:])
                X_test = pd.DataFrame(test_scaled, columns=X_test.columns[1:])

            if self.target_class == 'MOV':
                self._predict_MOV(X_train, Y_train, X_test, year)
            elif self.target_class == 'win_prob':
                self._predict_win_prob(X_train, Y_train, X_test, year)
            else:
                logger.info("Invalid argument for target_class")

        #self.target = self.target.drop(['Season','teamA','teamB'],axis=1)
        #self._evaluate_all(self.target)
        return self.target

    def _predict_MOV(self, X_train, Y_train, X_test, year):
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric)
        knn.fit(X_train, Y_train)
        knn_pred = knn.predict(X_test)

        ecdf = ECDF(Y_train)
        probabilities = ecdf(knn_pred)

        for i in self.target[self.target['Season'] == year].index:
            prob = probabilities[i - self.target[self.target['Season'] == year].index[0]]
            if prob != 0.0:
                self.target.set_value(i, 'pred', prob)
            else:
                self.target.set_value(i, 'pred', 0.05)
                # logger.info("{0}-->{1}".format(i,self.target.loc[i]['pred']))
        self._evaluate(year, self.target[self.target['Season'] == year])

    def _predict_win_prob(self, X_train, Y_train, X_test, year):
        if self.metric == 'mahalanobis':
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric, metric_params={'V': np.cov(X_train)})
        else:
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric)
        Y_train = np.array(Y_train)
        Y_train[Y_train > 0] = 1
        Y_train[Y_train <= 0] = 0
        knn.fit(X_train, Y_train)
        knn_prob = knn.predict_proba(X_test)
        for i in self.target[self.target['Season'] == year].index:
            prob = knn_prob[i - self.target[self.target['Season'] == year].index[0]][1]  # -> index 1 is probability of first team winning
            if (prob != 0.0) & (prob != 1.0):
                self.target.set_value(i, 'pred', prob)
            elif (prob == 0.0):
                self.target.set_value(i, 'pred', 0.05)
            else:
                self.target.set_value(i, 'pred', 0.95)
        #self._evaluate(year, self.target[self.target['Season'] == year])

    def _load_files(self):
        self.regular_season_train = pd.read_csv(ABSOLUTE_PATH+"data/RegularSeason_TeamStatsBySeason.csv", encoding='latin-1')
        self.tournament_train = pd.read_csv(ABSOLUTE_PATH+"data/Tournament_TeamStatsBySeason.csv", encoding='latin-1')
        self.to_predict = pd.read_csv(ABSOLUTE_PATH+"data/TargetStatsBySeason.csv", encoding='latin-1')

    def _evaluate(self,year, prediction):
        true = pd.read_csv(ABSOLUTE_PATH+"data/evaluation/{}_target_no_play_in.csv".format(year))
        Metrics.logloss(true,prediction,year=year)
        Metrics.accuracy(true, prediction, year=year)

    def _evaluate_all(self, prediction):
        target = pd.read_csv(ABSOLUTE_PATH+"data/evaluation/ALL_target_no_play_in.csv")
        Metrics.logloss(target,prediction)
        Metrics.accuracy(target, prediction)

