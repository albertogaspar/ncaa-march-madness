import pandas as pd
import logging
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from statsmodels.distributions.empirical_distribution import ECDF
from src.evaluation import Metrics
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile
import numpy as np
from matplotlib import pyplot as plt
from random import random
from src import LoadDatasets

logger = logging.getLogger('root')
ABSOLUTE_PATH = LoadDatasets.get_absolute_path()

class NCAALogisticRegression(object):

    def __init__(self, shrinkage_technique='None', shrink_param=1.0, normalize = True, threashold=None):
        """
        :param shrinkage_technique: 'lasso' or 'ridge' or None
        :param shrink_param: value for lambda (regularization term)
        :param normalize: normalize the input or not before fedding it to the models
        """
        self.shrinkage_technique = shrinkage_technique
        self.shrink = shrink_param
        self.normalize = normalize
        self.threashold = threashold
        self.COLOR = {2013:'r', 2014:'g', 2015:'b', 2016:'orange'}
        return

    def predict_df(self,train,to_predict,target, features_cv=False, plot=False, max_features=None):
        """
        :param train: dataframe used for training, if None use default datsets
        :param to_predict: dataframe with the matchup to predict, must have same set of features as train dataframe
        :param target: dataframe with the matchup to predict; the resulting probabilities are written here
        :param features_cv: build different models aggregating features one at a time
        :param plot: plot the graph representing  no. features vs LogLoss
        :param max_features: run the model with a fixed number of features (the most powerful are taken)
        :return: target dataframes with probabilities.
        """
        self.target=target
        self.features_cv=features_cv
        self.max_features = max_features
        self.plot = plot
        self.train = train
        self.to_predict = to_predict

        years = self.to_predict['season'].unique()
        for year in years:
            logger.info("Generating prediction for year {0}".format(year))
            if train is None:
                self._load_files()
                self._prepare_default_data(year)
            else:
                self._prepare_data(year)

            if self.normalize:
                self._normalize()

            self.get_features_by_score()

            if max_features is not None:
                self.X_train = self.X_train[self.allFeatureByScore[:self.max_features]]
                self.X_test = self.X_test[self.allFeatureByScore[:self.max_features]]

            self._get_model()
            self.lr.fit(self.X_train, self.Y_train)
            self.pred = self.lr.predict_proba(self.X_test)
            pred = self.pred

            scores=[]
            if self.features_cv == True:
                self._feature_cv(scores=scores, year=year)

                if self.plot == True:
                    self._plot(self.allFeatureByScore, scores, year)

            for i in self.target[self.target['Season'] == year].index:
                prob = pred[i - self.target[self.target['Season'] == year].index[0]][1]
                if prob != 0.0:
                    self.target.set_value(i, 'pred', prob)
                else:
                    self.target.set_value(i, 'pred', 0.05)
            self._evaluate(year, self.target[self.target['Season'] == year])

        #self.target = self.target.drop(['Season','teamA','teamB'],axis=1)

        self._evaluate_all(self.target)
        return self.target

    def _load_files(self):
        self.regular_season_train = pd.read_csv(ABSOLUTE_PATH+"data/RegularSeason_TeamStatsBySeason.csv", encoding='latin-1')
        self.tournament_train = pd.read_csv(ABSOLUTE_PATH+"data/Tournament_TeamStatsBySeason.csv", encoding='latin-1')
        self.to_predict = pd.read_csv(ABSOLUTE_PATH+"data/TargetStatsBySeason.csv", encoding='latin-1')

    def add_massey_ranking_only(self,df):
        m = df[['team1_MAS_rank', 'team2_MAS_rank']]
        df.drop(df.columns[-264:], inplace=True, axis=1)
        df = pd.concat([df,m])
        return df

    def _evaluate(self,year, prediction):
        true = pd.read_csv(ABSOLUTE_PATH+"data/evaluation/{}_target_no_play_in.csv".format(year))
        self.score = Metrics.logloss(true,prediction,year=year)
        Metrics.accuracy(true, prediction, year=year)

    def _evaluate_all(self, prediction):
        target = pd.read_csv(ABSOLUTE_PATH+"data/evaluation/ALL_target_no_play_in.csv")
        Metrics.logloss(target,prediction)
        Metrics.accuracy(target, prediction)

    def _plot(self, allFeatureByScore, scores, year):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        #Labels
        ax.set_xlabel("Number of features")
        ax.set_ylabel("LogLoss")
        # Define major and minor ticks for x and y axis
        major_ticks_y = np.arange(0.40, 0.80, 0.05)
        major_ticks_x = np.arange(1, 35, 5)
        minor_ticks_y = np.arange(0.40, 0.80, 0.01)
        minor_ticks_x = np.arange(1, 35, 1)
        ax.set_ylim(0.40, 0.80)
        ax.set_xlim(1, 35, 5)
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        # Major axis more relevant then minor axis
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        # Plot lines
        ax.plot(range(1, len(allFeatureByScore)), scores, self.COLOR[year])
        # One plot for each year
        plt.savefig('LogReg_FeaturesCV_{0}.png'.format(year))
        plt.close()

    def _feature_cv(self, year, scores):
        for max in range(1, len(self.allFeatureByScore)):
            print(max)
            print(self.allFeatureByScore)
            print([self.allFeatureByScore[:max]])
            self._get_model()
            self.lr.fit(self.X_train[self.allFeatureByScore[:max]], self.Y_train)
            probabilities = self.lr.predict_proba(self.X_test[self.allFeatureByScore[:max]])

            for i in self.target[self.target['Season'] == year].index:
                prob = probabilities[i - self.target[self.target['Season'] == year].index[0]][1]
                if (prob > 0.0) & (prob < 1.0):
                    # print(prob)
                    self.target.set_value(i, 'pred', prob)
                elif (prob == 1.0):
                    # print(prob)
                    self.target.set_value(i, 'pred', 0.95)
                elif (prob == 0.0):
                    # print(prob)
                    self.target.set_value(i, 'pred', 0.05)
                elif (prob < 0.0) | (prob > 1.0):
                    print("ERROR!!!" + str(prob))
            logger.info("features = {0}, length ={1} ".format(self.allFeatureByScore[:max], max))
            self._evaluate(year, self.target[self.target['Season'] == year])
            scores.extend([self.score])
        return scores

    def _normalize(self):
        train = self.X_train.drop('season', axis=1).values
        test = self.X_test.drop('season', axis=1).values
        min_max_scaler = preprocessing.MinMaxScaler()
        train_scaled = min_max_scaler.fit_transform(train)
        test_scaled = min_max_scaler.fit_transform(test)
        self.X_train = pd.DataFrame(train_scaled, columns=self.X_train.columns[1:])
        self.X_test = pd.DataFrame(test_scaled, columns=self.X_test.columns[1:])

    def get_features_by_score(self):
        selector = SelectPercentile()
        features = selector.fit_transform(X=self.X_train, y=self.Y_train)
        feature_names = self.X_train.columns.tolist()
        self.allFeatureByScore = [feature_names[i] for i in np.argsort(selector.scores_)[::-1]]

    def _prepare_data(self, year):
        self.X_train = self.train[self.train['season'] <= year]
        self.X_test = self.to_predict[self.to_predict['season'] == year]

        if self.threashold is not None:
            self.X_train = self.X_train[self.X_train['season']>=self.threashold]

        # Clean
        col_to_drop = []
        if self.X_test.dropna().shape[0] != self.X_test.shape[0]:
            col_to_drop = self.X_test.columns[pd.isnull(self.X_test).any()].tolist()

        self.Y_train = self.X_train['team1win'].tolist()
        self.X_train.drop(['Score_diff', 'team1win',] + col_to_drop, inplace=True, axis=1)
        self.X_test.drop(col_to_drop, inplace=True, axis=1)

    def _prepare_default_data(self,year):
        X_train_rs = self.regular_season_train[self.regular_season_train['season'] <= year].dropna()
        X_train_t = self.tournament_train[self.tournament_train['season'] < year].dropna()
        self.X_test = self.to_predict[self.to_predict['season'] == year]

        if self.threashold is not None:
            X_train_rs = X_train_rs[X_train_rs['season']>=self.threashold]
            X_train_t = X_train_t[X_train_t['season']>=self.threashold]

        col_to_drop = []
        if self.X_test.dropna().shape[0] != self.X_test.shape[0]:
            col_to_drop = self.X_test.columns[pd.isnull(self.X_test).any()].tolist()

        Y_train_rs = X_train_rs['team1win'].tolist()
        Y_train_t = X_train_t['team1win'].tolist()
        self.Y_train = Y_train_rs + Y_train_t

        # Remove Categorical value (useless, name ofconference team are also coded into ids) + value to predict
        # Remove 'team1_score', 'team2_score' -> worst!
        X_train_rs.drop(
            ['Unnamed: 0', 'Score_diff', 'team1_score', 'team2_score', 'team1', 'team2', 'team1win', 'team1_Team',
             'team2_Team', 'team1_Conference', 'team2_Conference'] + col_to_drop, inplace=True, axis=1)
        X_train_t.drop(
            ['Unnamed: 0', 'Score_diff', 'team1_score', 'team2_score', 'team1', 'team2', 'team1win', 'team1_Team',
             'team2_Team', 'team1_Conference', 'team2_Conference'] + col_to_drop, inplace=True, axis=1)
        self.X_test.drop(
            ['Unnamed: 0', 'team1', 'team2', 'team1_Team', 'team1_score', 'team2_score',
             'team2_Team', 'team1_Conference', 'team2_Conference'] + col_to_drop, inplace=True, axis=1)
        self.X_train = pd.concat([X_train_rs, X_train_t])

    def _get_model(self):
        if self.shrink is not None:
            if self.shrinkage_technique == 'lasso':
                self.lr = LogisticRegression(C=self.shrink, penalty='l1')
            elif self.shrinkage_technique =='ridge':
                self.lr = LogisticRegression(C=self.shrink, penalty='l2')
            else:
                self.lr = LogisticRegression()
        else:
            self.lr = LogisticRegression()
            logger.warning("shrinkage_technique was set to {0} but no shrinkage param has been defined. "+
                           "No regularization will be performed".format(self.shrinkage_technique))
