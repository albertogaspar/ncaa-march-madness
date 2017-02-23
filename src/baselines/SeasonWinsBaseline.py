import os
import pandas as pd
import logging
import numpy as np
from src.evaluation import Metrics
from src import LoadDatasets

logger = logging.getLogger('root')
ABSOLUTE_PATH = LoadDatasets.get_absolute_path()

class SeasonWinsBaseline(object):
    """
    Simple baseline that consider the result of the last two years
    (regular season, tournament, regular season + tournament)
    and assign a probability (no. win / no games played) to each matchup.
    We have (in general) 68 teams so we'll predict 68*67/2 = 2278 game results.
    """

    def __init__(self):
        return

    def predict(self,target, data_to_consider='r', weights_decay='exp', threashold=5, decay=1):
        """
        :param target:
        :param data_to_consider: 'r' (consider data from the previous 4 regular season only),
                                 't' (consider data from the previous 4 tournament only),
                                  'rt' (consider both)
        :param weights: 'linear', 'exp' (exponential decay)
        :return:
        """
        self.data_to_consider = data_to_consider
        self.target = target
        self.weights_decay = weights_decay
        self.threashold = threashold*2+1
        self.decay=decay

        years = self.target['Season'].unique()
        for year in years:
            logger.info("Generating prediction for year {0}".format(year))
            self._load_files(year)
            self._set_weights()
            #Seasons are weighted depending on their year (the older the data, the smaller their importance)
            for i,df in enumerate(self.files):
                df['GamesPlayed'] = df['GamesPlayed'] * self.weights[i]
                df['teamAWins'] = df['teamAWins'] * self.weights[i]
                df['MOV'] = df['MOV'] * self.weights[i]

            for i in self.target[self.target['Season']==year].index:
                prob = self._sum_probabilities(i)
                if prob != 0.0:
                    self.target.set_value(i,'pred',prob)
                else:
                    self.target.set_value(i, 'pred', 0.3)
                #logger.info("{0}-->{1}".format(i,self.target.loc[i]['pred']))
            self._evaluate(year, self.target[self.target['Season']==year])

        self.target = self.target.drop(['Season','teamA','teamB'],axis=1)
        self._evaluate_all(self.target)
        return self.target

    def _load_files(self,year):
        path = ABSOLUTE_PATH+"data/matchups_by_year/{0}".format(year)
        if self.data_to_consider == 'r':
            filenames = [f for f in os.listdir(path) if not f.__contains__('tourney')]
        elif self.data_to_consider == 't':
            filenames = [f for f in os.listdir(path) if f.__contains__('tourney')]
        elif self.data_to_consider == 'rt':
            filenames = [f for f in os.listdir(path)]
        else:
            print("Error")



        #files in increasing order of year
        filenames.sort()
        filenames = filenames[-self.threashold:]
        logger.info("Loaded files succesfully : {0}".format(filenames))
        self.files = [None]*len(filenames)
        for i,f in enumerate(filenames):
            self.files[i]=pd.read_csv(os.path.join(path,f), index_col=0)

    def _sum_probabilities(self, idx):
        num = 0.0
        denom = 0.0
        for df in self.files:
            num += df.loc[idx]['teamAWins'].item()
            denom += df.loc[idx]['GamesPlayed'].item()

        if denom>7: #enough support
            return num/denom
        elif denom > 0: #prevent division by zero
            return num/(denom+1)
        else:
            return 0.5

    def _set_weights(self):
        if self.weights_decay == 'linear':
            self.weights = np.linspace(0, 1, num=self.files.__len__())
        elif self.weights_decay == 'exp':
            self.weights = np.exp(-(np.linspace(0, 1, num=self.files.__len__()))*self.decay)[::-1] #*2 so that it decays faster
        else:
            logger.error("Invalid argument for weights_decay : {0} not supported.".format(self.weights_decay))

    def _evaluate(self,year, prediction):
        target = pd.read_csv(ABSOLUTE_PATH+"data/evaluation/{}_target_no_play_in.csv".format(year))
        Metrics.logloss(target,prediction,year=year)
        Metrics.accuracy(target, prediction, year=year)

    def _evaluate_all(self, prediction):
        target = pd.read_csv(ABSOLUTE_PATH+"data/evaluation/ALL_target_no_play_in.csv")
        Metrics.logloss(target,prediction)
        Metrics.accuracy(target, prediction)