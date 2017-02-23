import pandas as pd
from src.evaluation import Metrics
from src import LoadDatasets
import logging

logger = logging.getLogger('root')
ABSOLUTE_PATH = LoadDatasets.get_absolute_path()

class TopSeedClassifier(object):

    def __init__(self, tourneySeeds):
        self.tourneySeeds = tourneySeeds

    def _build_dict(self, year):
        team_seed = self.tourneySeeds[self.tourneySeeds['Season']==year].groupby('Team')['Seed'].apply(list)
        team_seed = team_seed.map(lambda x : x[0])
        return team_seed.to_dict()

    def predict(self, target):
        self.target = target
        for year in target['Season'].unique():
            team_seed_dict = self._build_dict(year)
            for i in self.target[self.target['Season']==year].index:
                teamA = self.target.loc[i]['teamA']
                teamB = self.target.loc[i]['teamB']
                prob = 0.5 - 0.03*(team_seed_dict[teamA]-team_seed_dict[teamB])
                self.target.set_value(i,'pred',prob)
                #logger.info("{0}-->{1}, {2}, {3}".format(i,self.target.loc[i]['pred'], team_seed_dict[teamA], team_seed_dict[teamB]))
            self._evaluate(year, self.target[self.target['Season']==year])

        self.target = self.target.drop(['Season','teamA','teamB'],axis=1)
        self._evaluate_all(self.target)
        return self.target

    def _evaluate(self,year, prediction):
        target = pd.read_csv(ABSOLUTE_PATH+"data/evaluation/{}_target_no_play_in.csv".format(year))
        Metrics.logloss(target,prediction,year=year)
        Metrics.accuracy(target, prediction, year=year)

    def _evaluate_all(self, prediction):
        target = pd.read_csv(ABSOLUTE_PATH+"data/evaluation/ALL_target_no_play_in.csv")
        Metrics.logloss(target,prediction)
        Metrics.accuracy(target, prediction)


