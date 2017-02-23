import numpy as np
import logging

logger = logging.getLogger('root')

def logloss(target,prediction,year=None):
    ALL = target.shape[0]
    sum = 0.0
    for i in range(ALL):
        if target.iloc[i]['GamesPlayed'] != 0:
            y_true = target.iloc[i]['teamAWins']
            y_pred = prediction.iloc[i]['pred']
            if(y_pred!=0.0) & (y_pred!=1.0):
                sum += y_true*np.log(y_pred) + (1-y_true) * np.log(1-y_pred)
            elif (y_pred == 0.0):
                sum += y_true * np.log(0.05) + (1 - y_true) * np.log(1 - 0.05)
            else:
                sum += y_true * np.log(0.95) + (1 - y_true) * np.log(1 - 0.95)
            #logger.info(sum)

    N = target[target['GamesPlayed'] != 0].shape[0]
    score = -(1/N)*sum
    if year is None:
        logger.info("Log loss score: {0} , no. games considered {1}".format(score,N))
    else:
        logger.info("Log loss score: {0} , Tournament year: {1}, no. games considered {2}".format(score, year, N))
    return score

def accuracy(target, prediction, year = None):
    ALL = target.shape[0]
    sum = 0.0
    for i in range(ALL):
        if target.iloc[i]['GamesPlayed'] != 0:
            y_true = target.iloc[i]['teamAWins']
            y_pred = prediction.iloc[i]['pred']
            if y_pred >= 0.5:
                y_pred = 1.0
            else:
                y_pred = 0.0
            if (y_pred == y_true):
                sum += 1
    N = target[target['GamesPlayed'] != 0].shape[0]
    score = sum / N
    if year is None:
        logger.info("Accuracy score: {0} , no. games considered {1}".format(score, N))
    else:
        logger.info("Accuracy score: {0} , Tournament year: {1}, no. games considered {2}".format(score, year, N))
    return score




