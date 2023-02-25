import pandas as pd
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, recall_score, precision_score, confusion_matrix
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.workflows.sklearn_pipeline import SKLearnPipeline
from rampwf.workflows.sklearn_pipeline import Estimator

problem_title = 'Phone Operator Churn classification'


_prediction_label_name = [0, 1]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_name
)
workflow = SKLearnPipeline()


# -----------------------------------------------------------------------------
# Evaluation metric: Matthews correlation coefficient (MCC) 
# -----------------------------------------------------------------------------


class MCC(BaseScoreType):
    is_lower_the_better = False
    minimum = -1
    maximum = 1

    def __init__(self, name="MCC", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        term1 = tp*tn - fp*fn
        term2 = np.sqrt((tp + fp )*(tp + fn)*(tn + fp)*(tn + fn))
        return term1 / term2 

score_types = [
    MCC(name="MCC"),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)


# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------

def _read_data(path, type_):

    fname = 'churn-data-{}.csv'.format(type_)
    fp = os.path.join(path, 'data',fname)
    data = pd.read_csv(fp)
    col_names = list(data.columns)
    col_names.remove('Churn')
    features = data[col_names]
    y = data['Churn']

    # for the "quick-test" mode, use less data
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        N_small = 35000
        features = features[:N_small]
        y = y[:N_small]

    return features, y


def get_train_data(path='.'):
    return _read_data(path, 'train')


def get_test_data(path='.'):
    return _read_data(path, 'test')


#X, Y = get_train_data()

