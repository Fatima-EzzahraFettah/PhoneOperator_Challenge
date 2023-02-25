import pandas as pd
import os
# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def _read_data(path, type_):

    fname = 'churn-data-{}.csv'.format(type_)
    fp = os.path.join(path, 'data', 'phone_operator_churn',fname)
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


X, Y = get_train_data()

