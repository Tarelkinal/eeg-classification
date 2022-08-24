import numpy as np
import pandas as pd
import pytest


TRAIN_DATA_PATH = 'tests/data/train_tiny.npy'
TEST_DATA_PATH = 'tests/data/test_tiny.npy'


@pytest.fixture(scope='session')
def train_test_df():
    train_data = np.load(TRAIN_DATA_PATH, allow_pickle=True)
    df_train = pd.DataFrame(list(train_data))

    test_data = np.load(TEST_DATA_PATH, allow_pickle=True)
    df_test = pd.DataFrame(list(test_data))

    return df_train, df_test


@pytest.fixture()
def stat_features_column_list():
    return [
        'fpz_alpha_mean',
        'fpz_alpha_max',
        'fpz_alpha_min',
        'fpz_alpha_std',
        'fpz_alpha_median',
        'fpz_alpha_q95',
        'fpz_alpha_q5',
        'fpz_delta_mean',
        'fpz_delta_max',
        'fpz_delta_min',
        'fpz_delta_std',
        'fpz_delta_median',
        'fpz_delta_q95',
        'fpz_delta_q5',
        'fpz_theta_mean',
        'fpz_theta_max',
        'fpz_theta_min',
        'fpz_theta_std',
        'fpz_theta_median',
        'fpz_theta_q95',
        'fpz_theta_q5',
        'pz_alpha_mean',
        'pz_alpha_max',
        'pz_alpha_min',
        'pz_alpha_std',
        'pz_alpha_median',
        'pz_alpha_q95',
        'pz_alpha_q5',
        'pz_delta_mean',
        'pz_delta_max',
        'pz_delta_min',
        'pz_delta_std',
        'pz_delta_median',
        'pz_delta_q95',
        'pz_delta_q5',
        'pz_theta_mean',
        'pz_theta_max',
        'pz_theta_min',
        'pz_theta_std',
        'pz_theta_median',
        'pz_theta_q95',
        'pz_theta_q5',
        ]
