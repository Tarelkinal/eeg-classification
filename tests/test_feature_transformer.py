import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from pdb import set_trace

from src.data import FFTransformer, DataTransformer, \
    TrainValidSplitByColumnTransformer


@pytest.mark.parametrize(
    'scaler, value, mean, _min, _max',
    [
        pytest.param(
            'min_max',
            np.array([[1, 2, 3], [1, 2, 3]]),
            np.array([0.5, 0.5]),
            np.array([0, 0]),
            np.array([1, 1]),
            id='test min_max scaler'
        ),
        pytest.param(
            'standard',
            np.array([[1, 2, 3], [1, 2, 3]]),
            np.array([0, 0]),
            np.array([-1.225, -1.225]),
            np.array([1.225, 1.225]),
            id='test standard scaler'
        )
    ]
)
def test_FFTransformer_scaler(
        scaler,
        value,
        mean,
        _min,
        _max,
):

    fft = FFTransformer(
        eeg_value_column_name='value',
        time_period_sec=5,
        scaler=scaler,
        build_stat_feature=True
    )

    result = fft.scaler(value)
    assert result.shape[0] == 2
    assert result.min(axis=1) == pytest.approx(_min, rel=1e-2)
    assert result.max(axis=1) == pytest.approx(_max, rel=1e-2)
    assert result.mean(axis=1) == pytest.approx(mean)


def test_FFTransformer_can_be_fitted(train_test_df):
    df, _ = train_test_df

    fft = FFTransformer(
        eeg_value_column_name='value',
        time_period_sec=5,
        scaler='min_max',
        build_stat_feature=True
    )
    fft = fft.fit(df)
    spector_width_index_list_ = [x[0] for x in fft.spector_width_index_list]
    assert [0, 20, 40, 70] == spector_width_index_list_
    assert 500 == fft.num


def test_FFTransformer_can_create_fft_features(stat_features_column_list, train_test_df):
    df, _ = train_test_df

    fft = FFTransformer(
        eeg_value_column_name='value',
        time_period_sec=5,
        scaler='min_max',
        build_stat_feature=False,
        drop_eeg_value_column=True,
        build_eeg_raw_features=False,
        fft_left_bound=200,
    )
    df = fft.fit_transform(df)
    assert 'fft_feature' in df.columns
    assert 'value' not in df.columns
    assert df.iloc[0]['fft_feature'].shape[0] == 200
    assert list(df.iloc[0]['fft_feature'].T.min(axis=1)) == [0, 0]
    assert all(x not in df.columns for x in stat_features_column_list)


def test_FFTransformer_can_create_stat_features(stat_features_column_list, train_test_df):
    df, _ = train_test_df

    fft = FFTransformer(
        eeg_value_column_name='value',
        time_period_sec=5,
        scaler='min_max',
        build_stat_feature=True,
        build_eeg_raw_features=False,
    )
    df = fft.fit_transform(df)

    assert all(x in df.columns for x in stat_features_column_list)


def test_FFTransformer_can_create_raw_eeg_features(stat_features_column_list, train_test_df):
    df, _ = train_test_df

    fft = FFTransformer(
        eeg_value_column_name='value',
        time_period_sec=5,
        scaler='min_max',
        build_stat_feature=True,
        build_eeg_raw_features=True,
    )
    df = fft.fit_transform(df)

    assert all(x in df.columns for x in stat_features_column_list)
    assert 1042 == len(fft.feature_name_list)


def test_FFTransformer_can_create_raw_fft_features(stat_features_column_list, train_test_df):
    df, _ = train_test_df

    fft = FFTransformer(
        eeg_value_column_name='value',
        time_period_sec=5,
        scaler='min_max',
        build_stat_feature=False,
        build_fft_raw_features=True,
        build_eeg_raw_features=False,
        drop_fft_feature_column=True,
        drop_eeg_value_column=True,
    )
    df_ = fft.fit_transform(df)

    assert 200 == len(fft.feature_name_list)
    assert df.shape[0] == df_.shape[0]
    assert 'value' not in df_.columns
    assert 'fft_feature' not in df_.columns


def test_FFTransformer_raise_error_when_call_transform_without_fitting_before(train_test_df):
    df, _ = train_test_df

    fft = FFTransformer(
        eeg_value_column_name='value',
        time_period_sec=5,
        scaler='min_max',
        build_stat_feature=True
    )
    with pytest.raises(NotFittedError):
        df = fft.transform(df)


def test_DataTransformer_can_filter_values(train_test_df):
    df, _ = train_test_df

    assert 'Sleep stage 4' in df['label'].unique()

    dtt = DataTransformer(
        filter_values={'label': ['Sleep stage 4']}
    )
    df = dtt.transform(df)

    assert 'Sleep stage 4' not in df['label'].unique()


def test_DataTransformer_can_map_values(train_test_df):
    df, _ = train_test_df

    assert 'Sleep stage 4' in df['label'].unique()

    dtt = DataTransformer(
        map_values={'label': {'Sleep stage 4': 1}}
    )
    df = dtt.transform(df)

    assert 'Sleep stage 4' not in df['label'].unique()
    assert 1 in df['label'].unique()


def test_TrainValidSplitbyColumnTransformer_can_split_data(train_test_df):
    df, _ = train_test_df
    df = df[df['label'].isin([
        'Sleep stage 4', 'Sleep stage W'
    ])]
    df['label'] = df['label'].map({
        'Sleep stage 4': 1,
        'Sleep stage W': 0,
    })

    tvs = TrainValidSplitByColumnTransformer(
        split_column_name='person',
        target_column_name='label',
        valid_size=0.5,
    )
    df = tvs.fit_transform(df)
    assert 'valid_flag' in df.columns

    valid_mean_target = df.loc[df['valid_flag'], 'label'].mean()
    assert valid_mean_target != 0

    train_mean_target = df.loc[df['valid_flag'], 'label'].mean()
    assert train_mean_target != 0
