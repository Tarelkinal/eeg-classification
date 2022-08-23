from src.utils import Data


def test_dataclass_can_get_train_valid_features(train_test_df):
    df, _ = train_test_df
    df['valid_flag'] = False
    df.loc[:4, 'valid_flag'] = True

    df_data = Data(
        df=df,
        id_name='person',
        target_name='label',
        num_feature_list=['time'],
        valid_flag_name='valid_flag'
    )
    train, y_train, valid, y_valid = df_data.features

    assert 'time' in train.columns
    assert 'time' in valid.columns
    assert valid.shape == (5, 1)
    assert df_data.num_feature_list == ['time']
    assert df_data.cat_feature_list == []
