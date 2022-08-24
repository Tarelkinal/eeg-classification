import logging
import os

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
import pickle

from src.utils import Data


@hydra.main(
    config_path='../../conf',
    config_name='config',
    version_base='1.2',
)
def main(cfg: DictConfig):
    logger_name = os.environ.get('LOGGER_NAME') or __file__
    logger = logging.getLogger(logger_name)

    project_dir = os.environ.get('PROJECT_DIR')

    train_data_path = Path(f'{project_dir}/{cfg.paths.raw_data}/{cfg.files.train}')
    logger.info('read train data from: %s', train_data_path)
    train_data = np.load(train_data_path, allow_pickle=True)
    df_train = pd.DataFrame(list(train_data))

    test_data_path = Path(f'{project_dir}/{cfg.paths.raw_data}/{cfg.files.test}')
    logger.info('read train data from: %s', test_data_path)
    test_data = np.load(test_data_path, allow_pickle=True)
    df_test = pd.DataFrame(list(test_data))
    test_count = df_test.shape[0]

    logger.info('raw train data count: %s', train_data.shape[0])
    logger.info('raw test data count: %s', test_data.shape[0])

    # clear data
    logger.info(
        'start clear data with params:\n%s',
        OmegaConf.to_yaml(cfg.prep_features.data_transformer)
    )
    dtt = instantiate(cfg.prep_features.data_transformer)

    df_train = dtt.fit_transform(df_train)
    logger.info('cleaned train data count: %s', df_train.shape[0])

    df_test = dtt.transform(df_test)
    if df_test.shape[0] != test_count:
        raise ValueError('test dataset size should not be changed')

    logger.info('cleaned test data count: %s', df_test.shape[0])

    # train valid split
    logger.info(
        'start train split to train/valid with params:\n%s',
        OmegaConf.to_yaml(cfg.prep_features.train_valid_transformer)
    )
    tvs = instantiate(cfg.prep_features.train_valid_transformer)
    df_train = tvs.fit_transform(df_train)

    train_mean_target = df_train.loc[
        ~df_train['valid_flag'], cfg.general.raw_data.target_column_name
    ].mean()

    logger.info('train data count: %s', df_train[~df_train['valid_flag']].shape[0])
    logger.info('train mean target: %s', train_mean_target)

    valid_mean_target = df_train.loc[
        df_train['valid_flag'], cfg.general.raw_data.target_column_name
    ].mean()
    logger.info('valid data count: %s', df_train[df_train['valid_flag']].shape[0])
    logger.info('valid mean target: %s', valid_mean_target)

    # feature transform
    logger.info(
        'start feature transform with params:\n%s',
        OmegaConf.to_yaml(cfg.prep_features.feature_transformer)
    )

    fft = instantiate(cfg.prep_features.feature_transformer)
    df_train = fft.fit_transform(df_train)
    df_test = fft.transform(df_test)

    # put data in dataclass and save
    df_train_data = Data(
        df=df_train,
        id_name=cfg.general.raw_data.id_column_name,
        target_name=cfg.general.raw_data.target_column_name,
        num_feature_list=fft.feature_name_list,
        valid_flag_name='valid_flag'
    )
    df_train_data_path = Path(f'{project_dir}/{cfg.paths.prep_data}/train.pickle')

    logger.info('save prep train data into: %s', df_train_data_path)
    with open(df_train_data_path, 'wb') as f_out:
        pickle.dump(df_train_data, f_out)

    df_test_data = Data(
        df=df_test,
        id_name=cfg.general.raw_data.id_column_name,
        target_name=cfg.general.raw_data.target_column_name,
        num_feature_list=fft.feature_name_list,
    )
    df_test_data_path = Path(f'{project_dir}/{cfg.paths.prep_data}/test.pickle')
    logger.info('save prep test data into: %s', df_test_data_path)
    with open(df_test_data_path, 'wb') as f_out:
        pickle.dump(df_test_data, f_out)


if __name__ == '__main__':
    main()
