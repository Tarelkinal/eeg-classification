from typing import Optional, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import fft
from pdb import set_trace


DELTA_BOUND_HZ = 4
THETA_BOUND_HZ = 8
ALPHA_BOUND_HZ = 14


class FFTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            time_period_sec: int,
            eeg_value_column_name: str,
            scaler: str = 'min_max',
            fft_feature_name='fft_feature',
            fft_left_bound: int = 100,
            build_stat_feature: bool = True,
            build_fft_raw_features: bool = False,
            build_eeg_raw_features: bool = False,
            drop_eeg_value_column: bool = False,
            drop_fft_feature_column: bool = False,
    ):
        self.time_period_sec = time_period_sec
        self._scaler = scaler
        self.build_stat_feature = build_stat_feature
        self.feature_name_list = []
        self.eeg_value_column_name = eeg_value_column_name
        self.fft_feature_name = fft_feature_name
        self.fft_left_bound = fft_left_bound
        self.drop_eeg_value_column = drop_eeg_value_column
        self.drop_fft_feature_column = drop_fft_feature_column
        self.build_eeg_raw_features = build_eeg_raw_features
        self.build_fft_raw_features = build_fft_raw_features

        if self._scaler == 'min_max':
            self.scaler = self._min_max_scaler
        elif self._scaler == 'standard':
            self.scaler = self._standard_scaler
        else:
            raise NotImplementedError()

    def fit(self, df: pd.DataFrame, y=None):
        self.spector_width_index_list = []
        num = df.iloc[0][self.eeg_value_column_name].shape[0]
        freq_list = fft.fftfreq(num, self.time_period_sec / num)[:num // 2]
        self.spector_width_index_list.append((0, ''))

        # delta spector
        delta_left_bound = np.where(freq_list >= DELTA_BOUND_HZ)[0][0]
        self.spector_width_index_list.append((delta_left_bound, 'delta'))

        # theta spector
        delta_left_bound = np.where(freq_list >= THETA_BOUND_HZ)[0][0]
        self.spector_width_index_list.append((delta_left_bound, 'theta'))

        # alpha spector
        delta_left_bound = np.where(freq_list >= ALPHA_BOUND_HZ)[0][0]
        self.spector_width_index_list.append((delta_left_bound, 'alpha'))

        self.num = num

        self.fft_feature_list = []
        if self.build_fft_raw_features:
            for name in ['fft_fpz', 'fft_pz']:
                length = min(df[self.eeg_value_column_name].iloc[0].shape[0], self.fft_left_bound)
                for i in range(length):
                    self.fft_feature_list.append(f'{name}_{i}')

            self.feature_name_list.extend(self.fft_feature_list)

        self.eeg_feature_list = []
        if self.build_eeg_raw_features:
            for name in ['fpz', 'pz']:
                for i in range(df[self.eeg_value_column_name].iloc[0].shape[0]):
                    self.eeg_feature_list.append(f'{name}_{i}')
            self.feature_name_list.extend(self.eeg_feature_list)

        return self

    def transform(self, df: pd.DataFrame, y=None):
        check_is_fitted(
            self,
            ['spector_width_index_list', 'num', 'eeg_feature_list', 'fft_feature_list']
        )
        df = df.copy()
        df[self.eeg_value_column_name] = df[self.eeg_value_column_name].apply(
            lambda x: x.astype(np.float32)
        )

        df[self.fft_feature_name] = df[self.eeg_value_column_name].apply(
            lambda x: self.scaler(
                2.0 / self.num * np.abs(fft.fft(x.T))[0: self.num // 2]
            ).T[:self.fft_left_bound]
        )

        if self.build_stat_feature:
            for i in tqdm(range(len(self.spector_width_index_list) - 1)):
                left_bound, _ = self.spector_width_index_list[i]
                right_bound, spector_name = self.spector_width_index_list[i + 1]
                fpz_mean, pz_mean = np.stack(df[self.fft_feature_name].apply(
                    lambda x: x.T[:, left_bound: right_bound].mean(axis=1)
                ).values, axis=0).T
                fpz_max, pz_max = np.stack(df[self.fft_feature_name].apply(
                    lambda x: x.T[:, left_bound: right_bound].max(axis=1)
                ).values, axis=0).T
                fpz_min, pz_min = np.stack(df[self.fft_feature_name].apply(
                    lambda x: x.T[:, left_bound: right_bound].min(axis=1)
                ).values, axis=0).T
                fpz_std, pz_std = np.stack(df[self.fft_feature_name].apply(
                    lambda x: x.T[:, left_bound: right_bound].std(axis=1)
                ).values, axis=0).T
                fpz_median, pz_median = np.stack(df[self.fft_feature_name].apply(
                    lambda x: np.quantile(x.T[:, left_bound: right_bound], q=.5, axis=1)
                ).values, axis=0).T
                fpz_q95, pz_q95 = np.stack(df[self.fft_feature_name].apply(
                    lambda x: np.quantile(x.T[:, left_bound: right_bound], q=.95, axis=1)
                ).values, axis=0).T
                fpz_q05, pz_q05 = np.stack(df[self.fft_feature_name].apply(
                    lambda x: np.quantile(x.T[:, left_bound: right_bound], q=.05, axis=1)
                ).values, axis=0).T

                df = self._add_feature(df, f'fpz_{spector_name}_mean', fpz_mean)
                df = self._add_feature(df, f'fpz_{spector_name}_max', fpz_max)
                df = self._add_feature(df, f'fpz_{spector_name}_min', fpz_min)
                df = self._add_feature(df, f'fpz_{spector_name}_std', fpz_std)
                df = self._add_feature(df, f'fpz_{spector_name}_median', fpz_median)
                df = self._add_feature(df, f'fpz_{spector_name}_q95', fpz_q95)
                df = self._add_feature(df, f'fpz_{spector_name}_q5', fpz_q05)

                df = self._add_feature(df, f'pz_{spector_name}_mean', pz_mean)
                df = self._add_feature(df, f'pz_{spector_name}_max', pz_max)
                df = self._add_feature(df, f'pz_{spector_name}_min', pz_min)
                df = self._add_feature(df, f'pz_{spector_name}_std', pz_std)
                df = self._add_feature(df, f'pz_{spector_name}_median', pz_median)
                df = self._add_feature(df, f'pz_{spector_name}_q95', pz_q95)
                df = self._add_feature(df, f'pz_{spector_name}_q5', pz_q05)

        if self.build_fft_raw_features:
            fft_flatten = np.stack(df[self.fft_feature_name].apply(
                lambda x: x.T.flatten()
            ).values, axis=0)

            df_fft_raw = pd.DataFrame(fft_flatten, columns=self.fft_feature_list)
            df = pd.concat([df, df_fft_raw], axis=1)


        if self.drop_fft_feature_column:
            df = df.drop(columns=[self.fft_feature_name])

        if self.build_eeg_raw_features:
            eeg_flatten = np.stack(df[self.eeg_value_column_name].apply(
                lambda x: self.scaler(x.T).flatten()
            ).values, axis=0)
            df_eeg_raw = pd.DataFrame(eeg_flatten, columns=self.eeg_feature_list)
            df = pd.concat([df, df_eeg_raw], axis=1)


        if self.drop_eeg_value_column:
            df = df.drop(columns=[self.eeg_value_column_name])

        return df

    def _add_feature(self, df, feature_name, feature_value):
        if feature_name not in self.feature_name_list:
            self.feature_name_list.append(feature_name)
        df[feature_name] = feature_value
        return df

    @staticmethod
    def _min_max_scaler(value):
        result = (value - value.min(axis=1).reshape(2, -1)) \
                 / (value.max(axis=1).reshape(2, -1) - value.min(axis=1).reshape(2, -1))

        return result

    @staticmethod
    def _standard_scaler(value):
        result = (value - value.mean(axis=1).reshape(2, -1)) \
                 / value.std(axis=1).reshape(2, -1)

        return result


class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            filter_values: Optional[dict] = None,
            map_values: Optional[dict] = None,
    ):
        self.filter_values = filter_values or {}
        self.map_values = map_values or {}

    def fit(self, df: pd.DataFrame, y=None):
        return self

    def transform(self, df: pd.DataFrame, y=None):
        df = df.copy()
        for k, v in self.filter_values.items():
            df = df[~df[k].isin(v)]

        for k, v in self.map_values.items():
            df[k] = df[k].map(v)

        df = df.reset_index().drop(columns=['index'])

        return df


class TrainValidSplitByColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            split_column_name: str,
            target_column_name: str,
            valid_size: float,
            random_state: int = 0,
    ):
        self.split_column_name = split_column_name
        self.valid_size = valid_size
        self.random_state = random_state
        self.target_column_name = target_column_name

    def fit(self, df: pd.DataFrame, y=None):
        split_column_unique_values = df[self.split_column_name].unique()

        if len(split_column_unique_values) <= 1:
            raise ValueError(
                f'data can not be split by column {self.split_column_name}'
            )

        train, valid = train_test_split(
            split_column_unique_values,
            test_size=self.valid_size,
            random_state=self.random_state
        )

        if df.loc[df[self.split_column_name].isin(train), self.target_column_name].mean() == 0:
            raise ValueError(
                f'data can not be split with random state {self.random_state}'
            )

        if df.loc[df[self.split_column_name].isin(valid), self.target_column_name].mean() == 0:
            raise ValueError(
                f'data can not be split with random state {self.random_state}'
            )

        self.train_value_list = train
        self.valid_value_list = valid

        return self

    def transform(self, df: pd.DataFrame, y=None):
        check_is_fitted(self, ['train_value_list', 'valid_value_list'])
        df = df.copy()
        df['valid_flag'] = False
        df.loc[df[self.split_column_name].isin(self.valid_value_list), 'valid_flag'] = True

        return df
