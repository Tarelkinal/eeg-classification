import pandas as pd
from typing import List, Optional
from dataclasses import dataclass, field
from marshmallow.exceptions import ValidationError


@dataclass
class Data:
    df: pd.DataFrame
    id_name: str
    target_name: str
    num_feature_list: Optional[List[str]] = field(default_factory=list)
    cat_feature_list: Optional[List[str]] = field(default_factory=list)
    array_feature_list: Optional[List[str]] = field(default_factory=list)
    valid_flag_name: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.id_name not in self.df.columns:
            raise ValidationError(f'{self.id_name} should be in df columns')
        if self.target_name not in self.df.columns:
            raise ValidationError(f'{self.target_name} should be in df columns')
        if set(self.num_feature_list) & set(self.cat_feature_list):
            raise ValidationError('num feature and cat feature cant have the same name')

    @property
    def features(self):
        features_list = self.num_feature_list + self.cat_feature_list

        if self.valid_flag_name:
            X_train = self.df.loc[~self.df[self.valid_flag_name], features_list]
            y_train = self.df.loc[~self.df[self.valid_flag_name], self.target_name]
            X_valid = self.df.loc[self.df[self.valid_flag_name], features_list]
            y_valid = self.df.loc[self.df[self.valid_flag_name], self.target_name]

            return X_train, y_train, X_valid, y_valid

        X = self.df[features_list]
        y = self.df[self.target_name]

        return X, y
