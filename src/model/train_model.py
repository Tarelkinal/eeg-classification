import os
import logging

from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path
from sklearn.metrics import roc_auc_score
import hydra
import pickle

from src.utils import report_update


@hydra.main(
    config_path='../../conf',
    config_name='config',
    version_base='1.2',
)
def main(cfg: DictConfig):
    logger_name = os.environ.get('LOGGER_NAME') or __file__
    logger = logging.getLogger(logger_name)

    project_dir = os.environ.get('PROJECT_DIR')

    train_data_path = Path(f'{project_dir}/{cfg.paths.prep_data}/train.pickle')
    logger.info('load train data from: %s', train_data_path)
    with open(train_data_path, 'rb') as f_in:
        train = pickle.load(f_in)

    X_train, y_train, X_valid, y_valid = train.features

    model = instantiate(cfg.model.lgbm)
    early_stopping = instantiate(cfg.model.early_stopping)

    model.fit(
        X=X_train,
        y=y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping]
    )
    best_iter = model.best_iteration_
    report_update({'best_iter': best_iter}, Path(f'{os.getcwd()}/best_iter.yaml'))
    score = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])

    report = {'best_model_params_path': f'{cfg.output_dir}'}
    report_path = Path(f'{project_dir}/{cfg.paths.report}/{cfg.files.report}')
    report_update(report, report_path)

    return score


if __name__ == '__main__':
    main()
