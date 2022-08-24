from textwrap import dedent
from typing import Optional, Union
import logging
import os

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from sklearn.metrics import \
    precision_recall_curve, roc_auc_score
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import yaml

from src.utils import report_update, report_render

plt.rcParams.update({'font.size': 14})


def pr_evaluate(
        y_true: Union[list, np.array, pd.Series],
        y_pred: Union[list, np.array, pd.Series],
        baseline_recall: float,
        fig_path: Optional[Path] = None,
) -> tuple:
    """
    :param y_true: iterable with target
    :param y_pred: iterable with prediction
    :param baseline_recall: down boundary for recall estimation.
    :param fig_path: path to save pr_curve
    :return: best precision and recall
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ind = np.where(recall >= baseline_recall)[0].max()
    ind_ = np.argmax(precision[:ind + 1])
    precision_score, recall_score = precision[ind_], recall[ind_]

    if fig_path:
        title = Path(fig_path).name.split('-', maxsplit=1)[0]
        fig = plt.figure(figsize=(15, 10))
        plt.scatter(
            x=[recall_score],
            y=[precision_score],
            marker='*',
            c='r',
            s=150,
            label=dedent(
                f'''
                best value:
                recall = {round(recall_score, 3)},
                precision = {round(precision_score, 3)}
                '''
            )
        )
        plt.title(title)
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend()
        plt.grid()
        fig.savefig(fig_path)

    return precision_score, recall_score


@hydra.main(
    config_path='../../conf',
    config_name='config',
    version_base='1.2',
)
def main(cfg: DictConfig):

    experiment_report = {}
    logger_name = os.environ.get('LOGGER_NAME') or __file__
    logger = logging.getLogger(logger_name)
    logger.info('experiment name: %s', cfg.experiment_name)
    experiment_report['experiment_name'] = cfg.experiment_name

    project_dir = os.environ.get('PROJECT_DIR')

    train_data_path = Path(f'{project_dir}/{cfg.paths.prep_data}/train.pickle')
    logger.info('load train data from: %s', train_data_path)
    with open(train_data_path, 'rb') as f_in:
        train = pickle.load(f_in)

    X_train, y_train, X_valid, y_valid = train.features

    experiment_report['num_features'] = X_train.shape[1]
    experiment_report['num_train_objects'] = X_train.shape[0]

    test_data_path = Path(f'{project_dir}/{cfg.paths.prep_data}/test.pickle')
    logger.info('load test data from: %s', test_data_path)
    with open(test_data_path, 'rb') as f_in:
        test = pickle.load(f_in)

    X_test, y_test = test.features

    report_path = Path(f'{project_dir}/{cfg.paths.report}/{cfg.files.report}')

    with open(report_path, 'r') as f_in:
        report = yaml.safe_load(f_in)

    best_model_params_path = report['best_model_params_path']
    best_model_params_path = Path(
        f'{project_dir}/{best_model_params_path}/optimization_results.yaml'
    )

    with open(best_model_params_path, 'r') as f_in:
        best_model_params = OmegaConf.load(f_in)

    best_params_dict = {
        k.split('.')[-1]: v for (k, v) in best_model_params.best_params.items()
    }

    model = instantiate(cfg.model.lgbm, **best_params_dict)
    early_stopping = instantiate(cfg.model.early_stopping)

    model.fit(
        X=X_train,
        y=y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping],
    )
    experiment_report['num_estimators'] = model.best_iteration_
    train_roc_auc_score = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    logger.info('train roc_auc_score: %s', train_roc_auc_score)
    experiment_report['train_roc_auc_score'] = round(float(train_roc_auc_score), 5)

    valid_roc_auc_score = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    logger.info('valid roc_auc_score: %s', valid_roc_auc_score)
    experiment_report['valid_roc_auc_score'] = round(float(valid_roc_auc_score), 5)

    test_roc_auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    logger.info('test roc_auc_score: %s', test_roc_auc_score)
    experiment_report['test_roc_auc_score'] = round(float(test_roc_auc_score), 5)

    experiment_name_path = cfg.experiment_name.lower().replace(" ", "_")
    fig_path_1 = Path(f'{project_dir}/{cfg.paths.report}/{experiment_name_path}-{cfg.files.pr_fig_1}')
    precision_1_score, recall_1_score = pr_evaluate(
        y_true=y_test,
        y_pred=model.predict_proba(X_test)[:, 1],
        baseline_recall=cfg.baseline.stage_4.recall,
        fig_path=fig_path_1
    )
    logger.info(
        'precision_1: %s, recall_1: %s',
        precision_1_score,
        recall_1_score
    )
    experiment_report['precision_ss4'] = round(float(precision_1_score), 5)
    experiment_report['recall_ss4'] = round(float(recall_1_score), 5)

    precision_1_increase_pers = \
        (precision_1_score - cfg.baseline.stage_4.precision) / cfg.baseline.stage_4.precision

    logger.info(
        'precision_1 increase percentage: %s',
        precision_1_increase_pers * 100
    )
    fig_path_0 = Path(f'{project_dir}/{cfg.paths.report}/{experiment_name_path}-{cfg.files.pr_fig_0}')
    precision_0_score, recall_0_score = pr_evaluate(
        y_true=np.ones(len(y_test)) - np.array(y_test),
        y_pred=model.predict_proba(X_test)[:, 0],
        baseline_recall=cfg.baseline.stage_w.recall,
        fig_path=fig_path_0
    )

    logger.info(
        'precision_0: %s, recall_0: %s',
        precision_0_score,
        recall_0_score
    )
    experiment_report['precision_ssW'] = round(float(precision_0_score), 5)
    experiment_report['recall_ssW'] = round(float(recall_0_score), 5)

    precision_0_increase_pers = \
        (precision_0_score - cfg.baseline.stage_w.precision) / cfg.baseline.stage_w.precision

    logger.info(
        'precision_0 increase percentage: %s',
        precision_0_increase_pers * 100
    )

    model_dir_path = Path(f'{project_dir}/{cfg.paths.model}/{experiment_name_path}')
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    model_path = Path(f'{model_dir_path}/{cfg.files.model}')

    with open(model_path, 'wb') as f_out:
        pickle.dump(model, f_out)

    logger.info('save final model to: %s', model_path)

    report['experiments'].update(
        {cfg.experiment_name: experiment_report}
    )
    report_update(report, report_path)
    logger.info('save metrics in report: %s', report_path)

    template_path = Path(f'{project_dir}/{cfg.paths.report}/{cfg.files.report_template}')
    report_render(template_path, report_path)
    logger.info('render report from template: %s', template_path)


if __name__ == '__main__':
    main()
