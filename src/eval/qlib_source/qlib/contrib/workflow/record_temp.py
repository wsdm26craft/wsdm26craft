#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, matthews_corrcoef
from typing import Dict, Text, Any, Optional
from pprint import pprint

from ...contrib.eva.alpha import calc_ic
from ...workflow.record_temp import RecordTemp, ACRecordTemp, SignalRecord
from ...data import dataset as qlib_dataset
from ...log import get_module_logger

logger = get_module_logger("workflow", logging.INFO)


class MultiSegRecord(RecordTemp):
    """
    This is the multiple segments signal record class that generates the signal prediction.
    This class inherits the ``RecordTemp`` class.
    """

    def __init__(self, model, dataset, recorder=None):
        super().__init__(recorder=recorder)
        if not isinstance(dataset, qlib_dataset.DatasetH):
            raise ValueError("The type of dataset is not DatasetH instead of {:}".format(type(dataset)))
        self.model = model
        self.dataset = dataset

    def generate(self, segments: Dict[Text, Any], save: bool = False):
        for key, segment in segments.items():
            predics = self.model.predict(self.dataset, segment)
            if isinstance(predics, pd.Series):
                predics = predics.to_frame("score")
            labels = self.dataset.prepare(
                segments=segment, col_set="label", data_key=qlib_dataset.handler.DataHandlerLP.DK_R
            )
            # Compute the IC and Rank IC
            ic, ric = calc_ic(predics.iloc[:, 0], labels.iloc[:, 0])
            results = {"all-IC": ic, "mean-IC": ic.mean(), "all-Rank-IC": ric, "mean-Rank-IC": ric.mean()}
            logger.info("--- Results for {:} ({:}) ---".format(key, segment))
            ic_x100, ric_x100 = ic * 100, ric * 100
            logger.info("IC: {:.4f}%".format(ic_x100.mean()))
            logger.info("ICIR: {:.4f}%".format(ic_x100.mean() / ic_x100.std()))
            logger.info("Rank IC: {:.4f}%".format(ric_x100.mean()))
            logger.info("Rank ICIR: {:.4f}%".format(ric_x100.mean() / ric_x100.std()))

            if save:
                save_name = "results-{:}.pkl".format(key)
                self.save(**{save_name: results})
                logger.info(
                    "The record '{:}' has been saved as the artifact of the Experiment {:}".format(
                        save_name, self.recorder.experiment_id
                    )
                )


class SignalMseRecord(RecordTemp):
    """
    This is the Signal MSE Record class that computes the mean squared error (MSE).
    This class inherits the ``SignalMseRecord`` class.
    """

    artifact_path = "sig_analysis"
    depend_cls = SignalRecord

    def __init__(self, recorder, **kwargs):
        super().__init__(recorder=recorder, **kwargs)

    def generate(self):
        self.check()

        pred = self.load("pred.pkl")
        label = self.load("label.pkl")
        masks = ~np.isnan(label.values)
        mse = mean_squared_error(pred.values[masks], label[masks])
        metrics = {"MSE": mse, "RMSE": np.sqrt(mse)}
        objects = {"mse.pkl": mse, "rmse.pkl": np.sqrt(mse)}
        self.recorder.log_metrics(**metrics)
        self.save(**objects)
        logger.info("The evaluation results in SignalMseRecord is {:}".format(metrics))

    def list(self):
        return ["mse.pkl", "rmse.pkl"]

class SignalAccMccRecord(ACRecordTemp):
    """
    This is the Signal ACC MCC Record class that computes the accuracy and Matthews correlation coefficient (MCC).
    This class inherits the ``SignalRecord`` class.
    """

    artifact_path = "sig_analysis"
    depend_cls = SignalRecord

    def __init__(self, recorder, label_col=0, skip_existing=False):
        super().__init__(recorder=recorder, skip_existing=skip_existing)
        self.label_col = label_col

    def _generate(self, label: Optional[pd.DataFrame] = None, **kwargs):
        """
        Parameters
        ----------
        label : Optional[pd.DataFrame]
            Label should be a dataframe.
        """
        pred = self.load("pred.pkl")
        if label is None:
            label = self.load("label.pkl")
        if label is None or not isinstance(label, pd.DataFrame) or label.empty:
            logger.warning(f"Empty label.")
            return
        y = pd.DataFrame(label[label.columns[-1]]).rename(columns={label.columns[-1]: 'y'})
        y_pred = pd.DataFrame(pred[pred.columns[-1]]).rename(columns={pred.columns[-1]: 'y'})
        y = (y['y'] > 0).astype(int)
        y_pred = (y_pred['y'] > 0).astype(int)
        acc, mcc = accuracy_score(y, y_pred), matthews_corrcoef(y, y_pred)
        metrics = {"ACC": acc, "MCC": mcc}
        objects = {"acc.pkl": acc, "mcc.pkl": mcc}
        self.recorder.log_metrics(**metrics)
        pprint(metrics)
        return objects

    def list(self):
        paths = ["acc.pkl", "mcc.pkl"]
        return paths