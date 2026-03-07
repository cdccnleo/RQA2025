from __future__ import annotations
from datetime import datetime

import logging

logger = logging.getLogger(__name__)


class FeatureMetadata:

    def __init__(self, feature_params=None, data_source_version=None, feature_list=None, scaler_path=None, selector_path=None, version=None):

        import copy
        if feature_params is not None and not isinstance(feature_params, dict):
            raise TypeError("feature_params must be a dict")
        self.feature_params = copy.deepcopy(feature_params) if feature_params is not None else {}
        self.data_source_version = data_source_version if data_source_version is not None else (
            version if version is not None else "1.0")
        if feature_list is not None:
            if len(feature_list) != len(set(feature_list)):
                raise ValueError("feature_list contains duplicate features")
            if any(f == '' for f in feature_list):
                raise ValueError("feature_list contains empty feature name")
            self.feature_list = copy.deepcopy(feature_list)
        else:
            self.feature_list = []
        self.scaler_path = scaler_path
        self.selector_path = selector_path
        self.metadata = {}
        now = datetime.now().timestamp()
        self.created_at = now
        self.last_updated = now

    def update(self, new_feature_list):

        self.update_feature_columns(new_feature_list)

    def update_feature_columns(self, new_feature_list):

        self.feature_list = list(new_feature_list)
        self.last_updated = datetime.now().timestamp()

    def validate_compatibility(self, other):

        return True

    def _validate_alignment(self, data):

        pass

    def save(self, path):

        pass

    def save_metadata(self, path):

        pass
