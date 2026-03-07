from __future__ import annotations

import logging
"""
Feature Metadata Module

This module provides metadata management for features in the feature engineering system.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional
import pickle
import os


logger = logging.getLogger(__name__)


class FeatureMetadata:

    """Feature metadata management class"""

    def __init__(self,


                 feature_params: Optional[Dict[str, Any]] = None,
                 data_source_version: Optional[str] = None,
                 feature_list: Optional[List[str]] = None,
                 scaler_path: Optional[str] = None,
                 selector_path: Optional[str] = None,
                 version: Optional[str] = None,
                 metadata_path: Optional[str] = None):
        """
        Initialize feature metadata

        Args:
            feature_params: Dictionary of feature parameters
            data_source_version: Version of the data source
            feature_list: List of feature names
            scaler_path: Path to the scaler file
            selector_path: Path to the feature selector file
            version: Version string
            metadata_path: Path to save / load metadata
        """
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
        self.metadata_path = metadata_path
        self.metadata = {}

        now = datetime.now().timestamp()
        self.created_at = now
        self.last_updated = now

        # Load existing metadata if path is provided
        if metadata_path and os.path.exists(metadata_path):
            self.load_metadata(metadata_path)

    def update(self, new_feature_list: List[str]) -> None:
        """Update feature list"""
        self.update_feature_columns(new_feature_list)

    def update_feature_columns(self, new_feature_list: List[str]) -> None:
        """Update feature columns list"""
        self.feature_list = list(new_feature_list)
        self.last_updated = datetime.now().timestamp()
        logger.info(f"Updated feature columns: {len(new_feature_list)} features")

    def update_feature_params(self, new_params: Dict[str, Any]) -> None:
        """Update feature parameters"""
        self.feature_params.update(new_params)
        self.last_updated = datetime.now().timestamp()
        logger.info(f"Updated feature parameters: {len(new_params)} parameters")

    def validate_compatibility(self, other: 'FeatureMetadata') -> bool:
        """Validate compatibility with another metadata object"""
        # Basic compatibility check
        if not isinstance(other, FeatureMetadata):
            return False

        # Check if feature lists are compatible
        if set(self.feature_list) != set(other.feature_list):
            return False

        return True

    def _validate_alignment(self, data: Any) -> bool:
        """Validate data alignment with metadata"""
        # This is a placeholder for data alignment validation
        return True

    def save(self, path: str) -> None:
        """Save metadata to file"""
        self.save_metadata(path)

    def save_metadata(self, path: str) -> None:
        """Save metadata to pickle file"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Saved feature metadata to: {path}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {path}: {e}")
            raise

    def load_metadata(self, path: str) -> None:
        """Load metadata from pickle file"""
        try:
            with open(path, 'rb') as f:
                loaded_metadata = pickle.load(f)

            # Update current object with loaded data
            self.feature_params = loaded_metadata.feature_params
            self.feature_list = loaded_metadata.feature_list
            self.scaler_path = loaded_metadata.scaler_path
            self.selector_path = loaded_metadata.selector_path
            self.metadata = loaded_metadata.metadata
            self.created_at = loaded_metadata.created_at
            self.last_updated = loaded_metadata.last_updated

            logger.info(f"Loaded feature metadata from: {path}")
        except Exception as e:
            logger.error(f"Failed to load metadata from {path}: {e}")
            raise

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature information summary"""
        return {
            'feature_count': len(self.feature_list),
            'feature_list': self.feature_list,
            'parameters': self.feature_params,
            'created_at': datetime.fromtimestamp(self.created_at).isoformat(),
            'last_updated': datetime.fromtimestamp(self.last_updated).isoformat(),
            'version': self.data_source_version
        }

    def add_feature(self, feature_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Add a new feature to the metadata"""
        if feature_name not in self.feature_list:
            self.feature_list.append(feature_name)
            self.last_updated = datetime.now().timestamp()

            if params:
                self.feature_params[feature_name] = params

            logger.info(f"Added feature: {feature_name}")

    def remove_feature(self, feature_name: str) -> None:
        """Remove a feature from the metadata"""
        if feature_name in self.feature_list:
            self.feature_list.remove(feature_name)
            self.last_updated = datetime.now().timestamp()

            if feature_name in self.feature_params:
                del self.feature_params[feature_name]

            logger.info(f"Removed feature: {feature_name}")

    def __repr__(self) -> str:
        """String representation"""
        return f"FeatureMetadata(features={len(self.feature_list)}, version={self.data_source_version})"

    def __str__(self) -> str:
        """String representation"""
        return self.__repr__()
