import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import matplotlib.pyplot as plt
from ..utils.logger import Logger
from .preprocessor_tracker import PreprocessorTracker

logger = Logger()

class AdvancedPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the advanced preprocessor."""
        self.config = config
        self.tracker = PreprocessorTracker()
        self.pipeline_steps = []
        self.scaler = StandardScaler()
        self.anomaly_detectors = {}
        self.cleaning_history = []
    
    def detect_anomalies(self, 
                        data: pd.DataFrame,
                        target_column: str,
                        method: str = 'isolation_forest',
                        window: int = 24,
                        threshold: float = 3.0,
                        contamination: float = 0.1) -> pd.DataFrame:
        """Detect anomalies with tracking."""
        logger.info(f"Detecting anomalies using {method} method")
        
        try:
            # Start tracking
            self.tracker.start_tracking()
            self.tracker.log_preprocessing_config(self.config)
            
            if method == 'rolling_stats':
                anomalies = self._detect_rolling_stats_anomalies(
                    data, target_column, window, threshold
                )
            elif method == 'isolation_forest':
                anomalies = self._detect_isolation_forest_anomalies(
                    data, target_column, contamination
                )
            elif method == 'one_class_svm':
                anomalies = self._detect_svm_anomalies(
                    data, target_column, contamination
                )
            elif method == 'lof':
                anomalies = self._detect_lof_anomalies(
                    data, target_column, contamination
                )
            else:
                raise ValueError(f"Unsupported anomaly detection method: {method}")
            
            # Log anomaly detection stats
            n_anomalies = anomalies.sum()
            anomaly_percentage = (n_anomalies / len(data)) * 100
            
            self.tracker.log_anomaly_detection_stats(
                method=method,
                n_anomalies=n_anomalies,
                anomaly_percentage=anomaly_percentage,
                params={
                    "window": window,
                    "threshold": threshold,
                    "contamination": contamination
                }
            )
            
            # Add anomaly flag to data
            data['is_anomaly'] = anomalies
            
            self.pipeline_steps.append({
                "step": "detect_anomalies",
                "method": method,
                "n_anomalies": n_anomalies,
                "anomaly_percentage": anomaly_percentage
            })
            
            return data
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            raise
        finally:
            self.tracker.end_tracking()
    
    def clean_data(self, 
                  data: pd.DataFrame,
                  target_column: str,
                  features: List[str]) -> pd.DataFrame:
        """Clean data by handling anomalies with tracking."""
        logger.info("Cleaning data by handling anomalies")
        
        try:
            # Start tracking
            self.tracker.start_tracking()
            
            if 'is_anomaly' not in data.columns:
                raise ValueError("Anomalies must be detected before cleaning data")
            
            # Store original values for tracking
            original_stats = data[features].describe()
            
            # Handle anomalies
            method = self.config.get('anomaly_handling_method', 'interpolate')
            
            if method == 'interpolate':
                data.loc[data['is_anomaly'], features] = data[features].interpolate()
            elif method == 'mean':
                data.loc[data['is_anomaly'], features] = data[features].mean()
            elif method == 'median':
                data.loc[data['is_anomaly'], features] = data[features].median()
            elif method == 'remove':
                data = data[~data['is_anomaly']]
            
            # Log cleaning stats
            self.tracker.log_model_params({
                "anomaly_handling_method": method,
                "features_cleaned": features,
                "n_anomalies_handled": data['is_anomaly'].sum(),
                "original_stats": original_stats.to_dict(),
                "cleaned_stats": data[features].describe().to_dict()
            })
            
            self.pipeline_steps.append({
                "step": "clean_data",
                "method": method,
                "features_cleaned": features
            })
            
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
        finally:
            self.tracker.end_tracking()
    
    def _detect_rolling_stats_anomalies(self,
                                      data: pd.DataFrame,
                                      target_column: str,
                                      window: int,
                                      threshold: float) -> pd.Series:
        """Detect anomalies using rolling statistics."""
        rolling_mean = data[target_column].rolling(window=window).mean()
        rolling_std = data[target_column].rolling(window=window).std()
        
        z_scores = np.abs((data[target_column] - rolling_mean) / rolling_std)
        return z_scores > threshold
    
    def _detect_isolation_forest_anomalies(self,
                                         data: pd.DataFrame,
                                         target_column: str,
                                         contamination: float) -> pd.Series:
        """Detect anomalies using Isolation Forest."""
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(data[[target_column]])
        return anomalies == -1
    
    def _detect_svm_anomalies(self,
                            data: pd.DataFrame,
                            target_column: str,
                            contamination: float) -> pd.Series:
        """Detect anomalies using One-Class SVM."""
        svm = OneClassSVM(nu=contamination)
        anomalies = svm.fit_predict(data[[target_column]])
        return anomalies == -1
    
    def _detect_lof_anomalies(self,
                            data: pd.DataFrame,
                            target_column: str,
                            contamination: float) -> pd.Series:
        """Detect anomalies using Local Outlier Factor."""
        lof = LocalOutlierFactor(contamination=contamination)
        anomalies = lof.fit_predict(data[[target_column]])
        return anomalies == -1
    
    def detect_anomalies_isolation_forest(self, data: pd.DataFrame,
                                        features: List[str],
                                        contamination: float = 0.1) -> np.ndarray:
        """Detect anomalies using Isolation Forest."""
        logger.info(f"Detecting anomalies using Isolation Forest with contamination={contamination}")
        try:
            X = self.scaler.fit_transform(data[features])
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            self.anomaly_detectors['isolation_forest'] = iso_forest
            
            labels = iso_forest.fit_predict(X)
            n_anomalies = np.sum(labels == -1)
            logger.info(f"Detected {n_anomalies} anomalies using Isolation Forest")
            return labels
        except Exception as e:
            logger.error(f"Error in Isolation Forest anomaly detection: {str(e)}")
            raise
    
    def detect_anomalies_one_class_svm(self, data: pd.DataFrame,
                                      features: List[str],
                                      nu: float = 0.1) -> np.ndarray:
        """Detect anomalies using One-Class SVM."""
        logger.info(f"Detecting anomalies using One-Class SVM with nu={nu}")
        try:
            X = self.scaler.fit_transform(data[features])
            one_class_svm = OneClassSVM(nu=nu)
            self.anomaly_detectors['one_class_svm'] = one_class_svm
            
            labels = one_class_svm.fit_predict(X)
            n_anomalies = np.sum(labels == -1)
            logger.info(f"Detected {n_anomalies} anomalies using One-Class SVM")
            return labels
        except Exception as e:
            logger.error(f"Error in One-Class SVM anomaly detection: {str(e)}")
            raise
    
    def detect_anomalies_kmeans(self, data: pd.DataFrame,
                               features: List[str],
                               n_clusters: int = 3,
                               threshold: float = 2.0) -> np.ndarray:
        """Detect anomalies using K-means clustering."""
        logger.info(f"Detecting anomalies using K-means with {n_clusters} clusters and threshold={threshold}")
        try:
            X = self.scaler.fit_transform(data[features])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.anomaly_detectors['kmeans'] = kmeans
            
            # Fit K-means
            clusters = kmeans.fit_predict(X)
            
            # Calculate distances to cluster centers
            distances = np.min(
                [np.linalg.norm(X - center, axis=1) for center in kmeans.cluster_centers_],
                axis=0
            )
            
            # Mark points with distances > threshold * std as anomalies
            threshold_value = np.mean(distances) + threshold * np.std(distances)
            labels = np.where(distances > threshold_value, -1, 1)
            n_anomalies = np.sum(labels == -1)
            logger.info(f"Detected {n_anomalies} anomalies using K-means")
            return labels
        except Exception as e:
            logger.error(f"Error in K-means anomaly detection: {str(e)}")
            raise
    
    def detect_anomalies_dbscan(self, data: pd.DataFrame,
                               features: List[str],
                               eps: float = 0.5,
                               min_samples: int = 5) -> np.ndarray:
        """Detect anomalies using DBSCAN clustering."""
        logger.info(f"Detecting anomalies using DBSCAN with eps={eps} and min_samples={min_samples}")
        try:
            X = self.scaler.fit_transform(data[features])
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            self.anomaly_detectors['dbscan'] = dbscan
            
            # Fit DBSCAN
            clusters = dbscan.fit_predict(X)
            
            # Convert DBSCAN labels to anomaly indicators (-1 for anomalies, 1 for normal)
            labels = np.where(clusters == -1, -1, 1)
            n_anomalies = np.sum(labels == -1)
            logger.info(f"Detected {n_anomalies} anomalies using DBSCAN")
            return labels
        except Exception as e:
            logger.error(f"Error in DBSCAN anomaly detection: {str(e)}")
            raise
    
    def detect_anomalies_lof(self, data: pd.DataFrame,
                            features: List[str],
                            contamination: float = 0.1) -> np.ndarray:
        """Detect anomalies using Local Outlier Factor."""
        logger.info(f"Detecting anomalies using LOF with contamination={contamination}")
        try:
            X = self.scaler.fit_transform(data[features])
            lof = LocalOutlierFactor(contamination=contamination, novelty=True)
            self.anomaly_detectors['lof'] = lof
            
            labels = lof.fit_predict(X)
            n_anomalies = np.sum(labels == -1)
            logger.info(f"Detected {n_anomalies} anomalies using LOF")
            return labels
        except Exception as e:
            logger.error(f"Error in LOF anomaly detection: {str(e)}")
            raise
    
    def ensemble_anomaly_detection(self, data: pd.DataFrame,
                                 features: List[str],
                                 methods: List[str] = None,
                                 voting_threshold: float = 0.5) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Combine multiple anomaly detection methods using voting."""
        logger.info(f"Starting ensemble anomaly detection with voting threshold={voting_threshold}")
        try:
            if methods is None:
                methods = ['isolation_forest', 'one_class_svm', 'kmeans', 'dbscan', 'lof']
                logger.info(f"Using default methods: {methods}")
            
            results = {}
            for method in methods:
                logger.debug(f"Running {method} for ensemble detection")
                if method == 'isolation_forest':
                    results[method] = self.detect_anomalies_isolation_forest(data, features)
                elif method == 'one_class_svm':
                    results[method] = self.detect_anomalies_one_class_svm(data, features)
                elif method == 'kmeans':
                    results[method] = self.detect_anomalies_kmeans(data, features)
                elif method == 'dbscan':
                    results[method] = self.detect_anomalies_dbscan(data, features)
                elif method == 'lof':
                    results[method] = self.detect_anomalies_lof(data, features)
            
            # Combine results using voting
            votes = np.stack(list(results.values()))
            ensemble_prediction = np.mean(votes == -1, axis=0) >= voting_threshold
            final_labels = np.where(ensemble_prediction, -1, 1)
            
            n_anomalies = np.sum(final_labels == -1)
            logger.info(f"Ensemble detection completed. Found {n_anomalies} anomalies")
            return final_labels, results
        except Exception as e:
            logger.error(f"Error in ensemble anomaly detection: {str(e)}")
            raise
    
    def get_cleaning_report(self) -> pd.DataFrame:
        """Generate report of data cleaning history."""
        logger.info("Generating cleaning report")
        try:
            report = pd.DataFrame(self.cleaning_history)
            if not report.empty:
                report['anomaly_percentage'] = (
                    report['anomalies_detected'] / report['total_points'] * 100
                )
                logger.info(f"Report generated with {len(report)} cleaning records")
            else:
                logger.warning("No cleaning history available")
            return report
        except Exception as e:
            logger.error(f"Error generating cleaning report: {str(e)}")
            raise
    
    def plot_anomaly_comparison(self, data: pd.DataFrame,
                              target_column: str,
                              features: List[str] = None,
                              save_path: str = None):
        """Plot comparison of different anomaly detection methods."""
        logger.info(f"Generating anomaly comparison plot for {target_column}")
        try:
            if features is None:
                features = [target_column]
                logger.info(f"Using default features: {features}")
            
            # Get results from all methods
            _, method_results = self.ensemble_anomaly_detection(data, features)
            
            # Create subplot for each method
            n_methods = len(method_results)
            fig, axes = plt.subplots(n_methods, 1, figsize=(15, 4*n_methods))
            
            for (method, labels), ax in zip(method_results.items(), axes):
                logger.debug(f"Plotting results for {method}")
                # Plot original data
                ax.plot(data.index, data[target_column], label='Original', alpha=0.5)
                
                # Plot anomalies
                anomaly_mask = labels == -1
                ax.scatter(data.index[anomaly_mask],
                          data.loc[anomaly_mask, target_column],
                          color='red', label='Anomalies')
                
                ax.set_title(f'Anomalies detected by {method}')
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                logger.info(f"Saving plot to {save_path}")
                plt.savefig(save_path)
            
            logger.info("Anomaly comparison plot generated successfully")
            return fig
        except Exception as e:
            logger.error(f"Error generating anomaly comparison plot: {str(e)}")
            raise 