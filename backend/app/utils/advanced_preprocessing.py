import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import matplotlib.pyplot as plt
from ..utils.logger import Logger

logger = Logger()

class AdvancedPreprocessor:
    def __init__(self):
        logger.info("Initializing AdvancedPreprocessor")
        self.scaler = StandardScaler()
        self.anomaly_detectors = {}
        self.cleaning_history = []
    
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
    
    def clean_data(self, data: pd.DataFrame,
                  target_column: str,
                  features: List[str] = None,
                  methods: List[str] = None) -> pd.DataFrame:
        """Clean data using ensemble anomaly detection and interpolation."""
        logger.info(f"Starting data cleaning for target column: {target_column}")
        try:
            if features is None:
                features = [target_column]
                logger.info(f"Using default features: {features}")
            
            # Store original data
            original_data = data.copy()
            
            # Detect anomalies
            logger.info("Running ensemble anomaly detection")
            anomaly_labels, method_results = self.ensemble_anomaly_detection(
                data, features, methods
            )
            
            # Create cleaned dataset
            cleaned_data = data.copy()
            anomaly_mask = anomaly_labels == -1
            
            # Store cleaning information
            cleaning_info = {
                'timestamp': pd.Timestamp.now(),
                'total_points': len(data),
                'anomalies_detected': np.sum(anomaly_mask),
                'method_results': method_results,
                'features_used': features
            }
            self.cleaning_history.append(cleaning_info)
            logger.info(f"Cleaning info stored: {cleaning_info}")
            
            # Replace anomalies with NaN and interpolate
            logger.info("Interpolating detected anomalies")
            cleaned_data.loc[anomaly_mask, target_column] = np.nan
            cleaned_data[target_column] = cleaned_data[target_column].interpolate(method='time')
            
            # Fill remaining NaN values at edges
            cleaned_data[target_column] = cleaned_data[target_column].fillna(method='ffill')
            cleaned_data[target_column] = cleaned_data[target_column].fillna(method='bfill')
            
            logger.info("Data cleaning completed successfully")
            return cleaned_data
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
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
    
    def detect_anomalies(self, df: pd.DataFrame, target_column: str, 
                        method: str = 'rolling_stats',
                        window: int = 24,
                        threshold: float = 3.0,
                        contamination: float = 0.1) -> pd.DataFrame:
        """Detect and flag anomalies in the data using various methods.
        
        Args:
            df: Input DataFrame
            target_column: Column to analyze for anomalies
            method: Method to use for anomaly detection ('rolling_stats', 'isolation_forest', 'one_class_svm', 'lof')
            window: Window size for rolling statistics
            threshold: Number of standard deviations for rolling stats method
            contamination: Expected proportion of anomalies in the data
            
        Returns:
            DataFrame with added 'is_anomaly' column
        """
        logger.info(f"Detecting anomalies using {method} method")
        try:
            df = df.copy()
            
            if method == 'rolling_stats':
                logger.info(f"Using rolling statistics with window={window} and threshold={threshold}")
                rolling_mean = df[target_column].rolling(window=window).mean()
                rolling_std = df[target_column].rolling(window=window).std()
                
                df['is_anomaly'] = (
                    (df[target_column] - rolling_mean).abs() > (threshold * rolling_std)
                ).astype(int)
                
            elif method == 'isolation_forest':
                logger.info(f"Using Isolation Forest with contamination={contamination}")
                labels = self.detect_anomalies_isolation_forest(
                    df, [target_column], contamination
                )
                df['is_anomaly'] = (labels == -1).astype(int)
                
            elif method == 'one_class_svm':
                logger.info(f"Using One-Class SVM with nu={contamination}")
                labels = self.detect_anomalies_one_class_svm(
                    df, [target_column], contamination
                )
                df['is_anomaly'] = (labels == -1).astype(int)
                
            elif method == 'lof':
                logger.info(f"Using Local Outlier Factor with contamination={contamination}")
                labels = self.detect_anomalies_lof(
                    df, [target_column], contamination
                )
                df['is_anomaly'] = (labels == -1).astype(int)
                
            else:
                logger.error(f"Unsupported anomaly detection method: {method}")
                raise ValueError(f"Unsupported method: {method}")
            
            n_anomalies = df['is_anomaly'].sum()
            logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.2f}% of data)")
            return df
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            raise 