import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import ripser
import persim
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class TopologicalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Feature extractor using persistent homology for topological data analysis.
    
    This class generates features from analyzing persistence diagrams
    obtained through persistent homology on time series windows.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the topological feature extractor.
        
        Args:
            config: Configuration dictionary containing TDA parameters
        """
        self.config = config
        self.tda_config = config.get('tda', {})
        self.max_homology_dim = self.tda_config.get('max_homology_dim', 1)
        self.window_sizes = self.tda_config.get('window_sizes', [10, 20, 30])
        self.step_size = self.tda_config.get('step_size', 5)
        self.max_persistence = self.tda_config.get('max_persistence', np.inf)
        self.persistence_features = self.tda_config.get('persistence_features', 
                                                       ['sum', 'max', 'mean', 'std', 'entropy'])
        self.diagram_statistics = []
        self.feature_names = []
        
    def _time_delay_embedding(self, series: np.ndarray, dimension: int, delay: int) -> np.ndarray:
        """Create a time-delay embedding representation of the series.
        
        Args:
            series: One-dimensional time series
            dimension: Dimension of the embedding space
            delay: Time delay between points
            
        Returns:
            np.ndarray: Points in the embedding space
        """
        if len(series) <= (dimension - 1) * delay:
            raise ValueError(f"Series too short for embedding with dimension={dimension}, delay={delay}")
            
        num_points = len(series) - (dimension - 1) * delay
        embedding = np.zeros((num_points, dimension))
        
        for i in range(dimension):
            embedding[:, i] = series[i * delay:i * delay + num_points]
            
        return embedding
    
    def _compute_persistence_diagram(self, window: np.ndarray) -> Dict[str, Any]:
        """Compute the persistence diagram for a given window.
        
        Args:
            window: Window data
            
        Returns:
            Dict: Persistence diagram and associated metrics
        """
        # For one-dimensional data, use time-delay embedding
        if window.ndim == 1 or window.shape[1] == 1:
            flat_window = window.flatten()
            dimension = self.tda_config.get('embedding_dimension', 3)
            delay = self.tda_config.get('embedding_delay', 1)
            points = self._time_delay_embedding(flat_window, dimension, delay)
        else:
            points = window
            
        # Compute persistent homology
        diagrams = ripser.ripser(points, maxdim=self.max_homology_dim, 
                                do_cocycles=False, thresh=self.max_persistence)['dgms']
        
        # Calculate statistics for each dimension
        stats = {}
        for dim in range(len(diagrams)):
            dim_diagram = diagrams[dim]
            if len(dim_diagram) > 0:
                # Filter out infinity points
                finite_points = dim_diagram[dim_diagram[:, 1] != np.inf]
                persistence = finite_points[:, 1] - finite_points[:, 0] if len(finite_points) > 0 else np.array([])
                
                dim_stats = {
                    'diagram': dim_diagram,
                    'num_features': len(dim_diagram),
                    'persistence_sum': np.sum(persistence) if len(persistence) > 0 else 0,
                    'persistence_max': np.max(persistence) if len(persistence) > 0 else 0,
                    'persistence_mean': np.mean(persistence) if len(persistence) > 0 else 0,
                    'persistence_std': np.std(persistence) if len(persistence) > 0 else 0,
                    'persistence_entropy': self._entropy(persistence) if len(persistence) > 0 else 0,
                    'persistence_values': persistence
                }
                stats[f'dim_{dim}'] = dim_stats
            else:
                stats[f'dim_{dim}'] = {
                    'diagram': dim_diagram,
                    'num_features': 0,
                    'persistence_sum': 0,
                    'persistence_max': 0,
                    'persistence_mean': 0,
                    'persistence_std': 0,
                    'persistence_entropy': 0,
                    'persistence_values': np.array([])
                }
                
        return {
            'diagrams': diagrams,
            'stats': stats
        }
    
    def _entropy(self, values: np.ndarray) -> float:
        """Calculate the entropy of a set of persistence values.
        
        Args:
            values: Persistence values
            
        Returns:
            float: Entropy
        """
        if len(values) <= 1:
            return 0.0
            
        # Normalize values
        total = np.sum(values)
        if total == 0:
            return 0.0
            
        probabilities = values / total
        # Avoid log(0)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _extract_window_features(self, window: np.ndarray, window_size: int) -> Dict[str, float]:
        """Extract features from a window.
        
        Args:
            window: Window data
            window_size: Size of the window
            
        Returns:
            Dict: Extracted features
        """
        persistence_results = self._compute_persistence_diagram(window)
        features = {}
        
        for dim in range(self.max_homology_dim + 1):
            dim_key = f'dim_{dim}'
            if dim_key in persistence_results['stats']:
                dim_stats = persistence_results['stats'][dim_key]
                prefix = f'tda_win{window_size}_dim{dim}_'
                
                # Add number of features
                features[f'{prefix}count'] = dim_stats['num_features']
                
                # Add statistics on persistence values
                for stat in self.persistence_features:
                    if stat in dim_stats:
                        features[f'{prefix}{stat}'] = dim_stats[f'persistence_{stat}']
        
        return features
    
    def _get_sliding_windows(self, data: np.ndarray, window_size: int, step_size: int) -> List[np.ndarray]:
        """Generate sliding windows from the data.
        
        Args:
            data: Input data
            window_size: Size of the window
            step_size: Step between windows
            
        Returns:
            List: List of windows
        """
        n_samples = len(data)
        windows = []
        
        for start in range(0, n_samples - window_size + 1, step_size):
            windows.append(data[start:start + window_size])
            
        return windows
    
    def fit(self, X: pd.DataFrame, y=None) -> 'TopologicalFeatureExtractor':
        """Fit the feature extractor (no action needed).
        
        Args:
            X: Input data
            y: Target (not used)
            
        Returns:
            self
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by extracting topological features.
        
        Args:
            X: Input data
            
        Returns:
            pd.DataFrame: Data with added topological features
        """
        if not self.config.get('add_tda_features', False):
            logger.info("TDA feature extraction disabled")
            return X
            
        logger.info("Extracting topological features")
        
        # Get the target column (usually 'value')
        target_col = self.config.get('data', {}).get('target_column', 'value')
        if target_col not in X.columns:
            logger.warning(f"Target column {target_col} not found")
            return X
            
        # Extract the time series
        series = X[target_col].values
        
        # Initialize the feature dataframe
        tda_features = pd.DataFrame(index=X.index)
        
        # For each window size, calculate features
        for window_size in self.window_sizes:
            if len(series) < window_size:
                logger.warning(f"Series too short for window size {window_size}")
                continue
                
            # Generate sliding windows
            windows = self._get_sliding_windows(series, window_size, self.step_size)
            
            # Extract features for each window
            window_features_list = []
            for i, window in enumerate(windows):
                features = self._extract_window_features(window, window_size)
                # Add window index
                features['window_idx'] = i
                window_features_list.append(features)
            
            # Aggregate features across all windows
            agg_features = {}
            if window_features_list:
                # Find all unique keys
                all_keys = set()
                for features in window_features_list:
                    all_keys.update(features.keys())
                
                all_keys.discard('window_idx')  # Don't aggregate index
                
                # Aggregate each feature
                for key in all_keys:
                    values = [f.get(key, 0) for f in window_features_list]
                    agg_features[f'{key}_mean'] = np.mean(values)
                    agg_features[f'{key}_std'] = np.std(values)
                    agg_features[f'{key}_max'] = np.max(values)
                    agg_features[f'{key}_min'] = np.min(values)
            
            # Add to features DataFrame
            for key, value in agg_features.items():
                tda_features[key] = value
        
        # Store feature names
        self.feature_names = tda_features.columns.tolist()
        
        # Merge with original data
        result = pd.concat([X, tda_features], axis=1)
        
        logger.info(f"Added {len(self.feature_names)} topological features")
        return result
    
    def visualize_persistence_diagram(self, data: np.ndarray, save_path: Optional[str] = None) -> None:
        """Visualize the persistence diagram for the data.
        
        Args:
            data: Input data
            save_path: Path to save the diagram (optional)
        """
        persistence_results = self._compute_persistence_diagram(data)
        diagrams = persistence_results['diagrams']
        
        fig, axes = plt.subplots(1, len(diagrams), figsize=(5*len(diagrams), 5))
        if len(diagrams) == 1:
            axes = [axes]
            
        for i, diagram in enumerate(diagrams):
            persim.plot_diagrams(diagram, ax=axes[i])
            axes[i].set_title(f"Dimension {i}")
            
        plt.suptitle("Persistence Diagrams")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Diagram saved to {save_path}")
        else:
            plt.show()