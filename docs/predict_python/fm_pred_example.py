import ctypes
import logging
import threading
import numpy as np

class FMPredictor:
    def __init__(self, lib_path, model_config_path):
        """
        Initialize FM predictor with library path and model configuration.
        
        Args:
            lib_path: Path to fm_pred.so library
            model_config_path: Path to model configuration file
        """
        # Load FM library
        try:
            self.fm_lib = ctypes.CDLL(lib_path)
            self._setup_function_signatures()
        except OSError as e:
            raise RuntimeError(f"Failed to load FM prediction library '{lib_path}': {e}")

        # Initialize model
        self.fm_model = None
        self.local = threading.local()  # Thread-local storage for prediction instances
        self.model_config_path = model_config_path
        self._load_model()

    def _setup_function_signatures(self):
        """Setup C function signatures for type checking"""
        # fmModelCreate
        self.fm_lib.fmModelCreate.argtypes = [ctypes.c_char_p]
        self.fm_lib.fmModelCreate.restype = ctypes.c_void_p

        # fmModelRelease
        self.fm_lib.fmModelRelease.argtypes = [ctypes.c_void_p]
        self.fm_lib.fmModelRelease.restype = None

        # fmPredictInstanceCreate
        self.fm_lib.fmPredictInstanceCreate.argtypes = [ctypes.c_void_p]
        self.fm_lib.fmPredictInstanceCreate.restype = ctypes.c_void_p

        # fmPredictInstanceRelease
        self.fm_lib.fmPredictInstanceRelease.argtypes = [ctypes.c_void_p]
        self.fm_lib.fmPredictInstanceRelease.restype = None

        # fmPredictBatch
        self.fm_lib.fmPredictBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]
        self.fm_lib.fmPredictBatch.restype = ctypes.c_int

    def _load_model(self):
        """Load FM model using configuration"""
        if self.fm_model:
            self.fm_lib.fmModelRelease(self.fm_model)
        
        self.fm_model = self.fm_lib.fmModelCreate(self.model_config_path.encode('utf-8'))
        if not self.fm_model:
            raise RuntimeError("Failed to create FM model instance")

    def _get_predict_instance(self):
        """Get or create thread-local prediction instance"""
        if not hasattr(self.local, 'fm_instance'):
            self.local.fm_instance = self.fm_lib.fmPredictInstanceCreate(self.fm_model)
            if not self.local.fm_instance:
                raise RuntimeError("Failed to create FM prediction instance")
        return self.local.fm_instance

    def predict(self, feature_strings, debug=False):
        """
        Perform batch prediction on feature strings.
        
        Args:
            feature_strings: List of feature strings in the format required by the model
            debug: Boolean flag for debug output
            
        Returns:
            numpy array of prediction scores
        """
        # Convert feature strings to C-compatible format
        features_array = (ctypes.c_char_p * len(feature_strings))()
        for i, feat in enumerate(feature_strings):
            features_array[i] = feat.encode('utf-8')

        # Prepare output array
        scores = (ctypes.c_double * len(feature_strings))()

        # Get prediction instance
        instance = self._get_predict_instance()

        # Perform prediction
        ret = self.fm_lib.fmPredictBatch(
            instance,
            features_array,
            len(feature_strings),
            scores,
            int(debug)
        )

        if ret != 0:
            raise RuntimeError(f"FM prediction failed with error code: {ret}")

        return np.array(scores)

    def __del__(self):
        """Cleanup resources"""
        if self.fm_model:
            self.fm_lib.fmModelRelease(self.fm_model)


def main():
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize predictor
    predictor = FMPredictor(
        lib_path='exFM/lib/fm_pred.so',
        model_config_path='path/to/model_config.json'
    )

    # Example feature strings (format depends on your model configuration)
    feature_strings = [
        "user_id:123,item_id:456,category:electronics",
        "user_id:123,item_id:789,category:books"
    ]

    try:
        # Make predictions
        scores = predictor.predict(feature_strings, debug=True)
        print("Prediction scores:", scores)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main() 