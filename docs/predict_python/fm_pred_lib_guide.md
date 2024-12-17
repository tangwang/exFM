# FM Prediction Library (fm_pred.so) Python Guide

## Overview
The FM Prediction Library (fm_pred.so) is a C-based library that provides factorization machine model prediction capabilities. This guide explains how to use the library in Python applications.

## Library Functions

The library exposes the following main functions:

1. `fmModelCreate`: Creates a new FM model instance
   - Input: Model prediction config path (string)
   - Returns: Model handle (void pointer)

2. `fmModelRelease`: Releases a FM model instance
   - Input: Model handle
   - Returns: None

3. `fmPredictInstanceCreate`: Creates a prediction instance for a specific model
   - Input: Model handle
   - Returns: Prediction instance handle (void pointer)

4. `fmPredictInstanceRelease`: Releases a prediction instance
   - Input: Prediction instance handle
   - Returns: None

5. `fmPredictBatch`: Performs batch prediction
   - Inputs:
     - Prediction instance handle
     - Array of feature strings
     - Number of instances
     - Output scores array
     - Debug flag
   - Returns: Status code (0 for success)

## Setup Requirements

1. The fm_pred.so library must be accessible in your system
2. Python ctypes library for interfacing with the C library
3. Proper model configuration file

## Thread Safety

The library is designed to be thread-safe:
- One FM model can be shared across multiple threads
- Each thread should create its own prediction instance
- Use threading.local() for thread-local storage of prediction instances

## Error Handling

The library implements error checking at multiple levels:
- Library loading errors
- Model creation errors
- Prediction instance creation errors
- Batch prediction errors

## Configuration

The model requires a configuration file (typically in JSON format) that specifies:
- Model parameters
- Feature configurations
- Other prediction-related settings

## Best Practices

1. Always release resources properly
2. Create one prediction instance per thread
3. Use batch prediction for better performance
4. Implement proper error handling
5. Log important operations and errors

See the example code in `fm_pred_example.py` for a practical implementation. 