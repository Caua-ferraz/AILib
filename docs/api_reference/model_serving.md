# API Reference: Model Serving Tools

Deploying trained models for real-time inference is a common requirement in AI applications. **AILib** provides tools to serve models as APIs using **FastAPI**, enabling seamless integration into production environments. This section details how to set up and use the model serving functionalities.

## Overview

The `model_serving.py` module in AILib offers a simple interface to deploy trained models as RESTful APIs. It leverages FastAPI and Uvicorn to create high-performance serving endpoints.

## Setting Up Model Serving

### 1. Import Necessary Components

```python
from ailib import UnifiedModel
from ailib.model_serving import run_server
```

### 2. Load the Trained Model

Ensure you have a trained model saved previously.

```python
# Load a traditional ML model
ml_model = UnifiedModel.load('neural_network', "saved_traditional_ml_model.pkl")

# Load a fine-tuned LLM
llm_model = UnifiedModel.load('llm', "saved_fine_tuned_llm_model")
```

### 3. Configure the Serving Parameters

You can configure host and port as needed.

```python
# Define host and port
host = "0.0.0.0"  # Accessible from all IPs
port = 8000
```

### 4. Run the Server

Start the API server to serve predictions.

```python
# Serve the traditional ML model
run_server(ml_model, host=host, port=port, api_name="ml_predict")

# Serve the LLM model
run_server(llm_model, host=host, port=port, api_name="llm_generate")
```

## API Endpoints

### `/predict`

- **Method**: POST
- **Description**: Make predictions using the traditional ML model.
- **Payload**:
  - `data` (List[Any]): Input features for prediction.
  
- **Response**:
  - `prediction` (List[int]): Predicted labels.

- **Example Request**:

  ```bash
  curl -X POST "http://0.0.0.0:8000/predict" -H "Content-Type: application/json" -d '{"data": [[0.1, 0.2, 0.3, 0.4, 0.5]]}'
  ```

- **Example Response**:

  ```json
  {
      "prediction": [1]
  }
  ```

### `/generate`

- **Method**: POST
- **Description**: Generate text using the LLM model.
- **Payload**:
  - `prompt` (str): Input text prompt for generation.
  - `max_length` (int, optional): Maximum length of the generated text.
  
- **Response**:
  - `generated_text` (str): Generated text.

- **Example Request**:

  ```bash
  curl -X POST "http://0.0.0.0:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Artificial Intelligence is", "max_length": 50}'
  ```

- **Example Response**:

  ```json
  {
      "generated_text": "Artificial Intelligence is transforming the way we interact with technology..."
  }
  ```

## Running the Server

You can run the server using the provided `run_server` function. Here's an example script to serve both models:

```python
from ailib import UnifiedModel
from ailib.model_serving import run_server

# Load models
ml_model = UnifiedModel.load('neural_network', "saved_traditional_ml_model.pkl")
llm_model = UnifiedModel.load('llm', "saved_fine_tuned_llm_model")

# Run servers
import threading

# Serve ML model on port 8000
ml_thread = threading.Thread(target=run_server, args=(ml_model, "0.0.0.0", 8000, "predict"))
ml_thread.start()

# Serve LLM model on port 8001
llm_thread = threading.Thread(target=run_server, args=(llm_model, "0.0.0.0", 8001, "generate"))
llm_thread.start()
```

## Best Practices

- **Security**: Implement authentication and authorization mechanisms to protect your API endpoints.
- **Scalability**: Use load balancers and multiple server instances to handle high traffic and ensure reliability.
- **Monitoring**: Integrate monitoring tools to track API performance and usage metrics.
- **Error Handling**: Ensure robust error handling within the API to provide meaningful feedback to users.

## Advanced Serving Options

- **Batch Predictions**: Modify the API to handle batch prediction requests for improved efficiency.
- **Asynchronous Processing**: Utilize asynchronous endpoints in FastAPI to handle high-concurrency scenarios.
- **Deployment Automation**: Use containerization tools like Docker to deploy the API server consistently across environments.

## Example: Serving with Docker

1. **Create a `Dockerfile`**

   ```dockerfile
   FROM python:3.8-slim

   WORKDIR /app

   # Install dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY . .

   # Expose ports
   EXPOSE 8000 8001

   # Run the server
   CMD ["python", "serve_models.py"]
   ```

2. **Build and Run the Docker Container**

   ```bash
   docker build -t ailib-serving .
   docker run -d -p 8000:8000 -p 8001:8001 ailib-serving
   ```

## Next Steps

- Explore [Model Explainability](explainability.md) to interpret model predictions.
- Learn how to integrate [Logging](logging.md) for monitoring your API servers.
- Review [Configuration Management](configuration.md) to manage your serving settings effectively.
