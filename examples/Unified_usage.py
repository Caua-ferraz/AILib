# D:\AiProject\examples\Unified_usage.py

import numpy as np
from sklearn.datasets import load_iris
from ailib import UnifiedModel, preprocess_data, split_data

def demonstrate_unified_model():
    print("Demonstrating Unified Model:")
    
    # Traditional ML example
    print("\nTraditional ML with Neural Network:")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_processed, y_processed = preprocess_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)

    ml_model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5), max_iter=1000)
    ml_model.train(X_train, y_train)
    evaluation_results = ml_model.evaluate(X_test, y_test)
    print("Model Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")

    # LLM example
    print("\nLLM Text Generation:")
    
    # Using default GPT-2 model
    llm_model = UnifiedModel('llm')
    prompt = "Artificial Intelligence is"
    generated_text = llm_model.predict(prompt, max_length=100, temperature=0.7)
    print(f"Generated text (GPT-2): {generated_text[0]}")

    # Using a different model (e.g., GPT-2 medium)
    llm_model_medium = UnifiedModel('llm', model_name="gpt2-medium")
    generated_text_medium = llm_model_medium.predict(prompt, max_length=100, temperature=0.7)
    print(f"Generated text (GPT-2 medium): {generated_text_medium[0]}")

    # Demonstrating different generation parameters
    generated_text_creative = llm_model.predict(prompt, max_length=100, temperature=1.5, top_k=50)
    print(f"Generated text (more creative): {generated_text_creative[0]}")

    generated_text_focused = llm_model.predict(prompt, max_length=100, temperature=0.3, top_p=0.9)
    print(f"Generated text (more focused): {generated_text_focused[0]}")

    # Tokenization example
    text = "Hello, how are you?"
    tokens = llm_model.tokenize(text)
    print(f"Tokens: {tokens}")

if __name__ == "__main__":
    demonstrate_unified_model()