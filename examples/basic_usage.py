# D:\AiProject\examples\basic_usage.py

import numpy as np
from sklearn.datasets import load_iris
from ailib import AIModel, preprocess_data, split_data, train_model, evaluate_model, LLM

def demonstrate_traditional_ml():
    print("Demonstrating Traditional Machine Learning:")
    # Load a sample dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Preprocess the data
    X_processed, y_processed = preprocess_data(X, y)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)

    # Create and train a model
    model = AIModel('neural_network', {'hidden_layer_sizes': (10, 5), 'max_iter': 1000})
    trained_model = train_model(model, X_train, y_train)

    # Evaluate the model
    evaluation_results = evaluate_model(model, X_test, y_test)
    print("Model Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")

def demonstrate_llm():
    print("\nDemonstrating Large Language Model:")
    # Initialize the LLM
    llm = LLM()  # This will use the default 'gpt2' model

    # Generate text
    prompt = "Artificial Intelligence is"
    generated_text = llm.generate_text(prompt, max_length=50)
    print(f"Generated text: {generated_text[0]}")

    # Tokenize text
    text = "Hello, how are you?"
    tokens = llm.tokenize(text)
    print(f"Tokens: {tokens}")

    # Get token IDs
    token_ids = llm.get_token_ids(text)
    print(f"Token IDs: {token_ids}")

if __name__ == "__main__":
    demonstrate_traditional_ml()
    demonstrate_llm()