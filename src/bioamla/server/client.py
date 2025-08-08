# Example client code
import requests
import json

BASE_URL = "http://localhost:8000"

def test_sentiment(text):
    response = requests.post(
        f"{BASE_URL}/sentiment",
        json={"text": text}
    )
    return response.json()

def test_ner(text):
    response = requests.post(
        f"{BASE_URL}/ner",
        json={"text": text}
    )
    return response.json()

def test_qa(question, context):
    response = requests.post(
        f"{BASE_URL}/qa",
        json={"question": question, "context": context}
    )
    return response.json()

def test_batch_sentiment(texts):
    response = requests.post(
        f"{BASE_URL}/sentiment/batch",
        json={"texts": texts}
    )
    return response.json()

# Example usage
if __name__ == "__main__":
    # Test sentiment analysis
    result = test_sentiment("I love this API!")
    print("Sentiment:", result)
    
    # Test NER
    result = test_ner("Apple Inc. was founded by Steve Jobs.")
    print("NER:", result)
    
    # Test QA
    result = test_qa(
        "Who founded Apple?", 
        "Apple Inc. was founded by Steve Jobs in 1976."
    )
    print("QA:", result)
    
    # Test batch processing
    result = test_batch_sentiment([
        "Great product!",
        "Terrible service.",
        "It's okay."
    ])
    print("Batch sentiment:", result)