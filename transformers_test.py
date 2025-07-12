# Test transformers library
try:
    from transformers import pipeline
    print("Transformers import successful!")

    # Test a simple text classification pipeline
    classifier = pipeline("sentiment-analysis")
    result = classifier("I love using transformers!")
    print("Sentiment analysis result:", result)
except ImportError as e:
    print("Transformers import failed:", e)
except Exception as e:
    print("Transformers test failed:", e)
