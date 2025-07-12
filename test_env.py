import torch
import numpy as np
from transformers import pipeline

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test numpy
arr = np.random.rand(3, 3)
print("\nRandom numpy array:\n", arr)

# Test transformers
classifier = pipeline("sentiment-analysis")
result = classifier("I love using transformers!")
print("\nTransformers test:", result)
