
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "buy my viagra $$$ theyre free",
    candidate_labels=["Spam", "Not spam"],
)

print(result)

from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

print(result)