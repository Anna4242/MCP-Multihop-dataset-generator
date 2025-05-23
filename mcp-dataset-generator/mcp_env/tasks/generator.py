"""Very small task generator: random arithmetic within 0‑20."""
import random

def task_generator():
    while True:
        a, b = random.randint(0, 20), random.randint(0, 20)
        yield {
            "question": f"Compute {a} + {b}",
            "answer": a + b,
        }