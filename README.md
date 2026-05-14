# 🍎 vs 🍊 Apples vs Oranges — A Classifier That Finally Settles the Debate

> "You're comparing apples and oranges!" — Everyone, forever.
> "Actually, I can tell them apart." — This model.

## What is this?

A deep learning image classifier that does exactly what humans have been arguing you *can't* do: compare apples and oranges. And get it right.

Built with [fastai](https://www.fast.ai/) and PyTorch, fine-tuned on a ResNet18 pretrained model. Trained on 200 images scraped from the web. Achieved **92%+ accuracy** in under 30 seconds of training. 

## How it works

1. Scraped ~200 apple images and ~200 orange images using DuckDuckGo search
2. Fine-tuned a pretrained ResNet18 model (transfer learning)
3. 3 epochs later — the model knows its fruits

## Results

| Metric | Value |
|--------|-------|
| Error Rate | 0.000 (100% accuracy) |
| Training Images | ~200 |
| Training Time | ~30 seconds |
| Epochs | 3 |

## Usage

```python
from fastai.vision.all import *

learn = load_learner('apples_oranges.pkl')
pred, _, probs = learn.predict(PILImage.create('your_fruit.jpg'))
print(f"This is a: {pred}")
print(f"Probability apple: {probs[0]:.4f}")
print(f"Probability orange: {probs[1]:.4f}")
```

## Files

- `apples_vs_oranges.ipynb` — Full training notebook
- `apples_oranges.pkl` — Trained model (ready to use)

## Tech Stack

- Python
- fastai / PyTorch
- ResNet18 (pretrained on ImageNet)

---

*Turns out you CAN compare apples and oranges. You just need 200 images and a GPU.*
