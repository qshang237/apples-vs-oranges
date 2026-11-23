# 🍎 vs 🍊 Apples vs Oranges — A Classifier That Finally Settles the Debate

> "You're comparing apples and oranges!" — Everyone, forever.
> "Actually, I can tell them apart." — This model.

## What is this?

A deep learning image classifier that does exactly what humans have been arguing you *can't* do: compare apples and oranges. And get it right.

Built with [fastai](https://www.fast.ai/) and PyTorch, fine-tuned on a ResNet18 pretrained model. Trained on images scraped from the web. After 3 epochs and under 30 seconds of training, the error rate on the held-out validation images was **0**.

## How it works

1. Requested up to 200 apple images and up to 200 orange images from DuckDuckGo image search, and removed the files that did not open
2. Held out 20% of the images at random for validation
3. Fine-tuned a pretrained ResNet18 model (transfer learning)
4. 3 epochs later — the model knows its fruits

## Results

| Metric | Value |
|--------|-------|
| Validation error rate | 0.000 after epoch 3 (0.042 after epoch 1) |
| Images | up to 200 per class requested; 80% for training, 20% for validation |
| Training Time | ~30 seconds |
| Epochs | 3 |

The validation set is small, so an error rate of 0 means that no held-out image was misclassified. It does not mean that the model is always correct.

The two example predictions at the end of the notebook use images taken at random from the downloaded folders, so the model saw some of them during training. They show how to use the model. They are not a test.

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

*Turns out you CAN compare apples and oranges. You just need a small set of images and a GPU.*
