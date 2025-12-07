# 🔊 SonicSentience: End-to-End Speech Emotion Recognition

> **"Data is not given; it is taken."** > Most audio projects start with a clean `.zip` file from Kaggle. This project starts with raw API queries, handling the chaos of real-world unstructured audio data.

## 🧐 The Mission
The goal is to build a Machine Learning model capable of detecting **10 distinct human emotions** from short audio bursts. 

Instead of using the sanitized *RAVDESS* or *TESS* datasets (where actors speak clearly in a studio), I engineered a pipeline to scrape, clean, and label **wild audio** from the Freesound API. This introduces real-world challenges like background noise, variable sample rates, and labeling ambiguity.

## 📂 The "Big 8+2" Emotion Classes
I focused on capturing high-intensity emotional "bursts" (0.5s – 3.0s).

| Context | Emotions | Complexity |
| :--- | :--- | :--- |
| **Positive** | `Happy` | Distinctive pitch changes (laughter). |
| **High Energy** | `Angry`, `Fear`, `Pain` | Hard to distinguish. Differentiated by harmonic tones vs. dissonance. |
| **Low Energy** | `Sad`, `Disgust`, `Boredom` | Often confused with silence or background noise. |
| **Reactive** | `Surprise`, `Confused` | Short duration, relies on sharp intakes of breath. |
| **Baseline** | `Neutral` | The control group. Essential to prevent the model from hallucinating. |
