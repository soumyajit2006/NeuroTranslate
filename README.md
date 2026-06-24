# NeuroTranslate

# A Multilingual Neural Machine Translation System using Transformer Models

## Project Overview

This project implements a **Neural Machine Translation (NMT) System** in Python using transformer-based models from the Hugging Face Model Hub. The system translates text between multiple languages while preserving meaning, grammar, and fluency. It provides an interactive translation interface, supports configurable translation parameters, evaluates translation quality using BLEU scores, and compares the performance of two state-of-the-art transformer models.

### Models Used

1. **MarianMT** (Helsinki-NLP/opus-mt-en-fr)

   * Lightweight and fast translation model
   * Optimized for English → French translation
   * Lower memory and computational requirements

2. **M2M100** (facebook/m2m100_418M)

   * Many-to-Many multilingual translation model
   * Supports 100+ languages
   * Capable of translating directly between language pairs without relying on English as an intermediate language


## Objectives

* Develop an interactive translation application.
* Support multilingual translation across multiple languages.
* Allow users to configure translation generation parameters.
* Compare transformer-based translation models.
* Evaluate translation quality using BLEU scores.
* Analyze model behavior across different text domains.


## Features

### 1. Interactive Translation Interface

The system allows users to:

* Enter source text
* Select source language
* Select target language
* Obtain translated output in real time

### 2. Language Validation

The application performs:

* Empty input validation
* Invalid language validation
* Source-target language consistency checks
* Maximum input length restriction

### 3. Configurable Translation Parameters

Users can customize:

* Maximum Output Length
* Beam Search Width
* Length Penalty
* Temperature
* Top-k Sampling
* Top-p (Nucleus) Sampling

### 4. Model Performance Comparison

The following metrics are compared:

* Translation Quality
* Translation Speed
* Memory Usage
* CPU Utilization
* GPU Utilization
* Long Sentence Handling Capability
* Multilingual Support

### 5. BLEU Score Evaluation

Translation quality is measured using the BLEU (Bilingual Evaluation Understudy) metric by comparing generated translations against reference translations.

### 6. Experimental Analysis

The system evaluates performance on:

* News Article Snippets
* Technical/Scientific Text
* Conversational Text


## System Architecture

Input Text
↓
Language Validation
↓
Translation Parameters
↓
Translation Models
├── MarianMT
└── M2M100
↓
Translated Output
↓
BLEU Evaluation
↓
Performance Analysis


## Installation

### Clone Repository

```bash
git clone <repository-url>
cd neural-machine-translation
```

### Install Dependencies

```bash
pip install transformers
pip install sentencepiece
pip install sacrebleu
pip install accelerate
pip install psutil
pip install pandas
pip install matplotlib
pip install gradio
```

Or:

```bash
pip install transformers sentencepiece sacrebleu accelerate psutil pandas matplotlib gradio
```


## Google Colab Setup

Run:

```python
!pip install -q transformers sentencepiece sacrebleu accelerate psutil pandas matplotlib gradio
```

Enable GPU:

Runtime → Change Runtime Type → GPU


## Supported Languages

| Code | Language |
| ---- | -------- |
| en   | English  |
| fr   | French   |
| de   | German   |
| es   | Spanish  |
| hi   | Hindi    |


## Translation Parameters

| Parameter      | Description                                   |
| -------------- | --------------------------------------------- |
| max_length     | Maximum length of generated translation       |
| num_beams      | Beam search width                             |
| length_penalty | Controls translation length                   |
| temperature    | Controls randomness and diversity             |
| top_k          | Restricts token selection to top-k candidates |
| top_p          | Uses nucleus sampling for generation          |


## Experimental Dataset

### News Domain

"The government announced a new economic policy to support industrial growth."

### Technical Domain

"Machine learning algorithms improve performance using experience and training data."

### Conversational Domain

"Hello my friend, how are you doing today?"


## BLEU Score Evaluation

BLEU score is computed as:

BLEU = Precision of n-gram matches between candidate and reference translations.

Higher BLEU scores indicate better translation quality.


## Performance Metrics

The following metrics are recorded:

* Translation Time (seconds)
* Memory Usage (MB)
* CPU Utilization (%)
* GPU Memory Usage (MB)
* BLEU Score
  

## Results and Observations

### MarianMT

Advantages:

* Fast inference
* Lightweight architecture
* Low memory consumption
* Accurate English → French translations

Limitations:

* Language-pair specific
* Limited multilingual support

### M2M100

Advantages:

* Supports 100+ languages
* Strong multilingual performance
* Better handling of long sentences
* More flexible language translation

Limitations:

* Larger model size
* Higher memory consumption
* Slower inference speed

## Comparative Analysis

| Metric                 | MarianMT | M2M100    |
| ---------------------- | -------- | --------- |
| Translation Quality    | High     | High      |
| Translation Speed      | Faster   | Slower    |
| Memory Usage           | Lower    | Higher    |
| CPU Utilization        | Lower    | Higher    |
| GPU Utilization        | Lower    | Higher    |
| Long Sentence Handling | Moderate | Better    |
| Multilingual Support   | Limited  | Excellent |

## Applications

This Neural Machine Translation System can be applied in:

* Multilingual Chatbots
* Cross-Language Communication
* Content Localization
* Educational Platforms
* International Customer Support
* Document Translation
* Research and Academic Work

## Future Enhancements

* Integration of larger multilingual models such as mBART-50 and NLLB-200
* Real-time speech-to-text translation
* Translation confidence estimation
* Fine-tuning on domain-specific datasets
* Web deployment using Streamlit or Flask
* Mobile application integration

## Conclusion

This project demonstrates the implementation of a complete Neural Machine Translation System using modern transformer architectures. By comparing MarianMT and M2M100, the project highlights the trade-offs between translation quality, computational efficiency, multilingual capability, and resource consumption. Experimental results, BLEU score evaluation, and performance analysis provide a comprehensive understanding of transformer-based machine translation systems and their real-world applications.
