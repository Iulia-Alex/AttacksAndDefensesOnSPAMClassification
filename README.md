# Spam Detection with Adversarial Attacks and Defenses

## Project Overview

This project implements adversarial attacks and defense mechanisms for a spam detection classification task. The goal is to explore the vulnerabilities of machine learning models and improve their robustness against adversarial attacks.

### Configuration
- **Application**: Spam Detection (SMS classification)
- **Attacks**: 
  1. Projected Gradient Descent (PGD)
  2. Synonym Substitution Attack
- **Defenses**: 
  1. Adversarial Training
  2. Defensive Distillation

## Project Structure

```
spam-detection-adversarial/
├── data/
│   └── spam.csv                    # Dataset file
├── models/
│   ├── baseline_model.py           # LSTM-based spam classifier
│   ├── baseline_model.pth          # Trained baseline model
│   ├── adversarial_training_model.pth  # Defended model
│   └── defensive_distillation_model.pth # Defended model
├── attacks/
│   ├── pgd.py                      # PGD attack implementation
│   └── synonym_substitution.py     # Synonym substitution attack
├── defenses/
│   ├── adversarial_training.py     # Adversarial training defense
│   └── defensive_distillation.py   # Defensive distillation defense
├── utils/
│   ├── data_loader.py              # Data loading and preprocessing
│   └── visualization.py            # Evaluation graphics
│   └── visualization_extra.py      # More evaluation graphics
├── train_baseline.py               # Script to train baseline model
├── evaluate_attacks.py             # Script to evaluate attacks
├── evaluate_defenses.py            # Script to evaluate defenses
├── main.py                         # Main pipeline script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Iulia-Alex/AttacksAndDefensesOnSPAMClassification.git
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data (for Synonym Substitution)

```python
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 5. Prepare Dataset

Place your spam detection dataset in the `data/` directory as `spam.csv`. The dataset should have the following format:

```csv
label text
spam  Congratulations! You've won $1000
ham Hey, are we still meeting tomorrow?
...
```

**Dataset Recommendations**:

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip -O data/spam.zip
unzip data/spam.zip -d data/
```

## Running the Project

### Quick Start - Run Full Pipeline

```bash
python3 main.py
```

This will:
1. Load and preprocess the dataset
2. Train the baseline model
3. Evaluate both attacks on the baseline
4. Train defended models (Adversarial Training & Defensive Distillation)
5. Evaluate defenses against attacks
6. Generate a comparison report

### Individual Components

#### Train Baseline Model Only

```python
from utils.data_loader import load_spam_dataset
from train_baseline import train_baseline_model

data = load_spam_dataset('data/spam.csv')
model, history = train_baseline_model(data, epochs=20)
```

#### Evaluate PGD Attack

```python
from evaluate_attacks import evaluate_pgd_attack

results = evaluate_pgd_attack(model, data, epsilon=0.1, alpha=0.01, num_iter=40)
```

#### Evaluate Synonym Substitution Attack

```python
from evaluate_attacks import evaluate_synonym_attack

results = evaluate_synonym_attack(model, data, max_substitutions=5)
```

#### Train with Adversarial Training

```python
from evaluate_defenses import train_with_adversarial_training

defended_model = train_with_adversarial_training(data, baseline_model, epochs=15)
```

#### Train with Defensive Distillation

```python
from evaluate_defenses import train_with_defensive_distillation

defended_model = train_with_defensive_distillation(data, baseline_model, epochs=15, temperature=20)
```

## Implementation Details

### 1. Baseline Model (LSTM Classifier)

- **Architecture**: Bidirectional LSTM with 2 layers
- **Embedding Dimension**: 128
- **Hidden Dimension**: 256
- **Dropout**: 0.5
- **Output**: Binary classification (spam/ham)

### 2. PGD Attack

The PGD attack operates in the embedding space:

- **Method**: Iterative gradient-based attack
- **Attack Surface**: Word embeddings
- **Parameters**:
  - Epsilon (ε): Maximum perturbation budget (L∞ norm) = 0.1
  - Alpha (α): Step size = 0.01
  - Iterations: 40
- **Implementation**: Custom implementation from scratch

The attack generates adversarial perturbations by:
1. Computing gradients of the loss with respect to embeddings
2. Taking small steps in the direction that maximizes loss
3. Projecting perturbations back into the epsilon ball

### 3. Synonym Substitution Attack

A black-box attack that manipulates input text:

- **Method**: Replace words with synonyms from WordNet
- **Maximum Substitutions**: 5 words per text
- **Strategy**: Greedy search for synonyms that flip predictions
- **Implementation**: Custom implementation using NLTK's WordNet

The attack:
1. Identifies words in the vocabulary
2. Finds synonyms using WordNet
3. Tests each substitution to see if it fools the model
4. Stops when prediction flips or max substitutions reached

### 4. Adversarial Training Defense

- **Method**: Train with mixture of clean and adversarial examples
- **Mix Ratio**: 50% adversarial, 50% clean examples per batch
- **Adversarial Generation**: PGD attacks during training
- **Training Epochs**: 15 (fine-tuning from baseline)
- **Learning Rate**: 0.0001

### 5. Defensive Distillation Defense

- **Method**: Knowledge distillation with temperature scaling
- **Temperature**: 20 (softens probability distributions)
- **Teacher Model**: Pre-trained baseline model
- **Student Model**: Fresh model with same architecture
- **Loss**: Combination of KL divergence (soft targets) and cross-entropy (hard labels)
- **Alpha**: 0.7 (weight for distillation loss)

## Evaluation Metrics

For each model and attack combination, we measure:

1. **Clean Accuracy**: Accuracy on original, unperturbed test data
2. **Adversarial Accuracy**: Accuracy on adversarially perturbed test data
3. **Attack Success Rate**: Percentage of correctly classified examples that were fooled
4. **Robustness Improvement**: Increase in adversarial accuracy compared to baseline

## Citation

Original papers:

**PGD Attack**:
```
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017).
Towards deep learning models resistant to adversarial attacks.
arXiv preprint arXiv:1706.06083.
```

**Defensive Distillation**:
```
Papernot, N., McDaniel, P., Wu, X., Jha, S., & Swami, A. (2016).
Distillation as a defense to adversarial perturbations against deep neural networks.
In 2016 IEEE Symposium on Security and Privacy (SP).
```

## Team Information

- **Team Members**: Diana-Roxana Bratu, Iulia-Alexandra Orvas
- **Course**: Artificial Intelligence 3
- **Date**: 12th of November 2025

## License

This project is for educational purposes as part of Artificial Intelligence 3 course.