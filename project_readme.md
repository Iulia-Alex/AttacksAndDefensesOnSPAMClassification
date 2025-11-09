# Spam Detection with Adversarial Attacks and Defenses

## Project Overview

This project implements adversarial attacks and defense mechanisms for a spam detection classification task. The goal is to explore the vulnerabilities of machine learning models and improve their robustness against adversarial examples.

### Configuration
- **Application**: Spam Detection (Email/SMS classification)
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
│   ├── baseline_model.pth          # Trained baseline model (generated)
│   ├── adversarial_training_model.pth  # Defended model (generated)
│   └── defensive_distillation_model.pth # Defended model (generated)
├── attacks/
│   ├── pgd.py                      # PGD attack implementation
│   └── synonym_substitution.py     # Synonym substitution attack
├── defenses/
│   ├── adversarial_training.py     # Adversarial training defense
│   └── defensive_distillation.py   # Defensive distillation defense
├── utils/
│   ├── data_loader.py              # Data loading and preprocessing
│   └── evaluation.py               # Evaluation utilities
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
git clone <your-repo-url>
cd spam-detection-adversarial
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
text,label
"Congratulations! You've won $1000",spam
"Hey, are we still meeting tomorrow?",ham
...
```

**Dataset Recommendations**:
- [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)

## Running the Project

### Quick Start - Run Full Pipeline

```bash
python main.py
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
- **Adversarial Generation**: On-the-fly PGD attacks during training
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

## Expected Results

### Baseline Model
- Clean Accuracy: ~95-98%
- PGD Adversarial Accuracy: ~30-50%
- Synonym Adversarial Accuracy: ~60-75%

### Adversarial Training
- Clean Accuracy: ~92-95% (slight drop)
- PGD Adversarial Accuracy: ~70-85% (significant improvement)
- Better robustness to gradient-based attacks

### Defensive Distillation
- Clean Accuracy: ~93-96% (slight drop)
- PGD Adversarial Accuracy: ~60-75% (moderate improvement)
- Smoother decision boundaries

## Comparison with Official Implementations

To compare with official implementations, you can use:

1. **For PGD**: [Foolbox](https://github.com/bethgelab/foolbox) or [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
2. **For TextAttack**: [TextAttack library](https://github.com/QData/TextAttack) for text-based attacks

Example comparison:
```python
# Install: pip install foolbox
import foolbox as fb

# Wrap your model
fmodel = fb.PyTorchModel(model, bounds=(0, 1))

# Run official PGD
attack = fb.attacks.LinfPGD()
adversarials = attack(fmodel, images, labels, epsilons=[0.1])
```

## Visualization Examples

The project includes visualization of:

1. **Training curves**: Loss and accuracy over epochs
2. **Attack success rates**: Bar charts comparing different attacks
3. **Example adversarial texts**: Side-by-side comparison of original vs adversarial
4. **Robustness curves**: Accuracy vs perturbation budget (epsilon)

## Troubleshooting

### Issue: CUDA out of memory
- **Solution**: Reduce batch size in `train_baseline.py` (line 15)

### Issue: Synonym attack too slow
- **Solution**: Reduce `num_samples` in `evaluate_synonym_attack` or `max_substitutions`

### Issue: Poor baseline accuracy
- **Solution**: 
  - Increase training epochs
  - Check data quality and preprocessing
  - Adjust model hyperparameters (hidden_dim, num_layers)

### Issue: Models not improving with defenses
- **Solution**: 
  - Increase defense training epochs
  - Adjust mix_ratio (adversarial training) or temperature (distillation)
  - Ensure attack parameters match between evaluation and training

## Citation

If you use this code, please cite the original papers:

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

- **Team Members**: [Your Names]
- **Course**: [Course Name]
- **Date**: [Date]

## License

This project is for educational purposes as part of [Course Name].

## Acknowledgments

- Dataset: [Dataset Source]
- Framework: PyTorch
- NLP Tools: NLTK, WordNet

---

For questions or issues, please open an issue on GitHub or contact [your-email].