import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_history(history, save_path='results/training_history.png'):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.show()


def plot_attack_comparison(results_dict, save_path='results/attack_comparison.png'):
    """
    Plot comparison of different attacks
    
    Args:
        results_dict: Dictionary with format:
            {
                'Baseline': {'clean_acc': 95, 'adv_acc': 45},
                'PGD': {'clean_acc': 95, 'adv_acc': 40},
                ...
            }
    """
    models = list(results_dict.keys())
    clean_accs = [results_dict[m]['clean_acc'] for m in models]
    adv_accs = [results_dict[m]['adv_acc'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, clean_accs, width, label='Clean Accuracy', 
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, adv_accs, width, label='Adversarial Accuracy', 
                   color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model / Attack', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Attack Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attack comparison plot saved to {save_path}")
    plt.show()


def plot_defense_comparison(baseline_results, defense_results, 
                           save_path='results/defense_comparison.png'):
    """
    Plot comparison of baseline vs defended models
    
    Args:
        baseline_results: Dict with 'clean_acc' and 'adv_acc'
        defense_results: Dict of defense names to results
    """
    models = ['Baseline'] + list(defense_results.keys())
    clean_accs = [baseline_results['clean_acc']] + \
                 [defense_results[d]['clean_acc'] for d in defense_results]
    adv_accs = [baseline_results['adv_acc']] + \
               [defense_results[d]['adv_acc'] for d in defense_results]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, clean_accs, width, label='Clean Accuracy', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, adv_accs, width, label='Adversarial Accuracy', 
                   color='#9b59b6', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Defense Mechanism Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Defense comparison plot saved to {save_path}")
    plt.show()


def plot_epsilon_robustness(model, data, epsilons=[0.01, 0.05, 0.1, 0.15, 0.2],
                           save_path='results/epsilon_robustness.png'):
    """Plot adversarial accuracy vs epsilon (perturbation budget)"""
    from attacks.pgd import PGDAttack
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    test_dataset = TensorDataset(
        torch.LongTensor(data['X_test'][:500]),  # Use subset for speed
        torch.LongTensor(data['y_test'][:500])
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    accuracies = []
    
    for eps in epsilons:
        pgd = PGDAttack(model, epsilon=eps, alpha=eps/10, num_iter=40)
        correct = 0
        total = 0
        
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            adv_embeddings = pgd.attack(inputs, labels)
            
            with torch.no_grad():
                lstm_out, (hidden, cell) = model.lstm(adv_embeddings)
                hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
                hidden_concat = model.dropout(hidden_concat)
                hidden_concat = torch.nn.functional.relu(model.fc1(hidden_concat))
                hidden_concat = model.dropout(hidden_concat)
                outputs = model.fc2(hidden_concat)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        acc = 100. * correct / total
        accuracies.append(acc)
        print(f"Epsilon: {eps:.3f}, Accuracy: {acc:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, accuracies, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    plt.xlabel('Epsilon (Perturbation Budget)', fontsize=12, fontweight='bold')
    plt.ylabel('Adversarial Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Robustness vs Perturbation Budget', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, (eps, acc) in enumerate(zip(epsilons, accuracies)):
        plt.text(eps, acc + 2, f'{acc:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Epsilon robustness plot saved to {save_path}")
    plt.show()


def display_adversarial_examples(model, data, num_examples=5):
    """Display original texts and their adversarial versions"""
    from attacks.synonym_substitution import SynonymSubstitutionAttack
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    attack = SynonymSubstitutionAttack(model, data['preprocessor'], max_substitutions=5)
    
    # Select random test samples
    indices = np.random.choice(len(data['X_test']), num_examples, replace=False)
    
    print("\n" + "="*80)
    print("ADVERSARIAL EXAMPLES - SYNONYM SUBSTITUTION ATTACK")
    print("="*80)
    
    for i, idx in enumerate(indices):
        seq = data['X_test'][idx]
        label = data['y_test'][idx]
        label_name = 'SPAM' if label == 1 else 'HAM'
        
        # Reconstruct original text
        words = [data['preprocessor'].idx2word.get(i, '<UNK>') for i in seq if i != 0]
        original_text = ' '.join(words)
        
        # Get original prediction
        original_seq = torch.tensor([seq]).to(device)
        with torch.no_grad():
            original_pred = model(original_seq).argmax(dim=1).item()
            original_probs = torch.softmax(model(original_seq), dim=1)[0]
        
        original_pred_name = 'SPAM' if original_pred == 1 else 'HAM'
        
        # Generate adversarial example
        adv_text = attack.attack_text(original_text, label)
        
        # Get adversarial prediction
        adv_seq = torch.tensor([data['preprocessor'].text_to_sequence(adv_text)]).to(device)
        with torch.no_grad():
            adv_pred = model(adv_seq).argmax(dim=1).item()
            adv_probs = torch.softmax(model(adv_seq), dim=1)[0]
        
        adv_pred_name = 'SPAM' if adv_pred == 1 else 'HAM'
        
        print(f"\n--- Example {i+1} ---")
        print(f"True Label: {label_name}")
        print(f"\nOriginal Text:")
        print(f"  \"{original_text[:100]}...\"")
        print(f"  Prediction: {original_pred_name} (confidence: {original_probs[original_pred]:.2%})")
        print(f"\nAdversarial Text:")
        print(f"  \"{adv_text[:100]}...\"")
        print(f"  Prediction: {adv_pred_name} (confidence: {adv_probs[adv_pred]:.2%})")
        
        if original_pred == label and adv_pred != label:
            print(f"  [DONE] ATTACK SUCCESSFUL!")
        else:
            print(f"  [FAILED]  Attack failed")
        print("-" * 80)