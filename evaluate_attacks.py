import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def evaluate_pgd_attack(model, data, epsilon=0.1, alpha=0.01, num_iter=40):
    """Evaluate PGD attack on test set"""
    from attacks.pgd import PGDAttack
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Prepare test data
    test_dataset = TensorDataset(
        torch.LongTensor(data['X_test']),
        torch.LongTensor(data['y_test'])
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize attack
    pgd = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    print(f"Evaluating PGD attack (eps={epsilon}, alpha={alpha}, iter={num_iter})...")
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Clean accuracy
        with torch.no_grad():
            clean_outputs = model(inputs)
            _, clean_pred = clean_outputs.max(1)
            clean_correct += clean_pred.eq(labels).sum().item()
        
        # Generate adversarial examples
        adv_embeddings = pgd.attack(inputs, labels)
        
        # Adversarial accuracy (forward through model without embedding layer)
        with torch.no_grad():
            lstm_out, (hidden, cell) = model.lstm(adv_embeddings)
            hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            hidden_concat = model.dropout(hidden_concat)
            hidden_concat = torch.nn.functional.relu(model.fc1(hidden_concat))
            hidden_concat = model.dropout(hidden_concat)
            adv_outputs = model.fc2(hidden_concat)
            _, adv_pred = adv_outputs.max(1)
            adv_correct += adv_pred.eq(labels).sum().item()
        
        total += labels.size(0)
    
    clean_acc = 100. * clean_correct / total
    adv_acc = 100. * adv_correct / total
    attack_success_rate = 100. * (clean_correct - adv_correct) / clean_correct
    
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"Attack Success Rate: {attack_success_rate:.2f}%")
    
    return {
        'clean_acc': clean_acc,
        'adv_acc': adv_acc,
        'attack_success_rate': attack_success_rate
    }

def evaluate_synonym_attack(model, data, max_substitutions=15, num_samples=300):
    """
    Improved evaluation with better sampling and detailed tracking
    """
    from attacks.synonym_substitution import SynonymSubstitutionAttack
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    attack = SynonymSubstitutionAttack(model, data['preprocessor'], max_substitutions)
    
    # Use more test samples for better estimate
    num_samples = min(num_samples, len(data['X_test']))
    test_indices = np.random.choice(len(data['X_test']), num_samples, replace=False)
    
    clean_correct = 0
    adv_correct = 0
    successful_attacks = 0
    attack_attempts = 0
    
    print(f"Evaluating Synonym Substitution attack (max_subs={max_substitutions}, samples={num_samples})...")
    print("This may take a few minutes...")
    
    for idx_num, idx in enumerate(test_indices):
        if (idx_num + 1) % 50 == 0:
            print(f"  Processed {idx_num + 1}/{num_samples} samples...")
        
        # Get original text
        seq = data['X_test'][idx]
        words = [data['preprocessor'].idx2word.get(i, '<UNK>') for i in seq if i != 0]
        original_text = ' '.join(words)
        label = data['y_test'][idx]
        
        # Skip very short texts
        if len(words) < 3:
            continue
        
        # Original prediction
        original_seq = torch.tensor([seq]).to(device)
        with torch.no_grad():
            original_pred = model(original_seq).argmax(dim=1).item()
        
        # Only attack correctly classified examples
        if original_pred == label:
            clean_correct += 1
            attack_attempts += 1
            
            # Generate adversarial text
            adv_text = attack.attack_text(original_text, label, use_importance=True)
            
            # Adversarial prediction
            adv_seq = torch.tensor([data['preprocessor'].text_to_sequence(adv_text)]).to(device)
            with torch.no_grad():
                adv_pred = model(adv_seq).argmax(dim=1).item()
            
            if adv_pred == label:
                adv_correct += 1
            else:
                successful_attacks += 1
    
    if attack_attempts == 0:
        print("Warning: No valid attack attempts!")
        return {'clean_acc': 0, 'adv_acc': 0, 'attack_success_rate': 0}
    
    clean_acc = 100. * clean_correct / len(test_indices)
    adv_acc = 100. * adv_correct / len(test_indices)
    attack_success_rate = 100. * successful_attacks / attack_attempts
    
    print(f"\nResults:")
    print(f"  Valid attack attempts: {attack_attempts}/{len(test_indices)}")
    print(f"  Clean Accuracy: {clean_acc:.2f}%")
    print(f"  Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"  Attack Success Rate: {attack_success_rate:.2f}%")
    
    return {
        'clean_acc': clean_acc,
        'adv_acc': adv_acc,
        'attack_success_rate': attack_success_rate
    }

def diagnose_pgd_attack(model, data, epsilon=0.1):
    """
    Diagnose why PGD might be too effective
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from attacks.pgd import PGDAttack
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Test on small batch
    test_data = torch.LongTensor(data['X_test'][:32]).to(device)
    test_labels = torch.LongTensor(data['y_test'][:32]).to(device)
    
    # Get original embeddings
    with torch.no_grad():
        orig_embeddings = model.embedding(test_data)
        print(f"Original embedding shape: {orig_embeddings.shape}")
        print(f"Original embedding range: [{orig_embeddings.min():.4f}, {orig_embeddings.max():.4f}]")
        print(f"Original embedding mean: {orig_embeddings.mean():.4f}")
    
    # Run PGD attack
    pgd = PGDAttack(model, epsilon=epsilon, alpha=epsilon/10, num_iter=40)
    adv_embeddings = pgd.attack(test_data, test_labels)
    
    print(f"\nAdversarial embedding shape: {adv_embeddings.shape}")
    print(f"Adversarial embedding range: [{adv_embeddings.min():.4f}, {adv_embeddings.max():.4f}]")
    print(f"Adversarial embedding mean: {adv_embeddings.mean():.4f}")
    
    # Check perturbation
    perturbation = (adv_embeddings - orig_embeddings).abs()
    print(f"\nPerturbation magnitude: {perturbation.mean():.4f}")
    print(f"Max perturbation: {perturbation.max():.4f}")
    print(f"Perturbation within epsilon? {perturbation.max() <= epsilon}")
    
    # Check predictions
    with torch.no_grad():
        # Clean predictions
        clean_outputs = model(test_data)
        clean_preds = clean_outputs.argmax(dim=1)
        clean_acc = (clean_preds == test_labels).float().mean().item()
        
        # Adversarial predictions
        lstm_out, (hidden, cell) = model.lstm(adv_embeddings)
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        adv_outputs = model.fc2(model.dropout(model.bn(model.dropout(torch.nn.functional.relu(model.fc1(model.dropout(hidden_concat)))))))
        adv_preds = adv_outputs.argmax(dim=1)
        adv_acc = (adv_preds == test_labels).float().mean().item()
    
    print(f"\nClean accuracy on sample: {clean_acc*100:.2f}%")
    print(f"Adversarial accuracy on sample: {adv_acc*100:.2f}%")
    print(f"Examples flipped: {(clean_preds != adv_preds).sum().item()}/{len(test_labels)}")