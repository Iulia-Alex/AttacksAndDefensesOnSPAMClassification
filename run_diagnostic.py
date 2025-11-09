import torch
import numpy as np
from utils.data_loader import load_spam_dataset
from models.baseline_model import SpamClassifier

def quick_diagnostic():
    print("="*80)
    print("QUICK DIAGNOSTIC - CHECKING ATTACK IMPLEMENTATIONS")
    print("="*80)
    
    # Load data and model
    print("\n[1] Loading data and model...")
    data = load_spam_dataset('data/spam.csv')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('models/baseline_model.pth', map_location=device)
    
    model = SpamClassifier(checkpoint['vocab_size']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("[DONE] Loaded successfully")
    
    # Test 1: Check baseline accuracy
    print("\n[2] Testing baseline model accuracy...")
    test_data = torch.LongTensor(data['X_test'][:100]).to(device)
    test_labels = torch.LongTensor(data['y_test'][:100]).to(device)
    
    with torch.no_grad():
        outputs = model(test_data)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == test_labels).float().mean().item()
    
    print(f"[DONE] Baseline accuracy on 100 samples: {accuracy*100:.2f}%")
    
    # Test 2: Check embedding space
    print("\n[3] Analyzing embedding space...")
    with torch.no_grad():
        embeddings = model.embedding(test_data)
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        print(f"  Embedding mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
    
    # Test 3: Test PGD attack on one example
    print("\n[4] Testing PGD attack on single example...")
    from attacks.pgd import PGDAttack
    
    single_data = test_data[:1]
    single_label = test_labels[:1]
    
    # Get original prediction
    with torch.no_grad():
        orig_output = model(single_data)
        orig_pred = orig_output.argmax(dim=1).item()
        orig_conf = torch.softmax(orig_output, dim=1)[0]
    
    print(f"  Original prediction: {orig_pred} (confidence: {orig_conf[orig_pred]:.4f})")
    
    # Run PGD
    pgd = PGDAttack(model, epsilon=0.1, alpha=0.01, num_iter=40)
    adv_embeddings = pgd.attack(single_data, single_label)
    
    # Get adversarial prediction
    with torch.no_grad():
        lstm_out, (hidden, cell) = model.lstm(adv_embeddings)
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden_concat = model.dropout(hidden_concat)
        hidden_concat = torch.nn.functional.relu(model.fc1(hidden_concat))
        hidden_concat = model.bn(hidden_concat)
        hidden_concat = model.dropout(hidden_concat)
        adv_output = model.fc2(hidden_concat)
        adv_pred = adv_output.argmax(dim=1).item()
        adv_conf = torch.softmax(adv_output, dim=1)[0]
    
    print(f"  Adversarial prediction: {adv_pred} (confidence: {adv_conf[adv_pred]:.4f})")
    print(f"  Attack {'SUCCESSFUL' if orig_pred != adv_pred else 'FAILED'}")
    
    # Check perturbation magnitude
    with torch.no_grad():
        orig_emb = model.embedding(single_data)
        perturbation = (adv_embeddings - orig_emb).abs()
        print(f"  Perturbation: mean={perturbation.mean():.4f}, max={perturbation.max():.4f}")
    
    # Test 4: Test synonym attack on one example
    print("\n[5] Testing Synonym Substitution attack...")
    from attacks.synonym_substitution import SynonymSubstitutionAttack
    
    # Get a correctly classified example
    correct_idx = None
    for i in range(100):
        seq = data['X_test'][i]
        label = data['y_test'][i]
        with torch.no_grad():
            pred = model(torch.tensor([seq]).to(device)).argmax(dim=1).item()
        if pred == label:
            correct_idx = i
            break
    
    if correct_idx is None:
        print("  [FAILED]  Could not find correctly classified example!")
    else:
        seq = data['X_test'][correct_idx]
        label = data['y_test'][correct_idx]
        words = [data['preprocessor'].idx2word.get(i, '') for i in seq if i != 0]
        original_text = ' '.join(words)
        
        print(f"  Original text: '{original_text[:80]}...'")
        print(f"  True label: {label}")
        
        # Get original prediction
        with torch.no_grad():
            orig_pred = model(torch.tensor([seq]).to(device)).argmax(dim=1).item()
        print(f"  Original prediction: {orig_pred}")
        
        # Run synonym attack
        attack = SynonymSubstitutionAttack(model, data['preprocessor'], max_substitutions=15)
        adv_text = attack.attack_text(original_text, label, use_importance=True)
        
        print(f"  Adversarial text: '{adv_text[:80]}...'")
        
        # Get adversarial prediction
        adv_seq = torch.tensor([data['preprocessor'].text_to_sequence(adv_text)]).to(device)
        with torch.no_grad():
            adv_pred = model(adv_seq).argmax(dim=1).item()
        
        print(f"  Adversarial prediction: {adv_pred}")
        print(f"  Attack {'SUCCESSFUL' if orig_pred != adv_pred else 'FAILED'}")
        
        # Count word changes
        orig_words = set(original_text.split())
        adv_words = set(adv_text.split())
        changed = len(orig_words.symmetric_difference(adv_words))
        print(f"  Words changed: {changed}")
    
    # Test 5: Quick attack statistics
    print("\n[6] Quick attack statistics on 50 samples...")
    
    # PGD stats
    pgd_success = 0
    for i in range(50):
        data_point = torch.LongTensor([data['X_test'][i]]).to(device)
        label = torch.LongTensor([data['y_test'][i]]).to(device)
        
        with torch.no_grad():
            orig_pred = model(data_point).argmax(dim=1).item()
        
        if orig_pred == label.item():
            adv_emb = pgd.attack(data_point, label)
            
            with torch.no_grad():
                lstm_out, (hidden, cell) = model.lstm(adv_emb)
                hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
                hidden_concat = model.dropout(hidden_concat)
                hidden_concat = torch.nn.functional.relu(model.fc1(hidden_concat))
                hidden_concat = model.bn(hidden_concat)
                hidden_concat = model.dropout(hidden_concat)
                adv_output = model.fc2(hidden_concat)
                adv_pred = adv_output.argmax(dim=1).item()
            
            if adv_pred != label.item():
                pgd_success += 1
    
    print(f"  PGD attack success rate: {pgd_success}/50 = {pgd_success*2}%")
    
    # Synonym stats
    synonym_success = 0
    synonym_attempts = 0
    for i in range(50):
        seq = data['X_test'][i]
        label = data['y_test'][i]
        words = [data['preprocessor'].idx2word.get(i, '') for i in seq if i != 0]
        
        if len(words) < 3:
            continue
        
        original_text = ' '.join(words)
        
        with torch.no_grad():
            orig_pred = model(torch.tensor([seq]).to(device)).argmax(dim=1).item()
        
        if orig_pred == label:
            synonym_attempts += 1
            adv_text = attack.attack_text(original_text, label, use_importance=True)
            adv_seq = torch.tensor([data['preprocessor'].text_to_sequence(adv_text)]).to(device)
            
            with torch.no_grad():
                adv_pred = model(adv_seq).argmax(dim=1).item()
            
            if adv_pred != label:
                synonym_success += 1
    
    if synonym_attempts > 0:
        print(f"  Synonym attack success rate: {synonym_success}/{synonym_attempts} = {synonym_success*100//synonym_attempts}%")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nRecommendations:")
    
    if pgd_success > 40:
        print("  [DONE] PGD attack is working well (very effective)")
    elif pgd_success > 25:
        print("  ~ PGD attack is moderately effective")
    else:
        print("  [FAILED]  PGD attack seems too weak")
    
    if synonym_success > 10:
        print("  [DONE] Synonym attack is working (at least somewhat effective)")
    elif synonym_success > 5:
        print("  ~ Synonym attack is marginally effective")
    else:
        print("  [FAILED]  Synonym attack is too weak - needs improvement")
        print("    Try: Increase max_substitutions to 20-25")
        print("    Try: Use less strict synonym filtering")
        print("    Try: More aggressive importance calculation")

if __name__ == '__main__':
    quick_diagnostic()