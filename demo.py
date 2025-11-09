from utils.visualization import (
    plot_training_history, 
    plot_attack_comparison, 
    plot_epsilon_robustness, 
    display_adversarial_examples,
    plot_defense_comparison
)

def run_comprehensive_demo():
    """Run a comprehensive demo of the entire project"""
    import os
    import torch
    from utils.data_loader import load_spam_dataset
    from models.baseline_model import SpamClassifier
    
    print("="*80)
    print("COMPREHENSIVE PROJECT DEMO")
    print("="*80)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load data
    print("\n[1] Loading dataset...")
    data = load_spam_dataset('data/spam.csv')
    print(f"[DONE] Dataset loaded: {len(data['X_train'])} train, {len(data['X_test'])} test samples")
    
    # Load baseline model
    print("\n[2] Loading baseline model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('models/baseline_model.pth', map_location=device)
    
    model = SpamClassifier(checkpoint['vocab_size']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("[DONE] Model loaded")
    
    # Plot training history
    print("\n[3] Plotting training history...")
    plot_training_history(checkpoint['history'])
    
    # Evaluate attacks
    print("\n[4] Evaluating attacks...")
    from evaluate_attacks import evaluate_pgd_attack, evaluate_synonym_attack
    
    pgd_results = evaluate_pgd_attack(model, data, epsilon=0.1)
    synonym_results = evaluate_synonym_attack(model, data, num_samples=50)
    
    # Plot attack comparison
    attack_results = {
        'PGD (ε=0.1)': pgd_results,
        'Synonym Sub': synonym_results
    }
    plot_attack_comparison(attack_results)
    
    # Plot epsilon robustness
    print("\n[5] Analyzing robustness vs epsilon...")
    plot_epsilon_robustness(model, data)
    
    # Display adversarial examples
    print("\n[6] Displaying adversarial examples...")
    display_adversarial_examples(model, data, num_examples=3)
    
    # Load and compare defended models (if available)
    print("\n[7] Comparing defense mechanisms...")
    try:
        adv_model = SpamClassifier(checkpoint['vocab_size']).to(device)
        adv_model.load_state_dict(torch.load('models/adversarial_training_model.pth', map_location=device))
        adv_results = evaluate_pgd_attack(adv_model, data, epsilon=0.1)
        
        dist_model = SpamClassifier(checkpoint['vocab_size']).to(device)
        dist_model.load_state_dict(torch.load('models/defensive_distillation_model.pth', map_location=device))
        dist_results = evaluate_pgd_attack(dist_model, data, epsilon=0.1)
        
        defense_results = {
            'Adversarial Training': adv_results,
            'Defensive Distillation': dist_results
        }
        
        plot_defense_comparison(pgd_results, defense_results)
        
    except FileNotFoundError:
        print("Defended models not found. Run main.py to train defended models.")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("All visualizations saved to 'results/' directory")
    print("="*80)


if __name__ == '__main__':
    run_comprehensive_demo()