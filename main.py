import torch
import os
import numpy as np

def main():
    """Simplified working version"""
    print("=" * 80)
    print("SPAM DETECTION ADVERSARIAL PROJECT - WORKING VERSION")
    print("=" * 80)
    
    os.makedirs('results', exist_ok=True)
    
    # 1. Load data
    print("\n[1] Loading dataset...")
    from utils.data_loader import load_spam_dataset
    data = load_spam_dataset('data/spam.csv')
    print(f"[DONE] Loaded: {len(data['X_train'])} train, {len(data['X_test'])} test samples")
    
    # 2. Train baseline
    print("\n[2] Training baseline model...")
    from train_baseline import train_baseline_model
    baseline_model, history = train_baseline_model(
        data, epochs=30, batch_size=32, lr=0.001, 
        weight_decay=1e-4, patience=7
    )
    
    # 3. Evaluate attacks on baseline
    print("\n[3] Evaluating attacks on baseline...")
    from evaluate_attacks import evaluate_pgd_attack, evaluate_synonym_attack
    
    print("\n--- PGD Attack ---")
    pgd_01 = evaluate_pgd_attack(baseline_model, data, epsilon=0.1, alpha=0.01, num_iter=40)
    
    print("\n--- Synonym Attack ---")
    # Use the AGGRESSIVE version with max_subs=20
    synonym_results = evaluate_synonym_attack(baseline_model, data, 
                                             max_substitutions=20, num_samples=200)
    
    # 4. Train defenses
    print("\n[4] Training defended models...")
    from evaluate_defenses import train_with_adversarial_training, train_with_defensive_distillation
    
    print("\n--- Adversarial Training ---")
    adv_trained_model = train_with_adversarial_training(data, baseline_model, epochs=15)
    
    print("\n--- Defensive Distillation ---")
    distilled_model = train_with_defensive_distillation(data, baseline_model, 
                                                        epochs=20, temperature=20)
    
    # 5. Evaluate defenses
    print("\n[5] Evaluating defenses...")
    
    print("\n--- PGD on Adversarial Training ---")
    adv_train_pgd = evaluate_pgd_attack(adv_trained_model, data, epsilon=0.1)
    
    print("\n--- PGD on Defensive Distillation ---")
    distill_pgd = evaluate_pgd_attack(distilled_model, data, epsilon=0.1)
    
    print("\n--- Synonym on Adversarial Training ---")
    adv_train_syn = evaluate_synonym_attack(adv_trained_model, data, 
                                           max_substitutions=20, num_samples=100)
    
    print("\n--- Synonym on Defensive Distillation ---")
    distill_syn = evaluate_synonym_attack(distilled_model, data, 
                                         max_substitutions=20, num_samples=100)
    
    # 6. Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n>>> PGD ATTACK (epsilon=0.1) <<<")
    print(f"{'Model':<25} {'Clean Acc':<12} {'Adv Acc':<12} {'Improvement':<12}")
    print("-" * 60)
    print(f"{'Baseline':<25} {pgd_01['clean_acc']:>10.2f}% {pgd_01['adv_acc']:>10.2f}% {'---':>12}")
    
    adv_train_imp = adv_train_pgd['adv_acc'] - pgd_01['adv_acc']
    print(f"{'Adversarial Training':<25} {adv_train_pgd['clean_acc']:>10.2f}% "
          f"{adv_train_pgd['adv_acc']:>10.2f}% {f'+{adv_train_imp:.2f}%':>12}")
    
    distill_imp = distill_pgd['adv_acc'] - pgd_01['adv_acc']
    print(f"{'Defensive Distillation':<25} {distill_pgd['clean_acc']:>10.2f}% "
          f"{distill_pgd['adv_acc']:>10.2f}% {f'+{distill_imp:.2f}%':>12}")
    
    print("\n>>> SYNONYM SUBSTITUTION ATTACK <<<")
    print(f"{'Model':<25} {'Clean Acc':<12} {'Adv Acc':<12} {'Attack SR':<12}")
    print("-" * 60)
    print(f"{'Baseline':<25} {synonym_results['clean_acc']:>10.2f}% "
          f"{synonym_results['adv_acc']:>10.2f}% {synonym_results['attack_success_rate']:>10.2f}%")
    print(f"{'Adversarial Training':<25} {adv_train_syn['clean_acc']:>10.2f}% "
          f"{adv_train_syn['adv_acc']:>10.2f}% {adv_train_syn['attack_success_rate']:>10.2f}%")
    print(f"{'Defensive Distillation':<25} {distill_syn['clean_acc']:>10.2f}% "
          f"{distill_syn['adv_acc']:>10.2f}% {distill_syn['attack_success_rate']:>10.2f}%")
    
    # Save results
    with open('results/final_results.txt', 'w') as f:
        f.write("FINAL RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"PGD Attack (eps=0.1):\n")
        f.write(f"  Baseline: {pgd_01['adv_acc']:.2f}%\n")
        f.write(f"  Adv Training: {adv_train_pgd['adv_acc']:.2f}% (+{adv_train_imp:.2f}%)\n")
        f.write(f"  Distillation: {distill_pgd['adv_acc']:.2f}% (+{distill_imp:.2f}%)\n\n")
        f.write(f"Synonym Attack:\n")
        f.write(f"  Baseline: {synonym_results['attack_success_rate']:.2f}% SR\n")
        f.write(f"  Adv Training: {adv_train_syn['attack_success_rate']:.2f}% SR\n")
        f.write(f"  Distillation: {distill_syn['attack_success_rate']:.2f}% SR\n")
    
    print("\n✓ Results saved to 'results/final_results.txt'")
    print("\n" + "=" * 80)
    print("PROJECT COMPLETE!")
    print("=" * 80)
    
    return {
        'baseline_pgd': pgd_01,
        'baseline_synonym': synonym_results,
        'adv_train_pgd': adv_train_pgd,
        'adv_train_syn': adv_train_syn,
        'distill_pgd': distill_pgd,
        'distill_syn': distill_syn
    }

if __name__ == '__main__':
    results = main()