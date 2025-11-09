import random
import torch
import nltk
from nltk.corpus import wordnet

class SynonymSubstitutionAttack:
    """
    Simplified but aggressive synonym attack
    """
    def __init__(self, model, preprocessor, max_substitutions=20):
        self.model = model
        self.preprocessor = preprocessor
        self.max_substitutions = max_substitutions
        
        try:
            wordnet.ensure_loaded()
        except:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    
    def get_synonyms(self, word):
        """Get ALL synonyms without heavy filtering"""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', '').lower()
                if synonym != word and len(synonym) > 1:
                    synonyms.append(synonym)
        return list(set(synonyms))[:15]  # Limit to 15 for speed
    
    def attack_text(self, text, target_label, use_importance=False):
        """Simple but effective attack"""
        words = text.split()
        
        if len(words) < 3:
            return text
        
        # Get original prediction
        original_seq = torch.tensor([self.preprocessor.text_to_sequence(text)])
        with torch.no_grad():
            original_pred = self.model(original_seq).argmax(dim=1).item()
        
        if original_pred != target_label:
            return text
        
        # Try to substitute words
        modified_words = words.copy()
        substitutions = 0
        
        # Shuffle word order for randomness
        word_indices = list(range(len(words)))
        random.shuffle(word_indices)
        
        for idx in word_indices:
            if substitutions >= self.max_substitutions:
                break
            
            word = modified_words[idx].lower()
            
            # Skip short words or words not in vocab
            if len(word) <= 2 or word not in self.preprocessor.word2idx:
                continue
            
            # Get synonyms
            synonyms = self.get_synonyms(word)
            if not synonyms:
                continue
            
            # Try each synonym
            for synonym in synonyms:
                if synonym not in self.preprocessor.word2idx:
                    continue
                
                # Test this substitution
                test_words = modified_words.copy()
                test_words[idx] = synonym
                test_text = ' '.join(test_words)
                
                test_seq = torch.tensor([self.preprocessor.text_to_sequence(test_text)])
                with torch.no_grad():
                    test_pred = self.model(test_seq).argmax(dim=1).item()
                
                # If successful, keep it and move on
                if test_pred != target_label:
                    modified_words = test_words
                    substitutions += 1
                    break
                
                # Even if not successful, use synonyms that change the output
                # This makes the attack more aggressive
                if random.random() < 0.3:  # 30% chance to use anyway
                    modified_words[idx] = synonym
                    substitutions += 1
                    break
        
        return ' '.join(modified_words)