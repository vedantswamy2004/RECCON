import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import re
import spacy
from tqdm import tqdm

class EmotionCausalSpanExtractor:
    def __init__(self, use_context=True, use_pos=True, window_size=3):
        """
        Initialize the causal span extractor.
        
        Args:
            use_context (bool): Whether to use contextual information from previous turns
            use_pos (bool): Whether to use POS tagging features
            window_size (int): Number of previous turns to consider for context
        """
        self.use_context = use_context
        self.use_pos = use_pos
        self.window_size = window_size
        
        # Load spaCy model for linguistic features
        if use_pos:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Installing spaCy model...")
                import os
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize the SVC classifier pipeline
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('classifier', SVC(kernel='linear', probability=True, C=1.0))
        ])
        
        # For tracking candidate spans
        self.span_candidates = {}
        
    def preprocess_text(self, text):
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        return text
    
    def extract_linguistic_features(self, text):
        """Extract POS and dependency features using spaCy."""
        if not self.use_pos:
            return ""
        
        doc = self.nlp(text)
        pos_tags = " ".join([f"POS_{token.pos_}" for token in doc])
        dep_tags = " ".join([f"DEP_{token.dep_}" for token in doc])
        
        return f" {pos_tags} {dep_tags}"
    
    def generate_candidates(self, conversation, target_idx):
        """
        Generate candidate spans that could potentially be causal for the emotion.
        
        Args:
            conversation (list): List of conversation utterances
            target_idx (int): Index of the target utterance with emotion
            
        Returns:
            list: List of candidate spans with their positions
        """
        candidates = []
        
        # Determine the range of utterances to consider
        start_idx = max(0, target_idx - self.window_size)
        
        # Current utterance is always a candidate
        current_utterance = conversation[target_idx]['utterance']
        candidates.append({
            'span': current_utterance,
            'turn': conversation[target_idx]['turn'],
            'is_self': True
        })
        
        # Add previous utterances as candidates
        for i in range(start_idx, target_idx):
            prev_utterance = conversation[i]['utterance']
            candidates.append({
                'span': prev_utterance,
                'turn': conversation[i]['turn'],
                'is_self': False
            })
            
        # Also add sentence-level spans from the current utterance
        sentences = re.split(r'[.!?]+', current_utterance)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for i, sentence in enumerate(sentences):
            if sentence != current_utterance:  # Avoid duplicates
                candidates.append({
                    'span': sentence,
                    'turn': conversation[target_idx]['turn'],
                    'is_self': True,
                    'is_partial': True,
                    'sentence_idx': i
                })
                
        return candidates
    
    def extract_features(self, conversation, target_idx, candidate):
        """
        Extract features for a candidate span.
        
        Args:
            conversation (list): List of conversation utterances
            target_idx (int): Index of the target utterance with emotion
            candidate (dict): Candidate span information
            
        Returns:
            str: Feature representation
        """
        features = []
        
        # The span itself
        span_text = self.preprocess_text(candidate['span'])
        features.append(span_text)
        
        # Add linguistic features
        features.append(self.extract_linguistic_features(span_text))
        
        # Position features
        if candidate['is_self']:
            features.append("SELF_UTTERANCE")
        else:
            # Calculate distance based on turns
            target_turn = conversation[target_idx]['turn']
            candidate_turn = candidate['turn']
            distance = target_turn - candidate_turn
            features.append(f"DISTANCE_{distance}")
        
        # Speaker information
        target_speaker = conversation[target_idx]['speaker']
        if 'is_partial' not in candidate:  # For full utterances
            # Find the utterance with matching turn
            candidate_speaker = None
            for utt in conversation:
                if utt['turn'] == candidate['turn']:
                    candidate_speaker = utt['speaker']
                    break
                    
            if candidate_speaker:
                if target_speaker == candidate_speaker:
                    features.append("SAME_SPEAKER")
                else:
                    features.append("DIFFERENT_SPEAKER")
        
        # Emotion of the target
        target_emotion = conversation[target_idx]['emotion']
        features.append(f"EMOTION_{target_emotion}")
        
        return " ".join(features)
    
    def create_training_data(self, dataset):
        """
        Process the dataset to create training examples.
        
        Args:
            dataset (dict): Dictionary of conversations
            
        Returns:
            tuple: X (features) and y (labels) for training
        """
        X = []
        y = []
        
        for conv_id, conversation_data in tqdm(dataset.items(), desc="Processing conversations"):
            # Navigate the nested structure of your dataset
            conversation = conversation_data[0]
            
            for target_idx, utterance in enumerate(conversation):
                # Skip utterances without emotion or with neutral emotion
                if 'emotion' not in utterance or utterance['emotion'] == 'neutral':
                    continue
                
                # Get the emotion and any labeled causal spans
                emotion = utterance['emotion']
                causal_spans = []
                
                # Check both field formats
                if 'expanded_emotion_cause_span' in utterance:
                    causal_spans = utterance['expanded_emotion_cause_span']
                elif 'expanded emotion cause span' in utterance:
                    causal_spans = utterance['expanded emotion cause span']
                
                # If no spans are found, skip this utterance
                if not causal_spans:
                    continue
                
                # Generate candidate spans
                candidates = self.generate_candidates(conversation, target_idx)
                
                # Store candidates for later use
                key = f"{conv_id}_{target_idx}"
                self.span_candidates[key] = candidates
                
                # Extract features and label each candidate
                for candidate in candidates:
                    features = self.extract_features(conversation, target_idx, candidate)
                    X.append(features)
                    
                    # Check if this span is in the labeled causal spans
                    is_causal = False
                    span_text = candidate['span'].strip()
                    for causal_span in causal_spans:
                        causal_text = causal_span.strip()
                        # Check for exact match or significant overlap
                        if span_text == causal_text or \
                           (len(span_text) > 5 and causal_text.find(span_text) != -1) or \
                           (len(causal_text) > 5 and span_text.find(causal_text) != -1):
                            is_causal = True
                            break
                    
                    y.append(1 if is_causal else 0)
        
        return X, y
    
    def fit(self, dataset):
        """
        Train the SVC model using the provided dataset.
        
        Args:
            dataset (dict): Dictionary of conversations
            
        Returns:
            self: Trained model
        """
        X, y = self.create_training_data(dataset)
        
        if len(X) == 0:
            raise ValueError("No training examples were generated!")
        
        # Train-test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Training on {len(X_train)} examples, testing on {len(X_test)} examples")
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Retrain on all data
        self.pipeline.fit(X, y)
        
        return self
    
    def predict_causal_spans(self, conversation_data, target_idx=None):
        """
        Predict the causal spans for emotions in a conversation.
        
        Args:
            conversation_data: Either a complete conversation from the dataset or a specific conversation ID
            target_idx (int, optional): If provided, only predict for this specific utterance
            
        Returns:
            dict: Mapping of utterance indices to predicted causal spans
        """
        results = {}
        
        # If given a conversation ID, extract the conversation
        if isinstance(conversation_data, str) and conversation_data in self.dataset:
            conversation = self.dataset[conversation_data][0]
        else:
            # Assume it's the actual conversation list
            conversation = conversation_data
        
        # Determine which utterances to process
        if target_idx is not None:
            indices = [target_idx]
        else:
            indices = [i for i, utt in enumerate(conversation) 
                     if 'emotion' in utt and utt['emotion'] != 'neutral']
        
        # Process each target utterance
        for idx in indices:
            # Generate candidates
            candidates = self.generate_candidates(conversation, idx)
            
            # Extract features for each candidate
            X = []
            for candidate in candidates:
                features = self.extract_features(conversation, idx, candidate)
                X.append(features)
            
            if not X:  # Skip if no candidates
                continue
                
            # Get probabilities for each candidate
            probas = self.pipeline.predict_proba(X)
            
            # Get indices of positive predictions
            positive_indices = []
            for i, (prob_neg, prob_pos) in enumerate(probas):
                if prob_pos > 0.5:  # Threshold can be adjusted
                    positive_indices.append((i, prob_pos))
            
            # Sort by probability
            positive_indices.sort(key=lambda x: x[1], reverse=True)
            
            # Store the spans in order of confidence
            causal_spans = []
            for idx_candidate, _ in positive_indices:
                causal_spans.append(candidates[idx_candidate]['span'])
            
            results[idx] = causal_spans
        
        return results
    
    def evaluate_on_dataset(self, test_dataset=None):
        """Evaluate the model on a test dataset or the training dataset."""
        if test_dataset is None:
            test_dataset = self.dataset
        
        total_correct = 0
        total_predicted = 0
        total_gold = 0
        
        for conv_id, conversation_data in tqdm(test_dataset.items(), desc="Evaluating"):
            conversation = conversation_data[0]
            
            for target_idx, utterance in enumerate(conversation):
                if 'emotion' not in utterance or utterance['emotion'] == 'neutral':
                    continue
                
                # Get ground truth spans
                gold_spans = []
                if 'expanded_emotion_cause_span' in utterance:
                    gold_spans = utterance['expanded_emotion_cause_span']
                elif 'expanded emotion cause span' in utterance:
                    gold_spans = utterance['expanded emotion cause span']
                
                if not gold_spans:
                    continue
                
                # Get predictions
                pred_result = self.predict_causal_spans(conversation, target_idx)
                if target_idx not in pred_result:
                    continue
                
                pred_spans = pred_result[target_idx]
                
                # Count matches
                matches = 0
                for pred_span in pred_spans:
                    for gold_span in gold_spans:
                        if pred_span.strip() == gold_span.strip() or \
                           (len(pred_span) > 5 and gold_span.find(pred_span) != -1) or \
                           (len(gold_span) > 5 and pred_span.find(gold_span) != -1):
                            matches += 1
                            break
                
                total_correct += matches
                total_predicted += len(pred_spans)
                total_gold += len(gold_spans)
        
        # Calculate precision, recall and F1
        precision = total_correct / total_predicted if total_predicted > 0 else 0
        recall = total_correct / total_gold if total_gold > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def save_model(self, filepath):
        """Save the model to a file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'use_context': self.use_context,
                'use_pos': self.use_pos,
                'window_size': self.window_size
            }, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a model from a file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            use_context=data['use_context'],
            use_pos=data['use_pos'],
            window_size=data['window_size']
        )
        model.pipeline = data['pipeline']
        return model


def load_data_from_file(filepath):
    """Load and prepare the dataset from a JSON file."""
    with open(filepath, 'r') as f:
        dataset = json.load(f)
    return dataset


def main():
    # Example usage
    print("Loading dataset...")
    dataset_path = "data/original_annotation/dailydialog_train.json"  # Update with your actual path
    dataset = load_data_from_file(dataset_path)
    
    print("Creating and training model...")
    model = EmotionCausalSpanExtractor(use_context=True, use_pos=True, window_size=3)
    model.dataset = dataset  # Store the dataset for evaluation
    model.fit(dataset)
    
    # Evaluate on the dataset
    print("\nEvaluating model on training data:")
    metrics = model.evaluate_on_dataset()
    
    # Save the model
    model.save_model("emotion_causal_span_model.pkl")
    
    # Example prediction
    print("\nExample prediction:")
    example_conv_id = list(dataset.keys())[0]
    example_conv = dataset[example_conv_id][0]
    
    # Find an utterance with emotion and causal spans
    for i, utterance in enumerate(example_conv):
        if 'emotion' in utterance and utterance['emotion'] != 'neutral':
            causal_spans = None
            if 'expanded_emotion_cause_span' in utterance:
                causal_spans = utterance['expanded_emotion_cause_span']
            elif 'expanded emotion cause span' in utterance:
                causal_spans = utterance['expanded emotion cause span']
            
            if causal_spans:
                print(f"Utterance: {utterance['utterance']}")
                print(f"Emotion: {utterance['emotion']}")
                print(f"True causal spans: {causal_spans}")
                
                predicted_results = model.predict_causal_spans(example_conv, i)
                if i in predicted_results:
                    print(f"Predicted causal spans: {predicted_results[i]}")
                else:
                    print("No causal spans predicted.")
                break
    
    # Demonstrate how to use the model on a new conversation
    print("\nTo use the model on new data:")
    print("1. Load the model: model = EmotionCausalSpanExtractor.load_model('emotion_causal_span_model.pkl')")
    print("2. For a specific conversation and utterance: model.predict_causal_spans(conversation, target_idx)")


if __name__ == "__main__":
    main()