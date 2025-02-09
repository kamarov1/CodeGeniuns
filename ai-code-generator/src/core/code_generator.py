import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from datetime import datetime
import logging
from ..config.settings import MODEL_SAVE_PATH, FEEDBACK_THRESHOLD, RETRAINING_INTERVAL
from ..utils.code_similarity import calculate_code_similarity
from ..models.meta_learning import MetaLearner

logger = logging.getLogger(__name__)

class LearningEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.meta_learner = MetaLearner()
        self.feedback_buffer = []
        self.last_training_time = datetime.now()

    def train(self, code_samples: List[Dict[str, str]], labels: List[str]):
        """
        Treina o modelo com novas amostras de código.
        
        Args:
            code_samples (List[Dict[str, str]]): Lista de dicionários contendo 'prompt' e 'code'.
            labels (List[str]): Lista de labels correspondentes (e.g., 'good', 'bad').
        """
        X = [sample['prompt'] + ' ' + sample['code'] for sample in code_samples]
        y = labels

        X = self.vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        logger.info(f"Model trained. Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

        self._save_model()

    def predict(self, prompt: str, generated_code: str) -> float:
        """
        Prediz a qualidade do código gerado.
        
        Args:
            prompt (str): O prompt original.
            generated_code (str): O código gerado.
        
        Returns:
            float: Score de qualidade (0-1).
        """
        X = self.vectorizer.transform([prompt + ' ' + generated_code])
        probabilities = self.classifier.predict_proba(X)
        return probabilities[0][1]  # Assumindo que a classe 'good' é indexada como 1

    def process_feedback(self, prompt: str, generated_code: str, user_rating: float):
        """
        Processa o feedback do usuário e atualiza o buffer de feedback.
        
        Args:
            prompt (str): O prompt original.
            generated_code (str): O código gerado.
            user_rating (float): Avaliação do usuário (0-1).
        """
        self.feedback_buffer.append({
            'prompt': prompt,
            'code': generated_code,
            'rating': user_rating
        })

        if len(self.feedback_buffer) >= FEEDBACK_THRESHOLD:
            self._retrain_if_needed()

    def _retrain_if_needed(self):
        """Verifica se é necessário retreinar o modelo e o faz se for o caso."""
        current_time = datetime.now()
        if (current_time - self.last_training_time).days >= RETRAINING_INTERVAL:
            logger.info("Retraining model with new feedback data...")
            code_samples = [{'prompt': fb['prompt'], 'code': fb['code']} for fb in self.feedback_buffer]
            labels = ['good' if fb['rating'] > 0.5 else 'bad' for fb in self.feedback_buffer]
            self.train(code_samples, labels)
            self.feedback_buffer = []
            self.last_training_time = current_time

    def _save_model(self):
        """Salva o modelo treinado e o vectorizer."""
        joblib.dump(self.classifier, f"{MODEL_SAVE_PATH}/classifier.joblib")
        joblib.dump(self.vectorizer, f"{MODEL_SAVE_PATH}/vectorizer.joblib")
        logger.info(f"Model saved to {MODEL_SAVE_PATH}")

    def load_model(self):
        """Carrega um modelo previamente salvo."""
        try:
            self.classifier = joblib.load(f"{MODEL_SAVE_PATH}/classifier.joblib")
            self.vectorizer = joblib.load(f"{MODEL_SAVE_PATH}/vectorizer.joblib")
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.warning("No saved model found. Using a new model.")

    def adapt_to_user_style(self, user_code_samples: List[str]):
        """
        Adapta o modelo para o estilo de código do usuário.
        
        Args:
            user_code_samples (List[str]): Amostras de código do usuário.
        """
        adapted_features = self.meta_learner.extract_style_features(user_code_samples)
        self.meta_learner.update_model(adapted_features)
        logger.info("Model adapted to user's coding style")

    def generate_improved_code(self, original_code: str, feedback: str) -> str:
        """
        Gera uma versão melhorada do código com base no feedback.
        
        Args:
            original_code (str): O código original gerado.
            feedback (str): Feedback do usuário para melhoria.
        
        Returns:
            str: Código melhorado.
        """
        improved_code = self.meta_learner.apply_improvements(original_code, feedback)
        similarity = calculate_code_similarity(original_code, improved_code)
        
        if similarity < 0.8:  # Evita mudanças muito drásticas
            logger.info(f"Generated improved code. Similarity with original: {similarity:.2f}")
            return improved_code
        else:
            logger.info("Improvement attempt resulted in too similar code. Returning original.")
            return original_code

    def analyze_error_patterns(self, error_logs: List[Dict[str, str]]):
        """
        Analisa padrões de erro para melhorar a geração de código.
        
        Args:
            error_logs (List[Dict[str, str]]): Lista de logs de erro.
        """
        error_types = [log['error_type'] for log in error_logs]
        error_messages = [log['error_message'] for log in error_logs]

        X = self.vectorizer.fit_transform(error_messages)
        error_clusters = self.meta_learner.cluster_errors(X)

        for cluster_id, error_indices in enumerate(error_clusters):
            cluster_errors = [error_types[i] for i in error_indices]
            most_common_error = max(set(cluster_errors), key=cluster_errors.count)
            logger.info(f"Error Cluster {cluster_id}: Most common error - {most_common_error}")

        self.meta_learner.update_error_patterns(error_clusters, error_types)

    def suggest_code_improvements(self, code: str) -> List[str]:
        """
        Sugere melhorias para um trecho de código.
        
        Args:
            code (str): O código a ser melhorado.
        
        Returns:
            List[str]: Lista de sugestões de melhoria.
        """
        code_features = self.meta_learner.extract_code_features(code)
        suggestions = self.meta_learner.generate_suggestions(code_features)
        return suggestions