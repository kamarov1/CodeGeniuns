import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from typing import Dict, Any, List, Tuple
from . import Plugin
from ..utils.performance_monitor import log_execution_time
from ..core.learning_engine import LearningEngine
from ..utils.code_analyzer import CodeAnalyzer
from ..config.settings import (
    FEEDBACK_BATCH_SIZE, 
    LEARNING_RATE, 
    BERT_MODEL_PATH, 
    DBSCAN_EPS, 
    DBSCAN_MIN_SAMPLES
)
import logging
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import spacy
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class AdvancedFeedbackProcessor(Plugin):
    def __init__(self):
        self.learning_engine = LearningEngine()
        self.code_analyzer = CodeAnalyzer()
        self.feedback_buffer = []
        self.sentiment_model = self._load_sentiment_model()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.clusterer = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.nlp = spacy.load("en_core_web_sm")

    def _load_sentiment_model(self):
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        model.eval()
        return model

    @log_execution_time
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa o feedback do usuário de forma assíncrona e atualiza o modelo de aprendizado.
        """
        try:
            self._validate_feedback_data(data)
            
            prompt, generated_code, user_rating, user_comments = self._extract_feedback_data(data)

            # Análise assíncrona
            code_quality_task = asyncio.create_task(self._analyze_code_quality(generated_code))
            sentiment_task = asyncio.create_task(self._analyze_sentiment(user_comments))
            
            code_quality_metrics, sentiment = await asyncio.gather(code_quality_task, sentiment_task)

            # Processamento do feedback
            await self.learning_engine.process_feedback(prompt, generated_code, user_rating, sentiment)

            # Adiciona ao buffer de feedback para análise em lote
            self.feedback_buffer.append({
                'prompt': prompt,
                'code': generated_code,
                'rating': user_rating,
                'comments': user_comments,
                'metrics': code_quality_metrics,
                'sentiment': sentiment
            })

            # Verifica se deve realizar análise em lote
            if len(self.feedback_buffer) >= FEEDBACK_BATCH_SIZE:
                await self._process_feedback_batch()

            insights = await self._analyze_user_feedback(user_comments, code_quality_metrics, sentiment)
            improvement_suggestions = await self._generate_improvement_suggestions(
                prompt, generated_code, user_rating, insights
            )

            return {
                "status": "success",
                "message": "Feedback processado com sucesso",
                "improvement_suggestions": improvement_suggestions,
                "insights": insights,
                "code_quality_metrics": code_quality_metrics,
                "sentiment_analysis": sentiment
            }

        except Exception as e:
            logger.error(f"Erro ao processar feedback: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Falha ao processar feedback: {str(e)}"
            }

    def _validate_feedback_data(self, data: Dict[str, Any]):
        """Valida os dados de feedback recebidos."""
        required_fields = ['prompt', 'generated_code', 'user_rating']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Campo obrigatório ausente: {field}")
        
        if not isinstance(data['user_rating'], (int, float)) or not 0 <= data['user_rating'] <= 5:
            raise ValueError("A avaliação do usuário deve ser um número entre 0 e 5")

    def _extract_feedback_data(self, data: Dict[str, Any]) -> Tuple[str, str, float, str]:
        """Extrai e normaliza os dados de feedback."""
        return (
            data['prompt'],
            data['generated_code'],
            data['user_rating'] / 5.0,  # Normaliza para 0-1
            data.get('user_comments', '')
        )

    async def _analyze_code_quality(self, code: str) -> Dict[str, float]:
        """Analisa a qualidade do código de forma assíncrona."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.code_analyzer.analyze, code
        )

    async def _analyze_sentiment(self, comments: str) -> float:
        """Realiza análise de sentimento avançada usando BERT."""
        inputs = self.tokenizer(comments, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        scores = outputs.logits.softmax(dim=1)
        return scores[0][1].item()  # Retorna o score de sentimento positivo

    async def _process_feedback_batch(self):
        """Processa o lote de feedbacks acumulados de forma assíncrona."""
        comments = [fb['comments'] for fb in self.feedback_buffer]
        codes = [fb['code'] for fb in self.feedback_buffer]

        # Vetorização e clustering dos comentários
        vectors = self.vectorizer.fit_transform(comments)
        clusters = self.clusterer.fit_predict(vectors.toarray())

        # Análise de código em lote
        code_metrics = await asyncio.gather(*[self._analyze_code_quality(code) for code in codes])

        # Atualização do modelo em lote
        update_tasks = []
        for fb, cluster, metrics in zip(self.feedback_buffer, clusters, code_metrics):
            task = self.learning_engine.update_model(
                fb['prompt'], 
                fb['code'], 
                fb['rating'], 
                fb['sentiment'], 
                int(cluster),
                metrics
            )
            update_tasks.append(task)

        await asyncio.gather(*update_tasks)

        # Análise de tendências
        await self._analyze_feedback_trends(clusters, code_metrics)

        # Limpa o buffer
        self.feedback_buffer.clear()

    async def _analyze_feedback_trends(self, clusters: np.ndarray, code_metrics: List[Dict[str, float]]):
        """Analisa tendências nos feedbacks para insights de longo prazo e atualiza estratégias de geração de código."""
        cluster_metrics = self._group_metrics_by_cluster(clusters, code_metrics)
        trend_insights = await self._detect_metric_trends(cluster_metrics)
        code_generation_updates = await self._generate_code_strategy_updates(trend_insights)
        
        await self._update_code_generation_strategies(code_generation_updates)
        await self._log_trend_analysis(trend_insights, code_generation_updates)

    def _group_metrics_by_cluster(self, clusters: np.ndarray, code_metrics: List[Dict[str, float]]) -> Dict[int, List[Dict[str, float]]]:
        """Agrupa métricas de código por cluster."""
        cluster_metrics = {}
        for cluster, metrics in zip(clusters, code_metrics):
            if cluster not in cluster_metrics:
                cluster_metrics[cluster] = []
            cluster_metrics[cluster].append(metrics)
        return cluster_metrics

    async def _detect_metric_trends(self, cluster_metrics: Dict[int, List[Dict[str, float]]]) -> Dict[str, Any]:
        """Detecta tendências significativas nas métricas de código ao longo do tempo."""
        trend_insights = {}
        for cluster, metrics_list in cluster_metrics.items():
            df = pd.DataFrame(metrics_list)
            
            # Verificar se o DataFrame está vazio
            if df.empty:
                trend_insights[cluster] = {}
                continue
                
            df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
            df.set_index('timestamp', inplace=True)

            # Análise de tendência temporal
            trend_insights[cluster] = {
                'temporal_trends': await self._analyze_temporal_trends(df),
                'metric_correlations': await self._analyze_metric_correlations(df),
                'anomalies': await self._detect_anomalies(df),
                'principal_components': await self._perform_pca_analysis(df)
            }

        return trend_insights

    async def _analyze_temporal_trends(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analisa tendências temporais nas métricas."""
        trends = {}
        for column in df.columns:
            trend = np.polyfit(range(len(df)), df[column], 1)[0]
            trends[column] = trend
        return trends

    async def _analyze_metric_correlations(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analisa correlações entre diferentes métricas."""
        return df.corr().to_dict()

    async def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, List[datetime]]:
        """Detecta anomalias nas métricas usando o método IQR."""
        anomalies = {}
        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
        return anomalies

    async def _perform_pca_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Realiza análise de componentes principais para identificar padrões latentes."""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'principal_components': pca_result.tolist()
        }

    async def _generate_code_strategy_updates(self, trend_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gera atualizações para as estratégias de geração de código com base nas tendências detectadas."""
        strategy_updates = []

        for cluster, insights in trend_insights.items():
            # Verificar se insights está vazio
            if not insights:
                continue
                
            temporal_trends = insights['temporal_trends']
            correlations = insights['metric_correlations']
            anomalies = insights['anomalies']
            pca_results = insights['principal_components']

            # Análise de tendências temporais
            for metric, trend in temporal_trends.items():
                if abs(trend) > 0.1:  # Tendência significativa
                    direction = "aumentando" if trend > 0 else "diminuindo"
                    strategy_updates.append({
                        'cluster': cluster,
                        'metric': metric,
                        'trend': direction,
                        'action': f"Ajustar geração para {'melhorar' if trend < 0 else 'manter'} {metric}"
                    })

            # Análise de correlações
            for metric1, corr_dict in correlations.items():
                for metric2, corr_value in corr_dict.items():
                    if abs(corr_value) > 0.7 and metric1 != metric2:
                        strategy_updates.append({
                            'cluster': cluster,
                            'metrics': [metric1, metric2],
                            'correlation': corr_value,
                            'action': f"Considerar trade-off entre {metric1} e {metric2} na geração"
                        })

            # Análise de anomalias
            for metric, anomaly_dates in anomalies.items():
                if anomaly_dates:
                    strategy_updates.append({
                        'cluster': cluster,
                        'metric': metric,
                        'anomalies': len(anomaly_dates),
                        'action': f"Investigar e ajustar geração para evitar anomalias em {metric}"
                    })

            # Análise PCA
            if pca_results['explained_variance_ratio'][0] > 0.5:
                strategy_updates.append({
                    'cluster': cluster,
                    'pca_insight': "Forte padrão dominante detectado",
                    'action': "Focar na otimização do componente principal na geração de código"
                })

        return strategy_updates

    async def _update_code_generation_strategies(self, strategy_updates: List[Dict[str, Any]]):
        """Atualiza as estratégias de geração de código com base nas análises de tendências."""
        for update in strategy_updates:
            if 'metric' in update:
                await self.learning_engine.adjust_generation_parameter(update['metric'], update['action'])
            elif 'metrics' in update:
                await self.learning_engine.adjust_metric_balance(update['metrics'], update['correlation'])
            elif 'pca_insight' in update:
                await self.learning_engine.optimize_principal_component(update['cluster'])

        # Atualiza os pesos do modelo com base nas novas estratégias
        await self.learning_engine.update_model_weights()

    async def _log_trend_analysis(self, trend_insights: Dict[str, Any], strategy_updates: List[Dict[str, Any]]):
        """Registra os resultados da análise de tendências e atualizações de estratégia."""
        logger.info("Análise de Tendências de Feedback:")
        for cluster, insights in trend_insights.items():
            logger.info(f"Cluster {cluster}:")
            # Verificar se insights está vazio
            if not insights:
                continue
            logger.info(f"  Tendências Temporais: {insights['temporal_trends']}")
            logger.info(f"  Anomalias Detectadas: {sum(len(dates) for dates in insights['anomalies'].values())}")
            logger.info(f"  Componentes Principais: {insights['principal_components']['explained_variance_ratio'][:3]}")

        logger.info("Atualizações de Estratégia de Geração de Código:")
        for update in strategy_updates:
            logger.info(f"  {update['action']}")

    async def _analyze_user_feedback(self, user_comments: str, code_quality_metrics: Dict[str, float], sentiment: float) -> Dict[str, Any]:
        """Analisa o feedback do usuário para extrair insights."""
        insights = {
            "sentiment": sentiment,
            "subjectivity": self._extract_subjectivity(user_comments),
            "keywords": self._extract_keywords(user_comments),
            "code_relation": self._relate_feedback_to_code(user_comments, code_quality_metrics)
        }
        return insights

    def _extract_subjectivity(self, text: str) -> float:
        """Extrai a subjetividade do texto usando spaCy."""
        doc = self.nlp(text)
        subjectivity = 0
        for token in doc:
            if token.pos_ == "ADJ" or token.pos_ == "ADV":
                subjectivity += 1
        return subjectivity / len(doc) if doc else 0

    def _extract_keywords(self, text: str) -> List[str]:
        """Extrai palavras-chave do texto."""
        doc = self.nlp(text)
        keywords = [token.text for token in doc if token.pos_ == "NOUN"]
        return keywords

    def _relate_feedback_to_code(self, user_comments: str, code_quality_metrics: Dict[str, float]) -> str:
        """Relaciona o feedback do usuário com as métricas de qualidade do código."""
        # Implementar lógica para relacionar comentários com métricas do código
        # Exemplo: verificar se o comentário menciona "complexidade" e a métrica de complexidade é alta
        if "complex" in user_comments.lower() and code_quality_metrics.get("cyclomatic_complexity", 0) > 10:
            return "Comentário sobre complexidade alta no código"
        return "Sem relação direta encontrada"

    async def _generate_improvement_suggestions(self, prompt: str, generated_code: str, user_rating: float, insights: Dict[str, Any]) -> List[str]:
        """Gera sugestões de melhoria com base no feedback do usuário e nos insights extraídos."""
        suggestions = []

        # Sugestões baseadas no sentimento
        if insights["sentiment"] < 0.3:
            suggestions.append("Melhorar a clareza e legibilidade do código.")
        elif insights["sentiment"] > 0.7:
            suggestions.append("Continuar mantendo a alta qualidade do código.")

        # Sugestões baseadas na subjetividade
        if insights["subjectivity"] > 0.5:
            suggestions.append("Tentar reduzir a subjetividade no código, focando em soluções mais objetivas.")

        # Sugestões baseadas nas palavras-chave
        if "performance" in insights["keywords"]:
            suggestions.append("Otimizar o código para melhor desempenho.")

        # Sugestões baseadas na relação com o código
        if "complexidade" in insights["code_relation"]:
            suggestions.append("Simplificar o código para reduzir a complexidade.")

        # Sugestões adicionais baseadas na avaliação do usuário
        if user_rating < 0.5:
            suggestions.append("Revisar a lógica do código para garantir que ele atenda aos requisitos.")

        return await self._prioritize_suggestions(suggestions)

    async def _prioritize_suggestions(self, suggestions: List[str]) -> List[str]:
        """Prioriza e filtra sugestões para fornecer as mais relevantes."""
        if not suggestions:
            return []

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(suggestions)
        
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        unique_suggestions = []
        for i, suggestion in enumerate(suggestions):
            if not any(similarity_matrix[i][j] > 0.7 for j in range(len(suggestions)) if i != j):
                unique_suggestions.append(suggestion)
        
        tfidf_scores = tfidf_matrix.sum(axis=1).A1
        scored_suggestions = [(suggestion, score) for suggestion, score in zip(unique_suggestions, tfidf_scores)]
        sorted_suggestions = sorted(scored_suggestions, key=lambda x: x[1], reverse=True)
        
        return [suggestion for suggestion, _ in sorted_suggestions[:5]]

    def get_metadata(self) -> Dict[str, str]:
        """
        Retorna metadados sobre o plugin de processamento de feedback avançado.
        """
        return {
            "name": "Advanced Feedback Processor",
            "version": "2.0",
            "description": "Processador de feedback avançado com análise de tendências e atualização dinâmica de estratégias",
            "author": "Sua Empresa",
            "requires": ["learning_engine", "code_analyzer", "spacy", "scikit-learn", "pandas", "numpy", "transformers"]
        }

# Registrar o plugin no gerenciador
from . import plugin_manager
plugin_manager.register_plugin("advanced_feedback_processor", AdvancedFeedbackProcessor())
