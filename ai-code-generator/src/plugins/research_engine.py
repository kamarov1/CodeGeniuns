import asyncio
import aiohttp
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLearation
import ast
from pylint import epylint as lint
from radon.complexity import cc_visit
from radon.metrics import h_visit
from ..config.settings import (
    GITHUB_API_TOKEN,
    STACKOVERFLOW_API_KEY,
    ARXIV_API_ENDPOINT,
    RESEARCH_CACHE_DURATION,
    MAX_CONCURRENT_REQUESTS,
    NLP_MODEL_NAME
)
from ..utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class ResearchEngine:
    def __init__(self):
        self.cache_manager = CacheManager()
        self.tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLearation.from_pretrained(NLP_MODEL_NAME)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def research_topic(self, topic: str) -> Dict[str, Any]:
        cached_result = self.cache_manager.get(topic)
        if cached_result:
            return cached_result

        tasks = [
            self.search_github(topic),
            self.search_stackoverflow(topic),
            self.search_arxiv(topic)
        ]
        results = await asyncio.gather(*tasks)

        combined_results = self.combine_results(results)
        summarized_results = await self.summarize_results(combined_results)

        self.cache_manager.set(topic, summarized_results, RESEARCH_CACHE_DURATION)
        return summarized_results

    async def search_github(self, query: str) -> List[Dict[str, Any]]:
        url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc"
        headers = {"Authorization": f"token {GITHUB_API_TOKEN}"}
        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    data = await response.json()
                    return [
                        {
                            "title": repo["name"],
                            "url": repo["html_url"],
                            "description": repo["description"],
                            "stars": repo["stargazers_count"],
                            "source": "GitHub"
                        }
                        for repo in data.get("items", [])[:10]
                    ]

    async def search_stackoverflow(self, query: str) -> List[Dict[str, Any]]:
        url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=votes&q={query}&site=stackoverflow&key={STACKOVERFLOW_API_KEY}"
        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    return [
                        {
                            "title": question["title"],
                            "url": question["link"],
                            "score": question["score"],
                            "source": "Stack Overflow"
                        }
                        for question in data.get("items", [])[:10]
                    ]

    async def search_arxiv(self, query: str) -> List[Dict[str, Any]]:
        url = f"{ARXIV_API_ENDPOINT}?search_query=all:{query}&start=0&max_results=10"
        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    text = await response.text()
                    soup = BeautifulSoup(text, 'xml')
                    entries = soup.find_all('entry')
                    return [
                        {
                            "title": entry.title.text,
                            "url": entry.id.text,
                            "summary": entry.summary.text,
                            "source": "arXiv"
                        }
                        for entry in entries
                    ]

    def combine_results(self, results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        all_results = [item for sublist in results for item in sublist]
        
        for result in all_results:
            if result["source"] == "GitHub":
                result["score"] = result["stars"] * np.log(1 + len(result.get("description", "")))
            elif result["source"] == "Stack Overflow":
                result["score"] = result["score"] * np.log(1 + len(result["title"]))
            else:
                result["score"] = len(result.get("summary", "")) / 100
        
        return sorted(all_results, key=lambda x: x["score"], reverse=True)

    async def summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        summaries = []
        for result in results[:5]:
            text = f"{result['title']}. {result.get('description', '')}. {result.get('summary', '')}"
            summary = self.summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            summaries.append(summary)

        combined_summary = " ".join(summaries)
        
        return {
            "top_results": results[:10],
            "summary": combined_summary,
            "trends": self.extract_trends(results),
            "best_practices": await self.extract_best_practices(results)
        }

    def extract_trends(self, results: List[Dict[str, Any]]) -> List[str]:
        texts = [f"{r['title']} {r.get('description', '')}" for r in results]
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-10:][::-1]
        trends = [feature_names[i] for i in top_indices]
        
        return trends

    async def extract_best_practices(self, results: List[Dict[str, Any]]) -> List[str]:
        best_practices = []
        for result in results:
            if result["source"] == "Stack Overflow":
                practices = await self.extract_practices_from_stackoverflow(result["url"])
                best_practices.extend(practices)
            elif result["source"] == "GitHub":
                practices = await self.extract_practices_from_github(result["url"])
                best_practices.extend(practices)

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(best_practices)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        unique_practices = []
        for i, practice in enumerate(best_practices):
            if not any(similarity_matrix[i][j] > 0.7 for j in range(i) if i != j):
                unique_practices.append(practice)

        return unique_practices[:10]

    async def extract_practices_from_stackoverflow(self, url: str) -> List[str]:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                answers = soup.find_all('div', class_='answer')
                practices = []
                for answer in answers:
                    text = answer.get_text()
                    practices.extend(self.extract_bullet_points(text))
                return practices

    async def extract_practices_from_github(self, url: str) -> List[str]:
        readme_url = f"{url}/raw/master/README.md"
        async with aiohttp.ClientSession() as session:
            async with session.get(readme_url) as response:
                if response.status == 200:
                    text = await response.text()
                    return self.extract_bullet_points(text)
        return []

    def extract_bullet_points(self, text: str) -> List[str]:
        lines = text.split('\n')
        bullet_points = [line.strip()[1:].strip() for line in lines if line.strip().startswith(('-', '*', '+'))]
        return [point for point in bullet_points if len(point) > 10]

    async def get_code_examples(self, topic: str) -> List[Dict[str, Any]]:
        url = f"https://api.github.com/search/code?q={topic}+in:file+language:python"
        headers = {"Authorization": f"token {GITHUB_API_TOKEN}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                data = await response.json()
                examples = []
                for item in data.get("items", [])[:5]:
                    raw_url = item["html_url"].replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                    async with session.get(raw_url) as code_response:
                        code = await code_response.text()
                        analysis = await self.analyze_code_quality(code)
                        examples.append({
                            "filename": item["name"],
                            "url": item["html_url"],
                            "code": code,
                            "analysis": analysis
                        })
                return examples

    async def analyze_code_quality(self, code: str) -> Dict[str, Any]:
        # Análise estática com pylint
        (pylint_stdout, pylint_stderr) = lint.py_run(code, return_std=True)
        pylint_score = float(pylint_stdout.getvalue().split('\n')[-3].split('/')[0])

        # Análise de complexidade com radon
        complexity = cc_visit(code)
        halstead = h_visit(code)

        # Análise AST
        try:
            tree = ast.parse(code)
            ast_analysis = {
                "num_functions": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                "num_classes": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                "num_imports": len([node for node in ast.walk(tree) if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)])
            }
        except SyntaxError:
            ast_analysis = {"error": "Invalid syntax"}

        return {
            "pylint_score": pylint_score,
            "cyclomatic_complexity": {f.name: f.complexity for f in complexity},
            "halstead_metrics": {
                "h1": halstead.h1,
                "h2": halstead.h2,
                "N1": halstead.N1,
                "N2": halstead.N2,
                "vocabulary": halstead.vocabulary,
                "length": halstead.length,
                "calculated_length": halstead.calculated_length,
                "volume": halstead.volume,
                "difficulty": halstead.difficulty,
                "effort": halstead.effort,
                "time": halstead.time,
                "bugs": halstead.bugs
            },
            "ast_analysis": ast_analysis
        }
