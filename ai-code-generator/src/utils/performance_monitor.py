import time
import psutil
import threading
from typing import Dict, List, Callable
from functools import wraps
import logging
from collections import deque
import matplotlib.pyplot as plt
from ..config.settings import PERFORMANCE_LOG_FILE, MAX_MEMORY_USAGE_PERCENT

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(PERFORMANCE_LOG_FILE)
logger.addHandler(file_handler)

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'execution_times': {},
            'api_calls': {},
            'errors': {}
        }
        self.lock = threading.Lock()
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Inicia o monitoramento contínuo em background"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._continuous_monitor)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Para o monitoramento em background"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _continuous_monitor(self):
        while self.monitoring:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            with self.lock:
                self.metrics['cpu_usage'].append(cpu)
                self.metrics['memory_usage'].append(memory)
            if memory > MAX_MEMORY_USAGE_PERCENT:
                logger.warning(f"High memory usage detected: {memory}%")
            time.sleep(1)

    def log_execution_time(self, func: Callable) -> Callable:
        """Decorator para registrar o tempo de execução de uma função"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            with self.lock:
                if func.__name__ not in self.metrics['execution_times']:
                    self.metrics['execution_times'][func.__name__] = deque(maxlen=100)
                self.metrics['execution_times'][func.__name__].append(execution_time)
            logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
            return result
        return wrapper

    def log_api_call(self, api_name: str):
        """Registra uma chamada de API"""
        with self.lock:
            self.metrics['api_calls'][api_name] = self.metrics['api_calls'].get(api_name, 0) + 1

    def log_error(self, error_type: str):
        """Registra um erro"""
        with self.lock:
            self.metrics['errors'][error_type] = self.metrics['errors'].get(error_type, 0) + 1

    def get_average_execution_time(self, func_name: str) -> float:
        """Retorna o tempo médio de execução para uma função específica"""
        with self.lock:
            times = self.metrics['execution_times'].get(func_name, [])
            return sum(times) / len(times) if times else 0

    def get_total_api_calls(self) -> int:
        """Retorna o número total de chamadas de API"""
        return sum(self.metrics['api_calls'].values())

    def get_error_count(self) -> int:
        """Retorna o número total de erros registrados"""
        return sum(self.metrics['errors'].values())

    def generate_performance_report(self) -> Dict:
        """Gera um relatório de desempenho"""
        with self.lock:
            report = {
                'average_cpu_usage': sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
                'average_memory_usage': sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                'total_api_calls': self.get_total_api_calls(),
                'total_errors': self.get_error_count(),
                'function_performance': {func: self.get_average_execution_time(func) for func in self.metrics['execution_times']},
                'api_call_distribution': dict(self.metrics['api_calls']),
                'error_distribution': dict(self.metrics['errors'])
            }
        return report

    def plot_performance_graphs(self):
        """Gera gráficos de desempenho"""
        plt.figure(figsize=(12, 8))
        
        # CPU Usage
        plt.subplot(2, 2, 1)
        plt.plot(list(self.metrics['cpu_usage']))
        plt.title('CPU Usage')
        plt.ylabel('Percentage')
        
        # Memory Usage
        plt.subplot(2, 2, 2)
        plt.plot(list(self.metrics['memory_usage']))
        plt.title('Memory Usage')
        plt.ylabel('Percentage')
        
        # API Calls
        plt.subplot(2, 2, 3)
        plt.bar(self.metrics['api_calls'].keys(), self.metrics['api_calls'].values())
        plt.title('API Calls')
        plt.xticks(rotation=45)
        
        # Errors
        plt.subplot(2, 2, 4)
        plt.bar(self.metrics['errors'].keys(), self.metrics['errors'].values())
        plt.title('Errors')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('performance_report.png')
        plt.close()

# Instância global do monitor de desempenho
performance_monitor = PerformanceMonitor()

def start_performance_monitoring():
    performance_monitor.start_monitoring()

def stop_performance_monitoring():
    performance_monitor.stop_monitoring()

def log_execution_time(func):
    return performance_monitor.log_execution_time(func)

def log_api_call(api_name: str):
    performance_monitor.log_api_call(api_name)

def log_error(error_type: str):
    performance_monitor.log_error(error_type)

def generate_performance_report() -> Dict:
    return performance_monitor.generate_performance_report()

def plot_performance_graphs():
    performance_monitor.plot_performance_graphs()