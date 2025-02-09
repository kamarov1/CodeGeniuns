from abc import ABC, abstractmethod
from typing import Dict, Any

class Plugin(ABC):
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método abstrato que cada plugin deve implementar para processar dados.
        
        Args:
            data (Dict[str, Any]): Dados de entrada para o plugin processar.
        
        Returns:
            Dict[str, Any]: Resultado do processamento do plugin.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, str]:
        """
        Retorna metadados sobre o plugin.
        
        Returns:
            Dict[str, str]: Dicionário contendo informações sobre o plugin.
        """
        pass

class PluginManager:
    def __init__(self):
        self.plugins = {}

    def register_plugin(self, name: str, plugin: Plugin):
        """
        Registra um novo plugin.
        
        Args:
            name (str): Nome do plugin.
            plugin (Plugin): Instância do plugin.
        """
        self.plugins[name] = plugin

    def get_plugin(self, name: str) -> Plugin:
        """
        Retorna um plugin pelo nome.
        
        Args:
            name (str): Nome do plugin.
        
        Returns:
            Plugin: Instância do plugin solicitado.
        
        Raises:
            KeyError: Se o plugin não estiver registrado.
        """
        return self.plugins[name]

    def execute_plugin(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa um plugin específico.
        
        Args:
            name (str): Nome do plugin a ser executado.
            data (Dict[str, Any]): Dados de entrada para o plugin.
        
        Returns:
            Dict[str, Any]: Resultado do processamento do plugin.
        
        Raises:
            KeyError: Se o plugin não estiver registrado.
        """
        plugin = self.get_plugin(name)
        return plugin.process(data)

    def list_plugins(self) -> Dict[str, Dict[str, str]]:
        """
        Lista todos os plugins registrados e seus metadados.
        
        Returns:
            Dict[str, Dict[str, str]]: Dicionário com nomes dos plugins e seus metadados.
        """
        return {name: plugin.get_metadata() for name, plugin in self.plugins.items()}

# Instância global do gerenciador de plugins
plugin_manager = PluginManager()
