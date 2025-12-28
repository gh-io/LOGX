# engine/analyzer.py
import ast
import os
from typing import Dict, List

class Analyzer:
    """
    LOGX Analyzer: parses scripts and extracts logical structure.
    Supports Python, JS/TS (placeholder), Swift (placeholder)
    """
    def __init__(self):
        pass

    def analyze_python_file(self, filepath: str) -> Dict:
        """Extract functions, classes, and top-level logic"""
        with open(filepath, "r") as f:
            tree = ast.parse(f.read(), filename=filepath)
        result = {"functions": [], "classes": []}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                result["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                result["classes"].append(node.name)
        return result

    def analyze_repo(self, repo_path: str) -> Dict[str, Dict]:
        """Analyze all Python files in repo_path"""
        output = {}
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    output[path] = self.analyze_python_file(path)
        return output
