# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from engine.analyzer import Analyzer
from engine.dependency import DependencyGraph
from engine.executor import Executor
from typing import Optional

app = FastAPI()
analyzer = Analyzer()

class RepoRequest(BaseModel):
    repo_path: str

@app.post("/analyze")
def analyze_repo(req: RepoRequest):
    analysis = analyzer.analyze_repo(req.repo_path)
    dep_graph = DependencyGraph()
    for file in analysis:
        dep_graph.add_dependency(file, None)  # placeholder
    execution_order = Executor(dep_graph.get_execution_order()).execution_order
    return {
        "analysis": analysis,
        "execution_order": execution_order
    }
