# examples/CLIAnalyzer/main.py
import sys
from engine.analyzer import Analyzer
from engine.dependency import DependencyGraph
from engine.executor import Executor

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <repo_path>")
        sys.exit(1)

    repo_path = sys.argv[1]

    analyzer = Analyzer()
    analysis = analyzer.analyze_repo(repo_path)

    dep_graph = DependencyGraph()
    for file in analysis:
        # Placeholder: no real dependency detection yet
        dep_graph.add_dependency(file, None)

    executor = Executor(dep_graph.get_execution_order())
    executor.run_simulation()
