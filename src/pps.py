import ast
import re
from typing import List, Set

def extract_python_imports(code: str) -> Set[str]:
    """Extract imports used in Python code."""
    try:
        tree = ast.parse(code)
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
        
        return imports
    except:
        # Fallback to regex if parsing fails
        import_pattern = r'^import\s+(\w+)|^from\s+(\w+)'
        matches = re.findall(import_pattern, code, re.MULTILINE)
        return {m[0] or m[1] for m in matches}

def validate_imports(imports: Set[str]) -> List[str]:
    """Check if imports are installable."""
    missing = []
    for imp in imports:
        if imp and not imp.startswith('_'):
            try:
                __import__(imp)
            except ImportError:
                missing.append(imp)
    return missing