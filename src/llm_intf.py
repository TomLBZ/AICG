import abc
from typing import Dict, List, Optional

class LLMInterface(abc.ABC):
    """Abstract interface for LLM interactions."""
    
    @abc.abstractmethod
    async def generate_code(
        self, 
        specification: str, 
        language: str, 
        examples: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.2,
    ) -> str:
        """Generate code from a specification.
        
        Args:
            specification: The detailed requirements for the code
            language: Target programming language
            examples: Optional few-shot examples
            max_tokens: Maximum response length
            temperature: Creativity parameter (lower = more deterministic)
            
        Returns:
            Generated code as string
        """
        pass