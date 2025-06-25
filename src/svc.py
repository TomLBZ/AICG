import json
import logging
from typing import Dict, List, Optional, Union
from llm_intf import LLMInterface

from pydantic import BaseModel, Field

class CodeGenerationRequest(BaseModel):
    """Model for code generation requests."""
    specification: str = Field(..., description="Detailed description of what the code should do")
    language: str = Field(..., description="Target programming language")
    examples: Optional[List[Dict[str, str]]] = Field(None, description="Optional few-shot examples")
    max_tokens: int = Field(2000, description="Maximum response length")
    temperature: float = Field(0.2, description="Creativity parameter")

class CodeGenerationResult(BaseModel):
    """Model for code generation results."""
    code: str
    validation_results: Optional[Dict[str, Union[bool, str]]] = None

class CodeGenerationService:
    """Service for generating and validating code."""
    
    def __init__(self, llm_interface: LLMInterface, validator=None):
        """Initialize with an LLM interface and optional validator.
        
        Args:
            llm_interface: Implementation of LLMInterface
            validator: Optional code validator
        """
        self.llm = llm_interface
        self.validator = validator
        self.logger = logging.getLogger(__name__)
    
    async def generate_code(
        self, 
        request: CodeGenerationRequest
    ) -> CodeGenerationResult:
        """Generate code from a specification.
        
        Args:
            request: The code generation request
            
        Returns:
            Result containing generated code and validation results
        """
        self.logger.info(f"Generating {request.language} code")
        
        try:
            # Generate the code
            code = await self.llm.generate_code(
                specification=request.specification,
                language=request.language,
                examples=request.examples,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            # Extract code if it's wrapped in markdown code blocks
            code = self._extract_code_from_markdown(code, request.language)
            
            # Validate the code if a validator is available
            validation_results = None
            if self.validator:
                validation_results = await self.validator.validate(
                    code=code, 
                    language=request.language,
                    specification=request.specification
                )
            
            return CodeGenerationResult(
                code=code,
                validation_results=validation_results
            )
            
        except Exception as e:
            self.logger.error(f"Error generating code: {str(e)}")
            raise
    
    def _extract_code_from_markdown(self, text: str, language: str) -> str:
        """Extract code from markdown code blocks."""
        if f"```{language}" in text:
            # Try to extract code blocks with language identifier
            parts = text.split(f"```{language}")
            if len(parts) > 1:
                code_block = parts[1].split("```")[0]
                return code_block.strip()
        elif "```" in text:
            # Try to extract code blocks without language identifier
            parts = text.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        
        # If no code block markers, return the original text
        return text