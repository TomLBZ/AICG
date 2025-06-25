import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from llm_intf import LLMInterface
from typing import Dict, List, Optional

class OpenAICodeGenerator(LLMInterface):
    """OpenAI implementation of the LLM interface."""
    
    def __init__(self, api_key: str, model: str = "gpt-4", base_url: str = "https://api.openai.com/v1"):
        """Initialize with API key and model.
        
        Args:
            api_key: OpenAI API key
            model: Model identifier to use
        """
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_code(
        self, 
        specification: str, 
        language: str, 
        examples: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.2,
    ) -> str:
        """Generate code using OpenAI API with retry logic for resilience."""
        # Construct the prompt
        prompt = self._construct_prompt(specification, language, examples)
        
        # Call the API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert programmer. Generate only code without explanations unless specifically requested."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        # Check for errors
        if response.choices[0].message.content:
            return response.choices[0].message.content
        return ""

    def _construct_prompt(
        self, 
        specification: str, 
        language: str, 
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Construct an effective prompt for code generation."""
        prompt = f"Generate {language} code for the following specification:\n\n{specification}\n\n"
        
        if examples:
            prompt += "Here are some examples of similar code:\n\n"
            for i, example in enumerate(examples):
                prompt += f"Example {i+1}:\nSpecification: {example['specification']}\nCode:\n{example['code']}\n\n"
        
        prompt += f"Now, generate only the {language} code that meets the specification."
        return prompt