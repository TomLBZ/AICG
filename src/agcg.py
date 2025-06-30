#!/home/lbz/venv/bin/python3
import abc
import openai
import logging
import asyncio
import subprocess
import ast
import re
import os
import glob

from enum import StrEnum, auto
from typing import Dict, List, Optional, Set
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

class LanguageEnum(StrEnum):
    """Enumeration for supported programming languages."""
    PYTHON = auto()
    JAVASCRIPT = auto()
    CPP = auto()
    NONE = auto()

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if the value is a valid language."""
        return value in cls._value2member_map_

class CodeGenerationSpecification(BaseModel):
    """Model for code generation specifications."""
    task_description: str = Field(..., description="High-level description of what the code should do")
    language: LanguageEnum = Field(..., description="Target programming language")
    framework: Optional[str] = Field(None, description="Optional framework to use (e.g., Django, Flask)")
    expected_inputs: Dict[str, str] = Field(..., description="Dictionary of input names and their descriptions")
    expected_outputs: Dict[str, str] = Field(..., description="Dictionary of output names and their descriptions")
    constraints: List[str] = Field(..., description="List of constraints the code must follow")

    def __repr__(self) -> str:
        """String representation of the specification."""
        return (
            f"CodeGenerationSpecification(task_description={self.task_description}, "
            f"language={self.language}, framework={self.framework}, "
            f"expected_inputs={self.expected_inputs}, "
            f"expected_outputs={self.expected_outputs}, "
            f"constraints={self.constraints})"
        )

    def __str__(self) -> str:
        """Structured string representation of the specification."""
        prompt = f"# Task: {self.task_description}\n\n"
        # Add language and framework
        prompt += f"## Language: {self.language}\n"
        if self.framework:
            prompt += f"## Framework: {self.framework}\n"
        # Add inputs (if length of inputs is 0, we don't add the section)
        if self.expected_inputs:
            prompt += "\n## Inputs:\n"
            for name, desc in self.expected_inputs.items():
                prompt += f"- {name}: {desc}\n"
        # Add outputs (if length of outputs is 0, we don't add the section)
        if self.expected_outputs:
            prompt += "\n## Expected Outputs:\n"
            for name, desc in self.expected_outputs.items():
                prompt += f"- {name}: {desc}\n"
        # Add constraints (if length of constraints is 0, we don't add the section)
        if self.constraints:
            prompt += "\n## Constraints:\n"
            for constraint in self.constraints:
                prompt += f"- {constraint}\n"
        return prompt

class CodeGenerationExample(BaseModel):
    """Model for code generation examples."""
    specification: CodeGenerationSpecification = Field(..., description="Specification of the code to generate")
    code: str = Field(..., description="Example code that satisfies the specification")

class CodeGenerationRequest(BaseModel):
    """Model for code generation requests."""
    specification: CodeGenerationSpecification = Field(..., description="Detailed description of what the code should do")
    examples: Optional[List[CodeGenerationExample]] = Field(None, description="Optional few-shot examples")
    max_tokens: int = Field(2000, description="Maximum response length")
    temperature: float = Field(0.2, description="Creativity parameter")

class CodeValidationResult(BaseModel):
    """Model for code validation results."""
    valid: bool = Field(..., description="Whether the code syntax is valid")
    message: str = Field(..., description="Validation message, e.g., error details or success message")

    def __repr__(self) -> str:
        """String representation of the validation result."""
        return f"CodeValidationResult(valid={self.valid}, message='{self.message}')"
    
    def __str__(self) -> str:
        """Structured string representation of the validation result."""
        return f"Validation Result: {'Valid' if self.valid else 'Invalid'}\nMessage: {self.message if self.message else 'None'}"

class CodeGenerationResult(BaseModel):
    """Model for code generation results."""
    code: str
    validation_results: Optional[CodeValidationResult] = None

class LLMInterface(abc.ABC):
    """Abstract interface for LLM interactions."""
    
    @abc.abstractmethod
    async def generate_code(self, request: CodeGenerationRequest) -> str:
        """Generate code from a specification.
        
        Args:
            code_generation_request: The detailed requirements for the code
            
        Returns:
            Generated code as string
        """
        pass

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
    async def generate_code(self, request: CodeGenerationRequest) -> str:
        """Generate code using OpenAI API with retry logic for resilience."""
        # Construct the prompt
        prompt = self._construct_prompt(request.specification, request.examples)

        # Call the API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert programmer. Generate only code without explanations unless specifically requested."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        # Check for errors
        if response.choices[0].message.content:
            return response.choices[0].message.content
        return ""
    
    def _add_examples_to_prompt(self, base_prompt: str, examples: List[CodeGenerationExample]) -> str:
        """Add few-shot examples to the base prompt.
        
        Args:
            base_prompt: The existing prompt
            examples: List of CodeGenerationExample instances
            
        Returns:
            Enhanced prompt with examples
        """
        prompt = base_prompt + "\n\n## Examples:\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"\n### Example {i}:\n"
            prompt += f"Specification:\n{example.specification}\n\n"
            prompt += f"Code:\n```{example.specification.language}\n{example.code}\n```\n"

        prompt += "\n## Now, generate code for the current task:"
        
        return prompt

    def _construct_prompt(self, specification: CodeGenerationSpecification, examples: Optional[List[CodeGenerationExample]] = None) -> str:
        """Create a structured prompt for code generation.
        
        Args:
            specification: The detailed code generation specification
            examples: Optional few-shot examples
            
        Returns:
            A structured prompt string
        """
        prompt = str(specification)
        
        # Final instruction
        prompt += "\n## Instructions:\n"
        prompt += "Generate code that satisfies the requirements above. Include comments to explain any complex logic.\n"
        prompt += "Do not include explanations outside the code. Return only the code itself.\n"
        
        # Add examples if provided
        if examples:
            prompt = self._add_examples_to_prompt(prompt, examples)

        return prompt

class CodeValidator:
    """Validates generated code for syntax and basic functionality."""
    
    async def validate(self, code: str, language: LanguageEnum) -> CodeValidationResult:
        """Validate the generated code.
        
        Args:
            code: The generated code to validate
            language: The programming language
            
        Returns:
            Dictionary with validation results
        """
        results = CodeValidationResult(
            valid=False,
            message=""
        )

        if not LanguageEnum.is_valid(language):
            results.message = f"Unknown language: {language}"
            return results

        if language == LanguageEnum.PYTHON:
            return await self._validate_python(code)
        else:
            results.message = f"Validation not implemented for {language}"
            return results
    
    def _extract_python_imports(self, code: str) -> Set[str]:
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

    def _validate_python_imports(self, imports: Set[str]) -> List[str]:
        """Check if imports are installable."""
        missing = []
        for imp in imports:
            if imp and not imp.startswith('_'):
                try:
                    __import__(imp)
                except ImportError:
                    missing.append(imp)
        return missing
    
    async def _validate_python(self, code: str) -> CodeValidationResult:
        """Validate Python code syntax."""
        results = CodeValidationResult(
            valid=False,
            message=""
        )

        # Check for imports
        imports = self._extract_python_imports(code)
        if imports:
            missing_imports = self._validate_python_imports(imports)
            if missing_imports:
                results.message = f"Missing imports: {', '.join(missing_imports)}"
                return results
        
        # Create a temporary file
        with open("temp_code.py", "w") as f:
            f.write(code)
        
        # Check syntax
        process = await asyncio.create_subprocess_shell(
            "python -m py_compile temp_code.py",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            results.valid = True
            results.message = "Code syntax is valid."
        else:
            error_message = stderr.decode()
            results.message = f"Syntax error: {error_message}"

        # Clean up
        subprocess.run(["rm", "temp_code.py"])
        
        return results
    
class CodeGenerationService:
    """Service for generating and validating code."""
    
    def __init__(self, llm_interface: LLMInterface, validator: Optional[CodeValidator]=None):
        """Initialize with an LLM interface and optional validator.
        
        Args:
            llm_interface: Implementation of LLMInterface
            validator: Optional code validator
        """
        self.llm = llm_interface
        self.validator = validator
        self.logger = logging.getLogger(__name__)
    
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResult:
        """Generate code from a specification.
        
        Args:
            request: The code generation request
            
        Returns:
            Result containing generated code and validation results
        """
        language = request.specification.language
        self.logger.info(f"Generating {language} code")

        try:
            # Generate the code
            code = await self.llm.generate_code(request)
            
            # Extract code if it's wrapped in markdown code blocks
            code = self._extract_code_from_markdown(code, language)

            # Validate the code if a validator is available
            validation_results = None
            if self.validator:
                validation_results = await self.validator.validate(code, language)

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

class CodebaseReader:
    """Utility for reading and understanding codebases."""
    
    def __init__(self, root_dir: str):
        """Initialize with the root directory of the codebase.
        
        Args:
            root_dir: Root directory of the codebase
        """
        self.root_dir = root_dir
        
    def get_file_content(self, file_path: str) -> str:
        """Get the content of a specific file."""
        with open(os.path.join(self.root_dir, file_path), 'r') as f:
            return f.read()
    
    def _find_related_python_files(self, target_file: str, max_files: int = 5) -> List[str]:
        """Find files that are likely related to the target file.
        
        Uses import statements and references to identify related files.
        
        Args:
            target_file: The file we want to modify
            max_files: Maximum number of related files to return
            
        Returns:
            List of related file paths
        """
        # This is a simplified implementation
        # A real implementation would parse imports, analyze dependencies, etc.
        related_files = []
        target_content = self.get_file_content(target_file)
        
        # Get all Python files in the codebase
        all_files = glob.glob(f"{self.root_dir}/**/*.py", recursive=True)
        
        # Extract potential module names from the target file
        module_names = self._extract_python_module_names(target_content)
        
        # Find files that might be related based on imports
        for file_path in all_files:
            rel_path = os.path.relpath(file_path, self.root_dir)
            if rel_path == target_file:
                continue  # Skip the target file itself
                
            # Check if this file is imported in the target file
            file_basename = os.path.basename(file_path).replace('.py', '')
            if any(mn == file_basename for mn in module_names):
                related_files.append(rel_path)
                
            # Check if the target file is imported in this file
            if len(related_files) < max_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        target_basename = os.path.basename(target_file).replace('.py', '')
                        if target_basename in self._extract_python_module_names(content):
                            related_files.append(rel_path)
                except:
                    pass  # Skip files that can't be read
            
            if len(related_files) >= max_files:
                break
                
        return related_files[:max_files]
    
    def _extract_python_module_names(self, content: str) -> Set[str]:
        """Extract module names from import statements."""
        import_pattern = r'import\s+([\w\.]+)|from\s+([\w\.]+)\s+import'
        import_matches = re.findall(import_pattern, content)
        
        module_names = set()
        for match in import_matches:
            module_path = match[0] or match[1]
            # Get the base module name
            module_names.add(module_path.split('.')[0])
            
        return module_names

    def find_related_files(self, target_file: str, language: LanguageEnum, max_files: int = 5) -> List[str]:
        """Find files related to the target file based on language."""
        if not LanguageEnum.is_valid(language):
            raise ValueError(f"Unsupported language: {language}")
        if language == LanguageEnum.PYTHON:
            return self._find_related_python_files(target_file, max_files)
        else:
            # For other languages, we can implement similar logic
            raise NotImplementedError(f"Finding related files for {language} is not implemented.")

class CodeModificationService:
    """Service for generating modifications to existing code."""
    
    def __init__(self, llm_interface: LLMInterface, codebase_reader: CodebaseReader, validator: Optional[CodeValidator] = None):
        """Initialize with LLM interface and codebase reader.
        
        Args:
            llm_interface: Implementation of LLMInterface
            codebase_reader: Codebase reader instance
        """
        self.llm = llm_interface
        self.codebase = codebase_reader
        self.validator = validator
        self.logger = logging.getLogger(__name__)

    def _construct_modification_prompt(self, target_file: str, modification_instruction: str, language: LanguageEnum, include_related_files: bool) -> str:
        """Construct a prompt for modifying an existing file.
        
        Args:
            target_file: Path to the file to modify
            modification_instruction: Specification of the required modification
            language: Programming language of the target file
            include_related_files: Whether to include related files as context

        Returns:
            A string prompt for the modification request.
        """
        # Get the content of the target file
        target_content = self.codebase.get_file_content(target_file)

        # Construct the prompt
        prompt = f"You are tasked with modifying the following file:\n\n"
        prompt += f"File: {target_file}\n\n"
        prompt += f"```{language}\n{target_content}\n```\n\n"

        # Add related files as context if requested
        if include_related_files:
            related_files = self.codebase.find_related_files(target_file, language)
            if related_files:
                prompt += "Here are some related files that might provide context:\n\n"
                for rel_file in related_files:
                    rel_content = self.codebase.get_file_content(rel_file)
                    # Summarize long files to avoid token limit issues
                    if len(rel_content) > 1000:
                        rel_content = rel_content[:1000] + "\n# ... (file continues)"
                    prompt += f"File: {rel_file}\n```{language}\n{rel_content}\n```\n\n"

        # Add the modification specification
        prompt += f"Modification Required:\n{modification_instruction}\n\n"
        prompt += "Please provide the complete modified version of the target file only. "
        prompt += "Maintain the same style, imports, and formatting as the original code. "
        prompt += "Add comments explaining your changes."

        return prompt

    async def generate_modification(
        self,
        language: LanguageEnum,
        target_file: str,
        modification_instruction: str,
        include_related_files: bool = True,
        skip_validation: bool = False
    ) -> CodeGenerationResult:
        """Generate a modification to an existing file.
        
        Args:
            language: Programming language of the target file
            target_file: Path to the file to modify
            modification_instruction: Instruction of the required modification
            include_related_files: Whether to include related files as context
            skip_validation: Whether to skip validation of the modified code
        Returns:
            Modified version of the target file
        """
        # Validate inputs
        if not LanguageEnum.is_valid(language):
            raise ValueError(f"Unsupported language: {language}")
        if not os.path.exists(os.path.join(self.codebase.root_dir, target_file)):
            raise FileNotFoundError(f"Target file {target_file} does not exist in the codebase.")
        
        # Construct the modification prompt
        prompt = self._construct_modification_prompt(
            target_file=target_file,
            modification_instruction=modification_instruction,
            language=language,
            include_related_files=include_related_files
        )
        
        # Generate the modified code
        specification = CodeGenerationSpecification(
            task_description=prompt,
            language=language,
            framework=None,
            expected_inputs={},
            expected_outputs={},
            constraints=[]
        )

        request = CodeGenerationRequest(
            specification=specification,
            examples=None,  # No examples for modifications
            max_tokens=2000,
            temperature=0.2
        )

        try:
            modified_code = await self.llm.generate_code(request)
            
            # Extract the code if it's wrapped in markdown
            code = self._extract_code_from_markdown(modified_code, language)

            # Validate the modified code if a validator is available
            validation_results = None
            if self.validator and not skip_validation:
                validation_results = await self.validator.validate(code, language)

            return CodeGenerationResult(
                code=code,
                validation_results=validation_results
            )
        except Exception as e:
            self.logger.error(f"Error generating modification: {str(e)}")
            raise

    def _extract_code_from_markdown(self, text: str, language: LanguageEnum) -> str:
        """Extract code from markdown code blocks."""
        if f"```{language}" in text:
            parts = text.split(f"```{language}")
            if len(parts) > 1:
                code_block = parts[1].split("```")[0]
                return code_block.strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        return text
    
    async def handle_large_file_modification(
        self,
        language: LanguageEnum,
        target_file: str,
        modification_instruction: str,
        max_chunk_size: int = 4000
    ) -> str:
        """Handle modifications to files that exceed the context window.
        
        Args:
            target_file: Path to the file to modify
            modification_spec: Specification of the required modification
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            Modified version of the target file
        """
        target_content = self.codebase.get_file_content(target_file)
        
        # If the file is small enough, use the standard approach
        if len(target_content) <= max_chunk_size:
            modified_result = await self.generate_modification(
                language, 
                target_file, 
                modification_instruction,
                skip_validation=True
            ) # Skip validation because we are handling large files differently (by sections)
            return modified_result.code

        # For larger files, we need a different strategy
        # First, try to identify the relevant sections
        prompt = f"I have a large {language} file ({len(target_content)} characters) that needs modification:\n\n"
        prompt += f"File: {target_file}\n\n"
        prompt += f"The first 1000 characters of the file are:\n```{language}\n{target_content[:1000]}\n```\n\n"
        prompt += f"The modification required is: {modification_instruction}\n\n"
        prompt += "Which sections or classes/functions of the file will likely need to be modified? "
        prompt += "Provide specific function names, class names, or line number ranges if possible."
        
        sections_specification = CodeGenerationSpecification(
            task_description=prompt,
            language=LanguageEnum.NONE,  # We're not generating code yet, just asking for sections
            framework=None,
            expected_inputs={},
            expected_outputs={},
            constraints=[]
        )

        sections_request = CodeGenerationRequest(
            specification=sections_specification,
            examples=None,  # No examples for this task
            max_tokens=2000,
            temperature=0.1
        )

        # Ask the LLM which parts to focus on
        sections_response = await self.llm.generate_code(sections_request)
        
        # Parse the response to identify key sections
        # This is a simplified approach; a real system would be more sophisticated
        relevant_sections = self._extract_relevant_sections(target_content, sections_response)
        
        # Create a new prompt with just the relevant sections
        modification_prompt = f"You need to modify specific parts of the file: {target_file}\n\n"
        modification_prompt += "Here are the relevant sections:\n\n"
        
        for section_name, section_content in relevant_sections.items():
            modification_prompt += f"Section: {section_name}\n```python\n{section_content}\n```\n\n"

        modification_prompt += f"The full modification required is: {modification_instruction}\n\n"
        modification_prompt += "Please provide the complete modified versions of ONLY the sections shown above. "
        modification_prompt += "Maintain the same style as the original code."
        
        modification_specification = CodeGenerationSpecification(
            task_description=modification_prompt,
            language=language,
            framework=None,
            expected_inputs={},
            expected_outputs={},
            constraints=[]
        )

        modification_request = CodeGenerationRequest(
            specification=modification_specification,
            examples=None,  # No examples for modifications
            max_tokens=2000,
            temperature=0.2
        )

        # Generate modifications for the relevant sections
        modified_sections_response = await self.llm.generate_code(modification_request)
        
        # Extract and apply the modifications to the original file
        return self._apply_section_modifications(
            target_content, 
            relevant_sections,
            modified_sections_response
        )
    
    def _extract_relevant_sections(self, content: str, response: str) -> Dict[str, str]:
        """Extract relevant sections based on LLM response.
        
        Args:
            content: Full content of the target file
            response: LLM response indicating which sections to modify
            
        Returns:
            Dictionary mapping section names to their content
        """
        # This is a simplified implementation
        # A real implementation would parse the response more intelligently
        sections = {}
        lines = response.splitlines()
        
        for line in lines:
            if line.strip():
                # Assume each line is a section name or identifier
                section_name = line.strip()
                # Extract the corresponding section from the content
                section_content = self._extract_section(content, section_name)
                if section_content:
                    sections[section_name] = section_content
        
        return sections
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from the content based on its name.
        
        Args:
            content: Full content of the target file
            section_name: Name of the section to extract
            
        Returns:
            Content of the specified section, or an empty string if not found
        """
        # This is a simplified implementation
        # A real implementation would parse the content more intelligently
        pattern = re.compile(rf"def {section_name}\s*\(.*?\):.*?(?=\n\n|$)", re.DOTALL)
        match = pattern.search(content)
        return match.group(0) if match else ""
    
    def _apply_section_modifications(
        self, 
        original_content: str, 
        relevant_sections: Dict[str, str], 
        modified_response: str
    ) -> str:
        """Apply modifications to the original content based on modified sections.
        
        Args:
            original_content: Original content of the target file
            relevant_sections: Sections that were identified as relevant
            modified_response: LLM response with modified sections
            
        Returns:
            The complete modified version of the target file
        """
        # This is a simplified implementation
        # A real implementation would handle merging more intelligently
        for section_name, section_content in relevant_sections.items():
            # extract the modified section from the response
            modified_section = self._extract_section(modified_response, section_name)
            # Replace the original section with the modified one
            original_content = original_content.replace(section_content, modified_section)

        return original_content.strip()
    
# Example usage:
BASE_URL = "https://ai.stdev.remoteblossom.com/engines/v1"
MODEL = "ai/phi4"
KEY = "IGNORED"
llm = OpenAICodeGenerator(api_key=KEY, model=MODEL, base_url=BASE_URL)
validator = CodeValidator()
service = CodeGenerationService(llm_interface=llm, validator=validator)

def multiline_input(prompt: str) -> str:
    """Read multiline input from the user."""
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting multiline input.")
            break
    return "\n".join(lines)

def input_to_specification(input: str) -> CodeGenerationSpecification:
    """Convert input string (json format) to CodeGenerationSpecification."""
    return CodeGenerationSpecification.model_validate_json(input)

async def generate(input: str) -> str:
    """Generate code from prompt."""
    specification = input_to_specification(input)
    request = CodeGenerationRequest(
        specification=specification,
        examples=[],
        max_tokens=2000,
        temperature=0.2
    )
    res = await service.generate_code(request)
    code, validation_results = res.code, res.validation_results
    validation_str = f"------- [Validation] -------\n{validation_results if validation_results else 'Not available'}"
    code_str = f"------- [Code] -------\n{code}"
    return f"{validation_str}\n{code_str}"

if __name__ == "__main__":
    while True:
        try:
            # Read multiline input from the user
            input_str = multiline_input("Enter the code generation specification in JSON format (newline to finish):")
            if not input_str.strip():
                print("No input provided.")
                continue
            # Run the async generate function
            result = asyncio.run(generate(input_str))
            print(result)
        except EOFError:
            print("\nExiting dependency service.")
            break
        except KeyboardInterrupt:
            print("\nExiting dependency service.")
            break