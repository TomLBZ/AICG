import asyncio
import subprocess
from typing import Dict, Union

class CodeValidator:
    """Validates generated code for syntax and basic functionality."""
    
    async def validate(
        self, 
        code: str, 
        language: str,
        specification: str = ""
    ) -> Dict[str, Union[bool, str]]:
        """Validate the generated code.
        
        Args:
            code: The generated code to validate
            language: The programming language
            specification: Original specification for context
            
        Returns:
            Dictionary with validation results
        """
        results = {"syntax_valid": False, "message": ""}
        
        if language.lower() == "python":
            return await self._validate_python(code)
        elif language.lower() in ["javascript", "js"]:
            return await self._validate_javascript(code)
        else:
            results["message"] = f"Validation not implemented for {language}"
            return results
    
    async def _validate_python(self, code: str) -> Dict[str, Union[bool, str]]:
        """Validate Python code syntax."""
        results = {"syntax_valid": False, "message": ""}
        
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
            results["syntax_valid"] = True
            results["message"] = "Code syntax is valid."
        else:
            error_message = stderr.decode()
            results["message"] = f"Syntax error: {error_message}"
        
        # Clean up
        subprocess.run(["rm", "temp_code.py"])
        
        return results
    
    async def _validate_javascript(self, code: str) -> Dict[str, Union[bool, str]]:
        """Validate JavaScript code syntax."""
        # Similar implementation for JavaScript
        # ...
        return {"syntax_valid": False, "message": "JavaScript validation not implemented yet."}