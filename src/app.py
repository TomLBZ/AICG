import asyncio
from oai_cgen import OpenAICodeGenerator
from svc import CodeGenerationService, CodeGenerationRequest, CodeGenerationResult
from validate import CodeValidator
from cb_gen import CodebaseReader, CodeModificationService
import os

BASE_URL = "https://ai.stdev.remoteblossom.com/engines/v1"
MODEL = "ai/phi4"
KEY = "IGNORED"
CURRENT_DIR = "."

llm = OpenAICodeGenerator(api_key=KEY, model=MODEL, base_url=BASE_URL)
validator = CodeValidator()
service = CodeGenerationService(llm_interface=llm, validator=validator)
codebase_reader = CodebaseReader(root_dir=CURRENT_DIR)
code_modification_service = CodeModificationService(llm_interface=llm, codebase_reader=codebase_reader)

async def generate_code(request: CodeGenerationRequest) -> CodeGenerationResult:
    """Generate code from a specification."""
    try:
        result = await service.generate_code(request)
        return result
    except Exception as e:
        raise RuntimeError(f"Code generation failed: {str(e)}")
    
async def print_code_generation(request: CodeGenerationRequest) -> str:
    """Print the generated code and validation results."""
    result = await generate_code(request)
    print("Generated Code:")
    print(result.code)
    if result.validation_results:
        print("Validation Results:")
        for key, value in result.validation_results.items():
            print(f"{key}: {value}")
    else:
        print("No validation results available.")
    return result.code
    
async def modify_codebase(fname: str, modification: str) -> str:
    """Modify the codebase based on the generated code."""
    try:
        modifications = await code_modification_service.generate_modification(fname, modification, True)
        print("Codebase modified successfully.")
        return modifications
    except Exception as e:
        raise RuntimeError(f"Code modification failed: {str(e)}")
    
async def print_code_modification(fname: str, modification: str) -> str:
    """Print the modifications made to the codebase."""
    modifications = await modify_codebase(fname, modification)
    print("Modifications:")
    print(modifications)
    return modifications

async def complete_test():
    """Run a complete test of code generation and modification."""
    request = CodeGenerationRequest(
        specification="Create a function that adds two numbers",
        language="python",
        examples=[],
        max_tokens=2000,
        temperature=0.2
    )
    fname = "add.py"
    modification = "rename variables to x and y"
    print("Starting code generation...")
    code = await print_code_generation(request)
    # write the generated code to a file
    with open(fname, "w") as f:
        f.write(code)
    print("Starting code modification...")
    modified_code = await print_code_modification(fname, modification)
    # write the modified code to a file
    with open(fname, "w") as f:
        f.write(modified_code)

if __name__ == "__main__":
    asyncio.run(complete_test())