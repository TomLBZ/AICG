import asyncio
from oai_cgen import OpenAICodeGenerator
from svc import CodeGenerationService, CodeGenerationRequest, CodeGenerationResult
from validate import CodeValidator

BASE_URL = "https://ai.stdev.remoteblossom.com/engines/v1"
MODEL = "ai/phi4"
KEY = "IGNORED"

llm = OpenAICodeGenerator(api_key=KEY, model=MODEL, base_url=BASE_URL)
validator = CodeValidator()
service = CodeGenerationService(llm_interface=llm, validator=validator)

async def generate_code(request: CodeGenerationRequest) -> CodeGenerationResult:
    """Generate code from a specification."""
    try:
        result = await service.generate_code(request)
        return result
    except Exception as e:
        raise RuntimeError(f"Code generation failed: {str(e)}")
    
async def print_code_generation(request: CodeGenerationRequest):
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
    
if __name__ == "__main__":
    # Example usage
    request = CodeGenerationRequest(
        specification="Create a function that adds two numbers",
        language="python",
        examples=[],
        max_tokens=2000,
        temperature=0.2
    )
    
    # Note: run inside async context
    asyncio.run(print_code_generation(request))