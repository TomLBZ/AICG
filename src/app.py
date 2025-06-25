import asyncio
from agcg import LanguageEnum, OpenAICodeGenerator, CodeGenerationService, \
    CodeGenerationSpecification, CodeGenerationRequest, CodeGenerationResult, \
    CodeValidator, CodebaseReader, CodeModificationService

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
    print("==== Generation ====\n")
    print(result.code)
    if result.validation_results:
        print("==== Validation ====\n")
        print(result.validation_results)
    else:
        print("No validation results available.")
    return result.code

async def modify_codebase(fname: str, modification: str, language: LanguageEnum) -> str:
    """Modify the codebase based on the generated code."""
    try:
        modifications = await code_modification_service.generate_modification(language, fname, modification, True)
        print("Codebase modified successfully.")
        return modifications
    except Exception as e:
        raise RuntimeError(f"Code modification failed: {str(e)}")
    
async def print_code_modification(fname: str, modification: str, language: LanguageEnum) -> str:
    """Print the modifications made to the codebase."""
    modifications = await modify_codebase(fname, modification, language)
    print("Modifications:")
    print(modifications)
    return modifications

async def complete_test():
    """Run a complete test of code generation and modification."""
    specification = CodeGenerationSpecification(
        task_description="Create a function that adds two numbers",
        language=LanguageEnum.PYTHON,
        framework=None,
        expected_inputs={"a": "int", "b": "int"},
        expected_outputs={"result": "int"},
        constraints=[]
    )
    request = CodeGenerationRequest(
        specification=specification,
        examples=[],
        max_tokens=2000,
        temperature=0.2
    )
    gen_fname = "generated.py"
    mod_fname = "modified.py"
    modification = "rename variables a and b to x and y"
    print("Starting code generation...")
    code = await print_code_generation(request)
    # write the generated code to a file
    with open(gen_fname, "w") as f:
        f.write(code)
    print("Starting code modification...")
    modified_code = await print_code_modification(gen_fname, modification, specification.language)
    # write the modified code to a file
    with open(mod_fname, "w") as f:
        f.write(modified_code)

if __name__ == "__main__":
    asyncio.run(complete_test())