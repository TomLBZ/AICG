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
code_modification_service = CodeModificationService(llm_interface=llm, codebase_reader=codebase_reader, validator=validator)

async def test_code_generation(request: CodeGenerationRequest) -> str:
    """Print the generated code and validation results."""
    result: CodeGenerationResult = await service.generate_code(request)
    print("==== Generation ====")
    print(result.code)
    if result.validation_results:
        print("==== Validation ====")
        print(result.validation_results)
    else:
        print("No validation results available.")
    return result.code

async def test_code_modification(fname: str, modification: str, language: LanguageEnum) -> str:
    """Print the modifications made to the codebase."""
    modifications: CodeGenerationResult = await code_modification_service.generate_modification(language, fname, modification, True)
    print("==== Modification ====")
    print(modifications.code)
    if modifications.validation_results:
        print("==== Validation ====")
        print(modifications.validation_results)
    else:
        print("No validation results available.")
    return modifications.code

async def complete_test():
    """Run a complete test of code generation and modification."""
    specification = CodeGenerationSpecification(
        task_description="Convert angles from degrees to radians.",
        function_signature=None,
        language=LanguageEnum.PYTHON,
        framework=None,
        expected_inputs={"x": "float value in degrees"},
        expected_outputs={"radians": "float value in radians"},
        constraints=[]
    )
    request = CodeGenerationRequest(
        specification=specification,
        examples=[],
        max_tokens=2000,
        temperature=0.2
    )
    gen_fname = "tmp/generated.py"
    mod_fname = "tmp/modified.py"
    modification = "make the input have 3 parameters: x, y and z and convert all of them to radians, write comments to describe the parameters"
    print("[GENERATION] Starting...")
    code = await test_code_generation(request)
    # write the generated code to a file
    with open(gen_fname, "w") as f:
        f.write(code)
    print("[GENERATION] Completed. Code written to", gen_fname)
    print("[MODIFICATION] Starting...")
    modified_code = await test_code_modification(gen_fname, modification, specification.language)
    # write the modified code to a file
    with open(mod_fname, "w") as f:
        f.write(modified_code)
    print("[MODIFICATION] Completed. Modified code written to", mod_fname)

if __name__ == "__main__":
    asyncio.run(complete_test())