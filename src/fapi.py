import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import our components
from oai_cgen import OpenAICodeGenerator
from svc import CodeGenerationService, CodeGenerationRequest, CodeGenerationResult
from validate import CodeValidator

# Create the FastAPI app
app = FastAPI(
    title="Code Generation API",
    description="API for generating code from specifications using LLMs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
llm = OpenAICodeGenerator(api_key=os.environ.get("OPENAI_API_KEY") or "")
validator = CodeValidator()
service = CodeGenerationService(llm_interface=llm, validator=validator)

@app.post("/generate", response_model=CodeGenerationResult)
async def generate_code(request: CodeGenerationRequest):
    """Generate code from a specification."""
    try:
        result = await service.generate_code(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))