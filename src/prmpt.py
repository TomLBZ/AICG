from typing import Dict, List, Optional

def create_code_generation_prompt(
    task_description: str,
    expected_inputs: Dict[str, str],
    expected_outputs: Dict[str, str],
    constraints: List[str],
    language: str,
    framework: Optional[str] = None
) -> str:
    """Create a structured prompt for code generation.
    
    Args:
        task_description: High-level description of what the code should do
        expected_inputs: Dictionary of input names and their descriptions
        expected_outputs: Dictionary of output names and their descriptions
        constraints: List of constraints the code must follow
        language: Target programming language
        framework: Optional framework to use
        
    Returns:
        A structured prompt string
    """
    prompt = f"# Task: {task_description}\n\n"
    
    # Add language and framework
    prompt += f"## Language: {language}\n"
    if framework:
        prompt += f"## Framework: {framework}\n"
    
    # Add inputs
    prompt += "\n## Inputs:\n"
    for name, desc in expected_inputs.items():
        prompt += f"- {name}: {desc}\n"
    
    # Add outputs
    prompt += "\n## Expected Outputs:\n"
    for name, desc in expected_outputs.items():
        prompt += f"- {name}: {desc}\n"
    
    # Add constraints
    prompt += "\n## Constraints:\n"
    for constraint in constraints:
        prompt += f"- {constraint}\n"
    
    # Final instruction
    prompt += "\n## Instructions:\n"
    prompt += "Generate code that satisfies the requirements above. Include comments to explain any complex logic.\n"
    prompt += "Do not include explanations outside the code. Return only the code itself.\n"
    
    return prompt

def add_examples_to_prompt(
    base_prompt: str,
    examples: List[Dict[str, str]]
) -> str:
    """Add few-shot examples to the base prompt.
    
    Args:
        base_prompt: The existing prompt
        examples: List of dictionaries with 'specification' and 'code' keys
        
    Returns:
        Enhanced prompt with examples
    """
    prompt = base_prompt + "\n\n## Examples:\n"
    
    for i, example in enumerate(examples, 1):
        prompt += f"\n### Example {i}:\n"
        prompt += f"Specification:\n{example['specification']}\n\n"
        prompt += f"Code:\n```\n{example['code']}\n```\n"
    
    prompt += "\n## Now, generate code for the current task:"
    
    return prompt