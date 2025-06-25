import os
import glob
from typing import Dict, List, Set, Tuple
import re
from llm_intf import LLMInterface

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
    
    def find_related_files(self, target_file: str, max_files: int = 5) -> List[str]:
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
        module_names = self._extract_module_names(target_content)
        
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
                        if target_basename in self._extract_module_names(content):
                            related_files.append(rel_path)
                except:
                    pass  # Skip files that can't be read
            
            if len(related_files) >= max_files:
                break
                
        return related_files[:max_files]
    
    def _extract_module_names(self, content: str) -> Set[str]:
        """Extract module names from import statements."""
        import_pattern = r'import\s+([\w\.]+)|from\s+([\w\.]+)\s+import'
        import_matches = re.findall(import_pattern, content)
        
        module_names = set()
        for match in import_matches:
            module_path = match[0] or match[1]
            # Get the base module name
            module_names.add(module_path.split('.')[0])
            
        return module_names

class CodeModificationService:
    """Service for generating modifications to existing code."""
    
    def __init__(self, llm_interface: LLMInterface, codebase_reader: CodebaseReader):
        """Initialize with LLM interface and codebase reader.
        
        Args:
            llm_interface: Implementation of LLMInterface
            codebase_reader: Codebase reader instance
        """
        self.llm = llm_interface
        self.codebase = codebase_reader
        
    async def generate_modification(
        self,
        target_file: str,
        modification_spec: str,
        include_related_files: bool = True
    ) -> str:
        """Generate a modification to an existing file.
        
        Args:
            target_file: Path to the file to modify
            modification_spec: Specification of the required modification
            include_related_files: Whether to include related files as context
            
        Returns:
            Modified version of the target file
        """
        # Get the content of the target file
        target_content = self.codebase.get_file_content(target_file)
        
        # Construct the prompt
        prompt = f"You are tasked with modifying the following file:\n\n"
        prompt += f"File: {target_file}\n\n"
        prompt += f"```python\n{target_content}\n```\n\n"
        
        # Add related files as context if requested
        if include_related_files:
            related_files = self.codebase.find_related_files(target_file)
            if related_files:
                prompt += "Here are some related files that might provide context:\n\n"
                for rel_file in related_files:
                    rel_content = self.codebase.get_file_content(rel_file)
                    # Summarize long files to avoid token limit issues
                    if len(rel_content) > 1000:
                        rel_content = rel_content[:1000] + "\n# ... (file continues)"
                    prompt += f"File: {rel_file}\n```python\n{rel_content}\n```\n\n"
        
        # Add the modification specification
        prompt += f"Modification Required:\n{modification_spec}\n\n"
        prompt += "Please provide the complete modified version of the target file only. " 
        prompt += "Maintain the same style, imports, and formatting as the original code. "
        prompt += "Add comments explaining your changes."
        
        # Generate the modified code
        modified_code = await self.llm.generate_code(
            specification=prompt,
            language="python",
            temperature=0.2
        )
        
        # Extract the code if it's wrapped in markdown
        return self._extract_code_from_markdown(modified_code)
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        if "```python" in text:
            parts = text.split("```python")
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
        target_file: str,
        modification_spec: str,
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
            return await self.generate_modification(target_file, modification_spec)
        
        # For larger files, we need a different strategy
        # First, try to identify the relevant sections
        prompt = f"I have a large Python file ({len(target_content)} characters) that needs modification:\n\n"
        prompt += f"File: {target_file}\n\n"
        prompt += f"The first 1000 characters of the file are:\n```python\n{target_content[:1000]}\n```\n\n"
        prompt += f"The modification required is: {modification_spec}\n\n"
        prompt += "Which sections or classes/functions of the file will likely need to be modified? "
        prompt += "Provide specific function names, class names, or line number ranges if possible."
        
        # Ask the LLM which parts to focus on
        sections_response = await self.llm.generate_code(
            specification=prompt,
            language="text",  # We're not generating code yet
            temperature=0.1
        )
        
        # Parse the response to identify key sections
        # This is a simplified approach; a real system would be more sophisticated
        relevant_sections = self._extract_relevant_sections(target_content, sections_response)
        
        # Create a new prompt with just the relevant sections
        modification_prompt = f"You need to modify specific parts of the file: {target_file}\n\n"
        modification_prompt += "Here are the relevant sections:\n\n"
        
        for section_name, section_content in relevant_sections.items():
            modification_prompt += f"Section: {section_name}\n```python\n{section_content}\n```\n\n"
        
        modification_prompt += f"The full modification required is: {modification_spec}\n\n"
        modification_prompt += "Please provide the complete modified versions of ONLY the sections shown above. "
        modification_prompt += "Maintain the same style as the original code."
        
        # Generate modifications for the relevant sections
        modified_sections_response = await self.llm.generate_code(
            specification=modification_prompt,
            language="python",
            temperature=0.2
        )
        
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
            # Replace the original section with the modified one
            original_content = original_content.replace(section_content, modified_response)
        
        return original_content.strip()