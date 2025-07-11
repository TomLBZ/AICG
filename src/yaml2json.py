#!/home/lbz/venv/bin/python3
import yaml
import json
import traceback
import os

def multiline_input(prompt: str, verbose: bool = False) -> str:
    """Read multiline input from the user."""
    if (verbose):
        print(prompt)
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "\x04": # ASCII End of Transmission (EOT)
                break
            lines.append(line)
        except EOFError:
            raise
        except KeyboardInterrupt:
            raise
    return "\n".join(lines)

def yaml_to_json_string(yaml_string: str) -> str:
    result = ""
    error = ""
    try:
        yaml_content = yaml.safe_load(yaml_string)
        result =  json.dumps(yaml_content["x-code-generation"], indent=4)
    except yaml.YAMLError as e:
        error = f"[Error]\nYAML parsing error: {e}"
    except KeyError as e:
        error = f"[Error]\nKeyError: {e}. Ensure the YAML contains 'x-code-generation' key."
    except Exception as e:
        error = f"[Error]\nAn unexpected error occurred: {traceback.format_exc()}"
    return result if not error else error

def yaml_to_json(yaml_file, json_file):
    with open(yaml_file, 'r') as yf:
        yaml_content = yaml.safe_load(yf)
    
    with open(json_file, 'w') as jf:
        json.dump(yaml_content["x-code-generation"], jf, indent=4)

if __name__ == "__main__":
    # yaml_file = 'yaml_sample.yaml'
    # json_file = 'spec_test.json'
    
    # if os.path.exists(yaml_file):
    #     yaml_to_json(yaml_file, json_file)
    #     print(f"Converted {yaml_file} to {json_file}")
    # else:
    #     print(f"{yaml_file} does not exist.")
    #     print("Please ensure the YAML file is present in the specified path.")
    while True:
        try:
            input_prompt = "Enter the yaml content (end with EOT 0x04):"
            input_str = multiline_input(input_prompt, verbose=False)
            
            if not input_str.strip(): # check if input is empty
                print("[Error]\nNo input provided.")
                continue
            
            json_string = yaml_to_json_string(input_str)
            print(json_string)
        except EOFError:
            print("\n[Command]\nEnd of Service.")
            break
        except KeyboardInterrupt:
            print("\n[Command]\nKeyboard Interrupt. Exiting.")
            break