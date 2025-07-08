# converts a yaml file to a json file
import yaml
import json
import os

def yaml_to_json(yaml_file, json_file):
    with open(yaml_file, 'r') as yf:
        yaml_content = yaml.safe_load(yf)
    
    with open(json_file, 'w') as jf:
        json.dump(yaml_content["x-code-generation"], jf, indent=4)

if __name__ == "__main__":
    yaml_file = 'yaml_sample.yaml'
    json_file = 'spec_test.json'
    
    if os.path.exists(yaml_file):
        yaml_to_json(yaml_file, json_file)
        print(f"Converted {yaml_file} to {json_file}")
    else:
        print(f"{yaml_file} does not exist.")
        print("Please ensure the YAML file is present in the specified path.")