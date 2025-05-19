import json
import sys
import os
import glob

def process_file(input_path):
    # Generate output path by adding _processed before .json
    output_path = os.path.splitext(input_path)[0] + '_processed.json'

    # Read the input file
    with open(input_path, 'r') as f:
        content = f.read()

    # Replace single quotes with double quotes
    content = content.replace("'", '"')

    # Remove all { and }
    content = content.replace('{', '')
    content = content.replace('}', '')

    # Remove array( and )} and add comma
    content = content.replace('array(', '')
    content = content.replace(')', ',')

    # Remove the last comma
    content = content.rstrip(',\n')

    # Wrap in dict
    content = '{' + content + '}'

    # Write to new processed file
    with open(output_path, 'w') as f:
        f.write(content)

    # Remove the initial file
    os.remove(input_path)
def main():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all files starting with p_s_given_d but not containing 'processed'
    pattern = os.path.join(current_dir, 'p_s_given_d*.json')
    input_files = glob.glob(pattern)
    
    # Filter out files containing 'processed'
    input_files = [f for f in input_files if 'processed' not in f]
    
    if not input_files:
        print("No matching files found", file=sys.stderr)
        sys.exit(1)
        
    # Process each file
    for input_file in input_files:
        print(f"Processing {input_file}")
        process_file(input_file)

if __name__ == "__main__":
    main()
