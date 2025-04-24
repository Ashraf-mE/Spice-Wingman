import subprocess
import re
import shutil
import os
from groq import Groq


ltspice_exe = os.path.join(os.environ["ProgramFiles"], "LTC", "LTspiceXVII", "XVIIx64.exe")

ltspice_path = os.environ["ProgramFiles"].replace("\\", "/") + "/LTC/LTspiceXVII"

def modify_ltspice_lib_path(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Ensure the path format matches LTspice conventions (using forward slashes)
    modified_content = re.sub(r'(?<=\.lib ).*?(?=/lib/)', ltspice_path, content)

    with open(file_path, 'w') as file:
        file.write(modified_content)

def convert_tran_ac(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    content = re.sub(r'(?m)^\.tran', ';tran', content)
    content = re.sub(r'(?m)^;ac', '.ac', content)
    with open(file_path, 'w') as file:
        file.write(content)

def reverse_convert_tran_ac(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    content = re.sub(r'(?m)^;tran', '.tran', content)
    content = re.sub(r'(?m)^\.ac', ';ac', content)
    with open(file_path, 'w') as file:
        file.write(content)

def optimize_circuit_parameters(file_path, api_key):
    with open(file_path, 'r') as file:
        text_string = file.read()
    
    stringified_text = repr(text_string)  # Preserve formatting

    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """
                Optimize this specific circuit and change the parameters accordingly while satisfying high gain 
                and low power consumption. 
                
                Note: The output has to be concise, and only the changed parameters need to be included. 
                No other text should be present except the optimized parameters in the format *R1:30K.
                """
            },
            {
                "role": "user",
                "content": stringified_text
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,  # Ensure a complete response
        stop=None,
    )

    output_text = completion.choices[0].message.content
    param_dict = {}

    for line in output_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("*") and ":" in line:
            key, value = line[1:].split(":", 1)
            param_dict[key.strip()] = value.strip()
    
    return param_dict

def run_ltspice(file_path, log_folder):
    subprocess.call([ltspice_exe, "-b", file_path])
    log_file = f"{os.path.splitext(file_path)[0]}.log"
    os.makedirs(log_folder, exist_ok=True)
    shutil.move(log_file, os.path.join(log_folder, os.path.basename(log_file)))



def update_netlist(file_path, output_path, param_dict):
    # Read the file content
    with open(file_path, "r") as file:
        text = file.read()

    # Replace values in the netlist while ignoring MOSFET components
    for component, new_value in param_dict.items():
        if component.startswith("M"):  # Ignore MOSFET components
            continue
        pattern = rf"^({component}\s+\S+\s+\S+)\s+\S+"  # Ensure it matches from the start of the line
        replacement = rf"\1 {new_value}"  # Replace the value while keeping nodes intact
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)  # Apply per line

    # Save the modified content to a new file
    with open(output_path, "w") as file:
        file.write(text)

    print(f"Replacement complete. Modified file saved as '{output_path}'.")

def read_value_from_log(file_path, keyword):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                if keyword in line:
                    return line.split('=')[1].split()[0].strip()  # Extract only the numerical value
    return "N/A"  # Return N/A if file not found or value is missing
