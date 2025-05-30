import subprocess
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# List of scripts to run in order
scripts = [
    "Merge.py",
    "MergedCSV_CleanUp.py",
    "MergedCSV_MrktCapProcessing.py",
    "trend_plot.py"
]

def run_script(script_name):
    script_path = os.path.join(current_dir, script_name)
    print(f"Running {script_name}...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"{script_name} completed successfully.")
        print(result.stdout)
    else:
        print(f"Error running {script_name}:")
        print(result.stderr)
    
    print("-" * 50)

# Run each script in order
for script in scripts:
    run_script(script)

print("All scripts have been executed.")
