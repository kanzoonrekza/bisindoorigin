import os

# Define the directories and file paths
logs_dir = 'Logs'
output_file = 'combined_summary.txt'

# Initialize an empty list to store the results
results = []

# Traverse the Logs directory
for root, dirs, files in os.walk(logs_dir):
    if 'summary.txt' in files:
        # Construct the full path to the summary.txt file
        file_path = os.path.join(root, 'summary.txt')

        # Open and read the summary.txt file
        try:
            with open(file_path, 'r') as f:
                # Read all lines
                lines = f.readlines()

                # Check if we have at least 19 lines
                if len(lines) >= 19:
                    # First line
                    line_1 = lines[0][16:-17].strip() + " " + \
                        lines[0][-7:].strip()
                    if len(line_1) == 13:
                        line_1 += " "

                    # Nineteenth line (best epoch line.)
                    line_19 = lines[18].strip()
                    if len(line_19) > 8 and line_19[8] == ':':
                        line_19 = line_19[:8] + ' :' + line_19[9:]

                    # Append the formatted result
                    results.append(f"{line_1} => {line_19}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Write all results to the output file
with open(output_file, 'w') as out_file:
    out_file.write('\n'.join(results))

print(f"Combined summary saved to {output_file}")
