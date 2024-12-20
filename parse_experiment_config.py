"""
Given a CSV, convert it to cli arguments for a python module call.
"""
import csv
import os
import sys

if __name__ == '__main__':
    source_path = sys.argv[1]
    dest_path = sys.argv[2]
    # read the CSV
    with open(source_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    # convert to cli arguments
    cli_args = []
    for row in rows:
        line = ""
        for i, arg in enumerate(row):
            if arg == 'FALSE':
                continue
            if arg == 'TRUE':
                line += f"--{header[i]} "
            else:
                line += f"--{header[i]}={arg} "
        cli_args.append(line)
    # write to file
    with open(dest_path, 'w') as f:
        f.write('\n'.join(cli_args))
    print(f"Saved to {dest_path}")
