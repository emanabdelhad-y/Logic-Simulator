import re


def parse_stim_file(filename):
    pairs = []

    # Open the .stim file for reading
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Regular expression to match the lines with dynamic input variables
    # Now it can handle `0`, `1`, `1'b0`, `1'b1`, `4'b1010`, `2'b00`
    regex = r"#(\d+)\s*((?:\w+\s*=\s*(\d+'b[01]+|[01]);?\s*)+)"

    # Dictionary to store the current state of input signals
    current_signals = {}

    for line in lines:
        match = re.search(regex, line)
        if match:
            # Extract the delay
            delay = int(match.group(1))
            # Get the assignments part of the line
            assignments = match.group(2)

            # Extract each variable assignment in the form var = value
            assignment_regex = r"(\w+)\s*=\s*(\d+'b[01]+|[01])"
            for assignment in re.findall(assignment_regex, assignments):
                variable, value = assignment[0], assignment[
                    1]  # Take the first capturing group (variable) and second (value)
                current_signals[variable] = value

            # Append the pair (delay, current_signals) to the list
            pairs.append((delay, current_signals.copy()))  # Use copy to avoid reference issues

    return pairs


# Example usage:
filename = 'fulladder.stim'
result = parse_stim_file(filename)
print(result)

filename = 'hadd.stim'
result = parse_stim_file(filename)
print(result)

filename = 'majority.stim'
result = parse_stim_file(filename)
print(result)

filename = 'mux4to1.stim'
result = parse_stim_file(filename)
print(result)

filename = 'dec2to4.stim'
result = parse_stim_file(filename)
print(result)