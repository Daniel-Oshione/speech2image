def validate_captions_file(caption_file):
    """
    Validates the captions file to ensure each line (except header) contains a tab separator.
    Prints lines that are malformed.

    Args:
        caption_file (str): Path to the captions file.
    """
    with open(caption_file, 'r') as file:
        lines = file.readlines()

    malformed_lines = []
    for i, line in enumerate(lines[1:], start=2):  # Skip header, line numbers start at 2
        if '\t' not in line:
            malformed_lines.append((i, line.strip()))

    if malformed_lines:
        print(f"Found {len(malformed_lines)} malformed lines in {caption_file}:")
        for line_num, content in malformed_lines:
            print(f"Line {line_num}: {content}")
    else:
        print(f"All lines in {caption_file} are properly formatted.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python validate_captions.py <path_to_captions_file>")
    else:
        validate_captions_file(sys.argv[1])
