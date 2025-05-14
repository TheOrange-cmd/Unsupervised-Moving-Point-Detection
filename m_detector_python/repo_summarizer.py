#!/usr/bin/env python3
import sys
import os

def create_repo_summary(folder_path, output_file_path, file_extension):
    """
    Recursively finds all files with a specified extension in a given folder
    and copies their contents into a single output text file. Each file's
    content in the output is preceded by a header indicating its source path.

    Args:
        folder_path (str): The path to the folder to search (e.g., './src').
        output_file_path (str): The path to the text file to create/overwrite 
                                (e.g., './repo_summary.txt').
        file_extension (str): The file extension to look for (e.g., '.py').
    """
    # Validate that the folder_path exists and is a directory
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found or is not a directory.")
        sys.exit(1) # Exit if the source folder is invalid

    found_files_count = 0
    try:
        # Open the output file in write mode ('w'). This will overwrite the file if it exists.
        # UTF-8 encoding is used for broad compatibility.
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # os.walk generates the file names in a directory tree by walking the tree 
            # either top-down or bottom-up.
            for dirpath, _, filenames in os.walk(folder_path):
                for filename in filenames:
                    # Check if the current file ends with the specified extension
                    if filename.endswith(file_extension):
                        found_files_count += 1
                        # Construct the full path to the source file
                        source_file_path = os.path.join(dirpath, filename)
                        
                        # Normalize the path for display (e.g., remove './', use forward slashes)
                        # os.path.normpath cleans the path (e.g., 'A/./B' becomes 'A/B').
                        normalized_path = os.path.normpath(source_file_path)
                        # Replace OS-specific separators with forward slashes for consistent display
                        display_path = normalized_path.replace(os.sep, '/')

                        # Define the header format as per the example
                        header_top = "############################\n"
                        file_line = f"FILE: {display_path}\n"
                        header_bottom = "############################\n"
                        
                        # Write the header to the output file
                        outfile.write(header_top)
                        outfile.write(file_line)
                        outfile.write(header_bottom)
                        outfile.write("\n")  # Add a blank line between the header and the file content

                        try:
                            # Open the source file in read mode ('r')
                            # 'errors="ignore"' will skip characters that can't be decoded,
                            # which can be useful for potentially mixed-encoding files.
                            with open(source_file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                                content = infile.read()
                                outfile.write(content) # Write the file's content
                            
                            # Ensure proper separation before the next file's header.
                            # If the content did not end with a newline, add one.
                            if content and not content.endswith('\n'):
                                outfile.write("\n")
                            # Add one more newline to create a blank line before the next header.
                            outfile.write("\n")

                        except Exception as e:
                            # If a specific file can't be read, write an error message in its place
                            # and print a warning to the console.
                            error_message = f"\nError reading file {display_path}: {e}\n"
                            outfile.write(error_message)
                            outfile.write("\n\n") # Maintain separation
                            print(f"Warning: Could not read file {display_path}: {e}")
        
        if found_files_count > 0:
            print(f"Successfully summarized {found_files_count} file(s) into '{output_file_path}'.")
        else:
            # If no files matched the extension, inform the user.
            print(f"No files with extension '{file_extension}' found in '{folder_path}'. "
                  f"Output file '{output_file_path}' was created but may be empty or "
                  f"contain only headers of unreadable files if any errors occurred.")

    except IOError as e:
        # Handle errors related to writing the output file itself
        print(f"Error writing to output file '{output_file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        # Handle any other unexpected errors during the process
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    """
    Parses command-line arguments and calls the create_repo_summary function.
    """
    # sys.argv contains the command-line arguments passed to the script.
    # sys.argv[0] is the script name.
    if len(sys.argv) != 4:
        script_name = os.path.basename(sys.argv[0]) # Get the script name for usage instructions
        print(f"Usage: python {script_name} <folder_to_scan> <output_summary_file> <file_extension>")
        print(f"Example: python {script_name} ./src ./repo_summary.txt '.py'")
        sys.exit(1) # Exit if arguments are incorrect

    # Assign arguments to variables
    folder_path = sys.argv[1]
    output_file_path = sys.argv[2]
    file_extension = sys.argv[3]

    # Validate the file extension format (e.g., must start with '.' and have some length)
    if not file_extension.startswith('.'):
        print(f"Error: File extension '{file_extension}' must start with a '.' (e.g., '.py').")
        sys.exit(1)
    if len(file_extension) < 2: # e.g., '.' is invalid, needs at least one char after dot like '.c'
        print(f"Error: File extension '{file_extension}' is too short (e.g., use '.py', not just '.').")
        sys.exit(1)

    # Call the main function to perform the summarization
    create_repo_summary(folder_path, output_file_path, file_extension)

if __name__ == "__main__":
    main()