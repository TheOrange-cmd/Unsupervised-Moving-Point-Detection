#!/usr/bin/env python3
import sys
import os
import yaml # For reading YAML configuration files

def create_repo_summary(root_path_for_walk, output_file_path, file_extension, 
                        excluded_abs_path=None, display_root_prefix=None):
    """
    Recursively finds all files with a specified extension in a given folder,
    excluding a specified subfolder, and copies their contents into a single
    output text file. Each file's content in the output is preceded by a
    header indicating its source path.

    Args:
        root_path_for_walk (str): The absolute path to the folder to search.
        output_file_path (str): The path to the text file to create/overwrite.
        file_extension (str): The file extension to look for (e.g., '.py').
        excluded_abs_path (str, optional): The absolute path of the subfolder to exclude.
                                           Defaults to None (no exclusion).
        display_root_prefix (str, optional): The original 'folder_to_scan' string from config,
                                             used for creating user-friendly display paths.
                                             Defaults to '.', meaning paths are relative to root_path_for_walk.
    """
    if not os.path.isdir(root_path_for_walk):
        print(f"Error: Folder '{root_path_for_walk}' (resolved from '{display_root_prefix or root_path_for_walk}') not found or is not a directory.")
        sys.exit(1)

    found_files_count = 0
    # Ensure display_root_prefix has a default for path joining if None
    effective_display_root = display_root_prefix if display_root_prefix is not None else '.'

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for dirpath, dirnames, filenames in os.walk(root_path_for_walk):
                current_dir_abs_path = os.path.abspath(os.path.normpath(dirpath))

                # Exclusion logic
                if excluded_abs_path:
                    # Check if the current directory is the excluded one or a subdirectory of it.
                    if current_dir_abs_path == excluded_abs_path or \
                       current_dir_abs_path.startswith(excluded_abs_path + os.sep):
                        dirnames[:] = []  # Don't descend into subdirectories of the excluded folder
                        # print(f"Debug: Skipping excluded directory: {current_dir_abs_path}") # Uncomment for debugging
                        continue          # Skip processing files in this excluded directory

                for filename in filenames:
                    if filename.endswith(file_extension):
                        found_files_count += 1
                        source_file_abs_path = os.path.join(current_dir_abs_path, filename)
                        
                        # Calculate display path to be consistent with original script's behavior
                        # (relative to the originally specified scan folder)
                        path_relative_to_walk_root = os.path.relpath(source_file_abs_path, root_path_for_walk)
                        
                        # Construct display path using the original display_root_prefix
                        # os.path.join handles cases like display_root_prefix = '.' correctly
                        raw_display_path = os.path.join(effective_display_root, path_relative_to_walk_root)
                        
                        # Normalize the path (e.g., remove './', use forward slashes for display)
                        normalized_display_path = os.path.normpath(raw_display_path)
                        display_path = normalized_display_path.replace(os.sep, '/')

                        header_top = "############################\n"
                        file_line = f"FILE: {display_path}\n"
                        header_bottom = "############################\n"
                        
                        outfile.write(header_top)
                        outfile.write(file_line)
                        outfile.write(header_bottom)
                        outfile.write("\n")

                        try:
                            with open(source_file_abs_path, 'r', encoding='utf-8', errors='ignore') as infile:
                                content = infile.read()
                                outfile.write(content)
                            
                            if content and not content.endswith('\n'):
                                outfile.write("\n")
                            outfile.write("\n")

                        except Exception as e:
                            error_message = f"\nError reading file {display_path}: {e}\n"
                            outfile.write(error_message)
                            outfile.write("\n\n")
                            print(f"Warning: Could not read file {display_path}: {e}")
        
        if found_files_count > 0:
            print(f"Successfully summarized {found_files_count} file(s) into '{output_file_path}'.")
        else:
            print(f"No files with extension '{file_extension}' found in '{display_root_prefix or root_path_for_walk}' "
                  f"(or matching files were all in excluded folders). "
                  f"Output file '{output_file_path}' was created but may be empty or "
                  f"contain only headers of unreadable files if any errors occurred.")

    except IOError as e:
        print(f"Error writing to output file '{output_file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    """
    Reads configuration from a YAML file and calls the create_repo_summary function.
    """
    script_name = os.path.basename(sys.argv[0])
    config_file_name = "summarizer_config.yaml"
    # Assume config file is in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    config_file_path = os.path.join(script_dir, config_file_name)

    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config is None: # Handles empty config file
            config = {}
    except FileNotFoundError:
        print(f"Error: Config file '{config_file_path}' not found.")
        print(f"Please create '{config_file_name}' in the script's directory with necessary parameters.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file '{config_file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config file '{config_file_path}': {e}")
        sys.exit(1)

    # Load parameters from config
    folder_to_scan_str = config.get('folder_to_scan')
    output_file_path_str = config.get('output_summary_file')
    file_extension_str = config.get('file_extension')
    excluded_subfolder_str = config.get('excluded_subfolder') # Can be None or empty

    # Validate required parameters
    if not all([folder_to_scan_str, output_file_path_str, file_extension_str]):
        print(f"Error: 'folder_to_scan', 'output_summary_file', and 'file_extension' "
              f"must be set in '{config_file_name}'.")
        sys.exit(1)

    # Validate file extension format
    if not file_extension_str.startswith('.'):
        print(f"Error: File extension '{file_extension_str}' in config must start with a '.' (e.g., '.py').")
        sys.exit(1)
    if len(file_extension_str) < 2:
        print(f"Error: File extension '{file_extension_str}' in config is too short (e.g., use '.py', not just '.').")
        sys.exit(1)

    # Prepare paths for the main function
    # root_path_for_walk: Absolute path for os.walk and internal logic
    root_path_for_walk = os.path.abspath(os.path.normpath(folder_to_scan_str))
    
    # display_root_prefix: The original string from config, used for display path generation
    display_root_prefix = folder_to_scan_str 

    # excluded_abs_path: Absolute path for the folder to be excluded, if specified
    excluded_abs_path = None
    if excluded_subfolder_str: # Checks for None and empty string implicitly
        # excluded_subfolder_str is resolved relative to the Current Working Directory if relative,
        # or used as is if absolute. Then, it's normalized.
        excluded_abs_path = os.path.abspath(os.path.normpath(excluded_subfolder_str))

    create_repo_summary(root_path_for_walk, 
                        output_file_path_str, 
                        file_extension_str, 
                        excluded_abs_path,
                        display_root_prefix)

if __name__ == "__main__":
    main()