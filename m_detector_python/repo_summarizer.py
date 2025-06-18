#!/usr/bin/env python3
import sys
import os
import yaml # For reading YAML configuration files

def create_repo_summary(root_path_for_walk, output_file_path, file_extension, 
                        excluded_abs_paths=None, display_root_prefix=None,
                        excluded_folder_names=None, max_exclusion_depth=None):
    """
    Recursively finds all files with a specified extension in a given folder,
    excluding a specified list of subfolders, and copies their contents into a single
    output text file. Each file's content in the output is preceded by a
    header indicating its source path.

    Args:
        root_path_for_walk (str): The absolute path to the folder to search.
        output_file_path (str): The path to the text file to create/overwrite.
        file_extension (str): The file extension to look for (e.g., '.py').
        excluded_abs_paths (list[str], optional): A list of absolute paths of subfolders to exclude.
                                                  Defaults to None (no exclusion).
        display_root_prefix (str, optional): The original 'folder_to_scan' string from config,
                                             used for creating user-friendly display paths.
                                             Defaults to '.', meaning paths are relative to root_path_for_walk.
        excluded_folder_names (list[str], optional): A list of folder names to exclude at any depth.
                                                     Defaults to None (no exclusion by name).
        max_exclusion_depth (int, optional): Maximum depth level for folder name exclusion.
                                             None means no depth limit. Depth 1 = immediate children,
                                             Depth 2 = grandchildren, etc.
    """
    if not os.path.isdir(root_path_for_walk):
        print(f"Error: Folder '{root_path_for_walk}' (resolved from '{display_root_prefix or root_path_for_walk}') not found or is not a directory.")
        sys.exit(1)

    found_files_count = 0
    # Ensure display_root_prefix has a default for path joining if None
    effective_display_root = display_root_prefix if display_root_prefix is not None else '.'

    def calculate_depth(path, root):
        """Calculate the depth of a path relative to the root."""
        rel_path = os.path.relpath(path, root)
        if rel_path == '.':
            return 0
        return len(rel_path.split(os.sep))

    def should_exclude_by_name(dir_path, dir_name):
        """Check if a directory should be excluded based on its name and depth."""
        if not excluded_folder_names or dir_name not in excluded_folder_names:
            return False
        
        if max_exclusion_depth is None:
            return True  # No depth limit, exclude at any depth
        
        depth = calculate_depth(dir_path, root_path_for_walk)
        return depth <= max_exclusion_depth

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for dirpath, dirnames, filenames in os.walk(root_path_for_walk):
                current_dir_abs_path = os.path.abspath(os.path.normpath(dirpath))
                current_dir_name = os.path.basename(current_dir_abs_path)

                # Check if the current directory should be excluded by absolute path.
                excluded_by_abs_path = False
                if excluded_abs_paths:
                    # An efficient check to see if the current path is, or is inside, any of the excluded paths.
                    excluded_by_abs_path = any(current_dir_abs_path == excluded_path or 
                                              current_dir_abs_path.startswith(excluded_path + os.sep) 
                                              for excluded_path in excluded_abs_paths)

                # Check if the current directory should be excluded by name and depth
                excluded_by_name = should_exclude_by_name(current_dir_abs_path, current_dir_name)
                
                if excluded_by_abs_path or excluded_by_name:
                    dirnames[:] = []  # Don't descend into subdirectories of an excluded folder.
                    exclusion_reason = "absolute path" if excluded_by_abs_path else f"name '{current_dir_name}'"
                    if excluded_by_name and max_exclusion_depth is not None:
                        depth = calculate_depth(current_dir_abs_path, root_path_for_walk)
                        exclusion_reason += f" at depth {depth}"
                    print(f"Debug: Skipping excluded directory ({exclusion_reason}): {current_dir_abs_path}")
                    continue          # Skip processing files in this excluded directory.

                # Also check if we should exclude subdirectories by name before descending
                if excluded_folder_names:
                    # Filter out directories that should be excluded by name
                    dirs_to_remove = []
                    for dirname in dirnames:
                        subdir_path = os.path.join(current_dir_abs_path, dirname)
                        if should_exclude_by_name(subdir_path, dirname):
                            dirs_to_remove.append(dirname)
                            exclusion_info = f"name '{dirname}'"
                            if max_exclusion_depth is not None:
                                depth = calculate_depth(subdir_path, root_path_for_walk)
                                exclusion_info += f" at depth {depth}"
                            print(f"Debug: Will skip subdirectory ({exclusion_info}): {subdir_path}")
                    
                    # Remove excluded directories from dirnames to prevent os.walk from entering them
                    for dirname in dirs_to_remove:
                        dirnames.remove(dirname)

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
            exclusion_info = []
            if excluded_folder_names:
                depth_info = f" (max depth: {max_exclusion_depth})" if max_exclusion_depth is not None else " (any depth)"
                exclusion_info.append(f"folder names {excluded_folder_names}{depth_info}")
            if excluded_abs_paths:
                exclusion_info.append(f"absolute paths")
            
            exclusion_text = f" with exclusions: {', '.join(exclusion_info)}" if exclusion_info else ""
            
            print(f"No files with extension '{file_extension}' found in '{display_root_prefix or root_path_for_walk}'{exclusion_text}. "
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
    
    excluded_subfolders_list = config.get('excluded_subfolders') # Expects a list, can be None
    excluded_folder_names_list = config.get('excluded_folder_names') # New: list of folder names to exclude
    max_exclusion_depth_int = config.get('max_exclusion_depth') # New: maximum depth for name-based exclusion

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

    # Validate excluded_folder_names
    if excluded_folder_names_list is not None and not isinstance(excluded_folder_names_list, list):
        print(f"Warning: 'excluded_folder_names' in config should be a list. Ignoring the setting.")
        excluded_folder_names_list = None

    # Validate max_exclusion_depth
    if max_exclusion_depth_int is not None:
        if not isinstance(max_exclusion_depth_int, int) or max_exclusion_depth_int < 1:
            print(f"Warning: 'max_exclusion_depth' should be a positive integer. Ignoring the setting.")
            max_exclusion_depth_int = None

    # Prepare paths for the main function
    # root_path_for_walk: Absolute path for os.walk and internal logic
    root_path_for_walk = os.path.abspath(os.path.normpath(folder_to_scan_str))
    
    # display_root_prefix: The original string from config, used for display path generation
    display_root_prefix = folder_to_scan_str 

    # Process the list of excluded folders 
    # excluded_abs_paths: A list of absolute paths for the folders to be excluded.
    excluded_abs_paths = []
    if excluded_subfolders_list: # Checks for None and empty list implicitly
        if isinstance(excluded_subfolders_list, list):
            # Use a list comprehension to resolve each path to its absolute, normalized form.
            excluded_abs_paths = [os.path.abspath(os.path.normpath(p)) for p in excluded_subfolders_list]
        else:
            print(f"Warning: 'excluded_subfolders' in config should be a list. Ignoring the setting.")

    create_repo_summary(root_path_for_walk, 
                        output_file_path_str, 
                        file_extension_str, 
                        excluded_abs_paths, # Pass the list of paths
                        display_root_prefix,
                        excluded_folder_names_list, # New parameter
                        max_exclusion_depth_int)    # New parameter

if __name__ == "__main__":
    main()