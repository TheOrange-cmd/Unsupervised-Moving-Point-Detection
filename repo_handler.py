#!/usr/bin/env python3

import os
import shutil
import argparse
import pathspec # For .gitignore parsing
import sys # For stderr

# --- Configuration ---
# File extensions to copy
TARGET_EXTENSIONS = {'.h', '.hpp', '.cpp'}
# Source folders within the repo to search for files to copy
SOURCE_FOLDERS = ['src', 'include']
# Output filename for the structure summary
STRUCTURE_FILENAME = 'structure.txt'
# Character to replace directory separators with during flattening
FLATTEN_SEPARATOR = '_'
# --- End Configuration ---

def parse_gitignore(repo_root):
    """
    Parses the .gitignore file in the repository root.
    Returns a pathspec object for matching paths.
    Handles cases where .gitignore doesn't exist.
    """
    gitignore_path = os.path.join(repo_root, '.gitignore')
    patterns = []
    if os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                patterns = [
                    line.strip() for line in f
                    if line.strip() and not line.strip().startswith('#')
                ]
            print(f"Read {len(patterns)} patterns from {gitignore_path}")
        except Exception as e:
            print(f"Warning: Could not read .gitignore: {e}", file=sys.stderr)
            return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, [])

    # Add common Git directory rule if not already present implicitly/explicitly
    # pathspec usually handles this, but being explicit doesn't hurt.
    if '.git/' not in patterns and '.git' not in patterns:
         patterns.append('.git/')

    # Create a PathSpec object using GitWildMatchPattern (like .gitignore)
    return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, patterns)

def copy_files_flattened(repo_root, dest_dir):
    """
    Copies files with TARGET_EXTENSIONS from SOURCE_FOLDERS
    within repo_root to dest_dir, FLATTENING the structure.
    Renames files based on their original relative path to avoid collisions.
    e.g., src/subdir/file.cpp -> dest_dir/src_subdir_file.cpp
    """
    print(f"\n--- Starting File Copy (Flattened) ---")
    copied_count = 0
    skipped_count = 0
    potential_collisions = {} # Track potential overwrites after renaming

    for src_folder_name in SOURCE_FOLDERS:
        abs_src_folder = os.path.join(repo_root, src_folder_name)

        if not os.path.isdir(abs_src_folder):
            print(f"Warning: Source folder '{src_folder_name}' not found in repository root. Skipping.")
            continue

        print(f"Searching in: {abs_src_folder}")

        # Walk through the source folder (src or include)
        for dirpath, _, filenames in os.walk(abs_src_folder):
            for filename in filenames:
                # Check if the file has one of the target extensions
                _, ext = os.path.splitext(filename)
                if ext.lower() in TARGET_EXTENSIONS:
                    # Construct the full source path
                    source_file_path = os.path.join(dirpath, filename)

                    # Calculate the relative path from the repo root
                    # e.g., src/subdir/file.cpp or include/utils/types.h
                    relative_path = os.path.relpath(source_file_path, repo_root)

                    # --- Create the flattened destination filename ---
                    # Replace directory separators with FLATTEN_SEPARATOR
                    # e.g., src/subdir/file.cpp -> src_subdir_file.cpp
                    flat_filename = relative_path.replace(os.sep, FLATTEN_SEPARATOR)

                    # Construct the full destination path (flattened)
                    dest_file_path = os.path.join(dest_dir, flat_filename)

                    # Check for potential collisions *after* renaming
                    if dest_file_path in potential_collisions:
                        print(f"Warning: Filename collision after flattening! Both '{potential_collisions[dest_file_path]}' and '{relative_path}' map to '{flat_filename}'. Overwriting.", file=sys.stderr)
                    potential_collisions[dest_file_path] = relative_path

                    # Copy the file, preserving metadata (like modification time)
                    try:
                        print(f"  Copying: {relative_path} -> {flat_filename}")
                        # No need to create subdirs in dest_dir as it's flat
                        shutil.copy2(source_file_path, dest_file_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"Error copying {source_file_path} to {dest_file_path}: {e}", file=sys.stderr)
                        skipped_count += 1

    print(f"--- File Copy Finished ---")
    print(f"Successfully copied: {copied_count} files")
    print(f"Skipped/Errors: {skipped_count} files")
    if skipped_count > 0 or any("Warning: Filename collision" in v for v in potential_collisions):
         print("Check warnings/errors above.", file=sys.stderr)


def generate_structure(repo_root, dest_dir, gitignore_spec):
    """
    Generates a text file listing the directory structure of repo_root,
    respecting .gitignore rules, and saves it in dest_dir.
    """
    print(f"\n--- Generating Structure Summary ({STRUCTURE_FILENAME}) ---")
    output_file_path = os.path.join(dest_dir, STRUCTURE_FILENAME)
    structure_lines = []
    ignored_count = 0
    processed_count = 0

    # Add the repo root directory name to the structure
    structure_lines.append(f"{os.path.basename(os.path.abspath(repo_root))}/")

    # Walk through the entire repository
    # topdown=True allows us to modify dirnames list to prune ignored directories
    for dirpath, dirnames, filenames in os.walk(repo_root, topdown=True, onerror=lambda err: print(f"Error accessing {err.filename}: {err.strerror}", file=sys.stderr)):

        # Calculate path relative to repo_root for gitignore matching and output formatting
        relative_dirpath = os.path.relpath(dirpath, repo_root)
        # Use '.' for the root directory itself in calculations, but handle output string
        display_dirpath = relative_dirpath if relative_dirpath != '.' else ''

        # Calculate indentation level based on the *display* path
        # Root contents are level 1, subdirs level 2, etc.
        level = display_dirpath.count(os.sep) if display_dirpath else 0
        indent = '  ' * level # Using 2 spaces for indent for better readability on deep trees
        prefix = indent + '|-- ' if level > 0 else '|-- ' # Root items start with |--

        # --- Filter Ignored Directories ---
        # We need to check directories *before* recursing into them.
        # Create a copy of dirnames to iterate over while modifying the original
        original_dirnames = sorted(list(dirnames)) # Sort for consistent order
        dirnames[:] = [] # Clear the original list; we'll add back only non-ignored ones

        for d in original_dirnames:
            # Path relative to repo root, including the directory name
            rel_path_to_check = os.path.join(relative_dirpath, d)
            # Normalize for pathspec (use forward slashes)
            spec_path_dir = rel_path_to_check.replace(os.sep, '/')
            # pathspec needs the trailing slash for directory matching patterns like 'build/'
            spec_path_dir_slash = spec_path_dir + '/'

            # Check if the directory matches any gitignore pattern
            # Check both 'dir' and 'dir/' forms against the patterns
            is_ignored = gitignore_spec.match_file(spec_path_dir) or gitignore_spec.match_file(spec_path_dir_slash)

            if is_ignored:
                # print(f"Ignoring directory: {spec_path_dir_slash}") # Optional debug
                ignored_count += 1 # Count the dir itself
                # We don't add it to structure_lines and don't add to dirnames, so os.walk skips it
            else:
                # If not ignored, add it back to dirnames so os.walk will descend into it
                dirnames.append(d)
                # Add the directory to our structure output
                structure_lines.append(f"{prefix}{d}/")
                processed_count += 1

        # --- Process Files in the Current Directory ---
        # Ensure files are processed *after* their parent directory entry is added
        # Need to recalculate indent for files based on parent dir's level
        file_indent = '  ' * (level + 1) # Files are one level deeper than their containing dir entry
        file_prefix = file_indent + '|-- '

        for filename in sorted(filenames): # Sort filenames for consistent output
            # Path relative to repo root
            rel_path_to_check = os.path.join(relative_dirpath, filename)
            # Normalize for pathspec
            spec_path_file = rel_path_to_check.replace(os.sep, '/')

            # Check if the file matches any gitignore pattern
            if gitignore_spec.match_file(spec_path_file):
                # print(f"Ignoring file: {spec_path_file}") # Optional debug
                ignored_count += 1
            else:
                # Add the file to our structure output
                structure_lines.append(f"{file_prefix}{filename}")
                processed_count += 1

    # Write the structure to the output file
    try:
        # Ensure destination directory exists
        os.makedirs(dest_dir, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in structure_lines:
                f.write(line + '\n')
        print(f"Structure summary saved to: {output_file_path}")
        print(f"Items listed: {processed_count}")
        # Note: ignored_count might be higher than expected if a dir is ignored,
        # as we don't count the files/subdirs *inside* it separately.
        print(f"Top-level items ignored (due to .gitignore): {ignored_count}")
    except Exception as e:
        print(f"Error writing structure file {output_file_path}: {e}", file=sys.stderr)

    print(f"--- Structure Summary Finished ---")


def main():
    """
    Main function to parse arguments and orchestrate tasks.
    """
    parser = argparse.ArgumentParser(
        description=f"Copies specific source files ({', '.join(TARGET_EXTENSIONS)}) from "
                    f"{'/'.join(SOURCE_FOLDERS)} folders of a local repository to a new directory, "
                    "flattening the structure (renaming files like path_to_file), "
                    f"and generates a {STRUCTURE_FILENAME} file showing the "
                    ".gitignore-aware directory structure of the original repo."
    )
    parser.add_argument(
        "repo_path",
        help="Path to the local repository root directory."
    )
    parser.add_argument(
        "dest_path",
        help="Path to the destination directory where flattened files and structure summary will be saved. "
             "It will be created if it doesn't exist."
    )

    args = parser.parse_args()

    # --- Validate Paths ---
    repo_root = os.path.abspath(args.repo_path)
    dest_dir = os.path.abspath(args.dest_path)

    if not os.path.isdir(repo_root):
        print(f"Error: Repository path '{repo_root}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1) # Exit if repo path is invalid

    # Check for potential nesting issue: destination inside source repo
    # Use os.path.realpath to resolve symlinks before comparing
    real_repo_root = os.path.realpath(repo_root)
    real_dest_dir = os.path.realpath(dest_dir)
    if os.path.commonpath([real_repo_root, real_dest_dir]) == real_repo_root:
        if real_dest_dir != real_repo_root:
             print(f"Warning: Destination path '{dest_dir}' is inside the source repository path '{repo_root}'.", file=sys.stderr)
             print("Ensure the destination path itself is ignored via .gitignore if you run this repeatedly.", file=sys.stderr)
             # Proceeding, but user should be aware.

    # Create destination directory if it doesn't exist
    try:
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Ensured destination directory exists: {dest_dir}")
    except OSError as e:
        print(f"Error: Could not create destination directory '{dest_dir}': {e}", file=sys.stderr)
        sys.exit(1) # Exit if dest dir cannot be created

    # --- Execute Tasks ---
    # 1. Parse .gitignore (needed for structure summary)
    gitignore_spec = parse_gitignore(repo_root)

    # 2. Copy specified files (Flattened)
    copy_files_flattened(repo_root, dest_dir)

    # 3. Generate structure summary
    generate_structure(repo_root, dest_dir, gitignore_spec)

    print("\nScript finished.")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()