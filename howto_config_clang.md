# Setting Up Consistent C++ Formatting with clang-format and VS Code

This guide explains how to ensure consistent C++ code formatting across the team using `clang-format` built from source and a shared VS Code task configuration.

**Goal:** Every team member can format the currently open C++ file using a VS Code task (or keyboard shortcut) that relies on a locally built `clang-format`, without needing user-specific paths in the shared configuration.

## 1. Build clang-format from Source

We will build `clang-format` as part of the Clang toolset within the LLVM project.

1.  **Clone the LLVM Project:**
    Choose a location to download the source code (e.g., `~/llvm-project`).
    ```bash
    cd ~
    git clone https://github.com/llvm/llvm-project.git
    cd llvm-project
    ```

2.  **Create a Build Directory:** It's best practice to build outside the source directory.
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure the Build with CMake:**
    This command tells CMake how to prepare the build. We only enable the `clang` project (which includes `clang-format`) and specify an installation directory *relative to the build directory*.

        ```bash
        cmake ../llvm \
              -DCMAKE_BUILD_TYPE=Release \
              -DLLVM_ENABLE_PROJECTS=clang \
              -DCMAKE_INSTALL_PREFIX=../install
        ```
    *   *Explanation:*
        *   `../llvm`: Path to the LLVM source code (CMakeLists.txt).
        *   `CMAKE_BUILD_TYPE=Release`: Builds optimized binaries.
        *   `LLVM_ENABLE_PROJECTS=clang`: Only build the Clang project (saves significant time).
        *   `CMAKE_INSTALL_PREFIX=../install`: Sets a local install directory (`~/llvm-project/install`).

4.  **Build clang-format:**
        ```bash
        make clang-format -j$4 # Uses four CPU cores - consider if more cores is acceptable as this process can take a while
        ```

5.  **Install Locally:** This copies the necessary built files to the `install` directory we specified.
    *   **If using Make:**
        ```bash
        make install
        ```
    After this step, you should find the `clang-format` binary inside `~/llvm-project/install/bin/`.

## 3. Install clang-format to Conventional Location

For the shared VS Code task to work, every team member needs to place the `clang-format` binary in the *same relative path* within their home directory. We will use `~/.clang-format/bin/`.

1.  **Create the Target Directory:**
    ```bash
    mkdir -p ~/.clang-format/bin
    ```

2.  **Copy the Binary:** Copy the built `clang-format` from the local install directory to the conventional location.
    ```bash
    cp ~/llvm-project/install/bin/clang-format ~/.clang-format/bin/
    ```

3.  **Verify Execution:** Make sure it runs.
    ```bash
    ~/.clang-format/bin/clang-format --version
    ```

## 4. Configure VS Code Task

1.  **Create/Open `tasks.json`:**
    *   In VS Code, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`).
    *   Type `Tasks: Configure Task`.
    *   If prompted, select "Create tasks.json file from template", then choose "Others".
    *   This creates/opens `.vscode/tasks.json` in your project root.

2.  **Add the Formatting Task:** Replace the contents of `tasks.json` with this:

    ```json
    {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "Format Current File (clang-format)", // Name shown in VS Code
                "type": "shell",
                // Use ${userHome} variable for portability
                "command": "${userHome}/.clang-format/bin/clang-format",
                "args": [
                    "-i",                                // Format in-place (overwrites file!)
                    "--style=Google",                    // Or your team's chosen style (e.g., LLVM, File)
                    "${file}"                            // VS Code variable for the current file path
                ],
                "problemMatcher": [],
                "presentation": {
                    "reveal": "never",                   // Don't show the terminal panel
                    "panel": "shared",
                    "showReuseMessage": false,
                    "clear": false
                },
                "group": {
                    "kind": "build",
                    "isDefault": false
                }
            }
        ]
    }
    ```
    *   **Key Points:**
        *   `command`: Uses `${userHome}` which automatically resolves to the user's home directory (e.g., `/home/drugge`). This requires the convention from Step 3.
        *   `args`:
            *   `-i`: **Formats the file in-place.** Ensure you use version control (Git) and commit changes before formatting, especially when first setting up.
            *   `--style=Google`: Change this if your team uses a different base style or a custom `.clang-format` file (see below).
            *   `${file}`: Passes the currently active file to the command.

3.  **Commit `.vscode/tasks.json`:** Add this file to your Git repository so the whole team gets the task definition.

## 5. Usage

1.  **Open a C++ File:** Open any `.cpp`, `.h`, or `.hpp` file you want to format.
2.  **Run the Task:**
    *   Open the Command Palette (`Ctrl+Shift+P`).
    *   Type `Tasks: Run Task`.
    *   Select "Format Current File (clang-format)".
    *   The file should be formatted according to the specified style.

3.  **(Optional) Keyboard Shortcut:** For faster access:
    *   Open the Command Palette (`Ctrl+Shift+P`).
    *   Type `Preferences: Open Keyboard Shortcuts (JSON)`.
    *   Add this entry to the JSON list (adjust `key` as desired):
        ```json
        {
            "key": "ctrl+alt+f", // Example shortcut
            "command": "workbench.action.tasks.runTask",
            "args": "Format Current File (clang-format)" // Must match the task "label"
        }
        ```
    *   Save the file. Now `Ctrl+Alt+F` (or your shortcut) will trigger the formatting task.

## Additional Note

While the C/C++ extension should be able to handle this as well, this did not seem to work when working on the ssh remote host iv-mind for unknown reasons. While you are free to try this for yourself, the approach described above *does* work. 