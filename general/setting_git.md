Step 0 — Create .gitignore
Do this before any git commands so unwanted files are never tracked.

Your current .gitignore looks good. Here's what each line does:


# Python
__pycache__/    # Python's compiled bytecode cache folder
*.pyc           # Compiled Python files
*.pyo           # Optimized compiled Python files
.venv/          # Virtual environment folder
venv/           # Alternative venv name
env/            # Another alternative venv name

# IDE
.vscode/        # VS Code settings folder
.idea/          # PyCharm/JetBrains settings folder

# OS
.DS_Store       # macOS folder metadata file
Thumbs.db       # Windows image cache file

# Logs
*.log           # Any log files

# Environment
.env            # Secret keys, API tokens, passwords — NEVER push this
Step 1 — Initialize git

git init
# Creates a hidden .git/ folder — this makes your folder a git repository
Step 2 — Connect to GitHub

git remote add origin <your-repo-link>
# Links your local repo to the remote GitHub repo
# "origin" is just a nickname for the remote URL
Step 3 — Stage files

git add .
# Stages ALL files (except those in .gitignore) for the next commit
# Think of it as "selecting" what to include in the snapshot
Step 4 — Commit

git commit -m "Initial commit"
# Takes a snapshot of all staged files and saves it with a message
# This is saved LOCALLY, nothing goes to GitHub yet
Step 5 — Rename branch to main

git branch -M main
# Renames your default branch to "main" (GitHub's default name)
# -M forces the rename even if "main" already exists
Step 6 — Push to GitHub

git push -u origin main
# Uploads your commits to GitHub
# -u sets "origin main" as the default, so next time you can just type "git push"
