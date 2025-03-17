import os

# Define the directory structure
project_structure = {
    "fashion-classification": [
        "data",
        "models",
        "notebooks",
        "static",
        "templates",
        "uploads",
        "sample_images"
    ]
}

# Define essential files
files_to_create = {
    "fashion-classification/templates/index.html": "<!-- Add your HTML template here -->",
    "fashion-classification/improved_fashion_classification.py": "# Main training script",
    "fashion-classification/improved_app.py": "# Flask API script",
    "fashion-classification/requirements.txt": "# Add required dependencies here",
    "fashion-classification/README.md": "# Fashion Classification Project\n\nDescription of the project."
}

# Function to create directories
def create_directories():
    for parent, subdirs in project_structure.items():
        os.makedirs(parent, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(parent, subdir), exist_ok=True)
    print("✅ Directories created successfully.")

# Function to create files with default content
def create_files():
    for file_path, content in files_to_create.items():
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(content)
            print(f"✅ Created: {file_path}")
        else:
            print(f"⚠️ File already exists: {file_path}")

# Main execution
if __name__ == "__main__":
    create_directories()
    create_files()
    print("🎉 Project setup completed!")
