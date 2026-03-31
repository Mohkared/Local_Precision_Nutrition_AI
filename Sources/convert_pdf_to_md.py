"""
This script converts all PDF files in the current directory to Markdown format using the opendataloader_pdf library. 
It checks for existing Markdown files to avoid overwriting and prompts the user for confirmation before proceeding with 
the conversion.
"""
import opendataloader_pdf
import os

print("\n\n")
print("═════════════════════════════════════════════════════════════════════════")
print("Files in current directory to convert to Markdown (.md):")

# Get a list of all files in the current directory and filter for PDFs without corresponding Markdown files
all_files = os.listdir(".")
files_to_convert = []
for filename in all_files:
    # check if the file is a PDF and has no corresponding Markdown file
    if filename.endswith(".pdf") and not os.path.exists(filename.replace(".pdf", ".md")):
        print(f'- {filename}')
        files_to_convert.append(filename)

print("═════════════════════════════════════════════════════════════════════════")
print(f"Total files to convert: {len(files_to_convert)}")

# Ask the user if they want to proceed with the conversion
proceed = input("Do you want to convert these files to Markdown? (y/n): ").strip().lower()
if proceed == "y":
    opendataloader_pdf.convert(
        input_path=files_to_convert,
        output_dir=".",  # Save Markdown files in the current directory
        format="markdown"#,json"
    )
    print("Conversion completed.")