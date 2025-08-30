import os
import csv

# Path to the extracted folder
base_path = "static\image"  # Replace with the extracted folder's path

# CSV output file path
output_csv = "jewelry_dataset.csv"

# Function to generate attributes from folder and file names
def extract_attributes(folder_name, file_name):
    # Example logic for extracting attributes
    face_shape = None
    category = None
    material = None
    jewelry_type = None
    color = None

    # Assign face shape based on folder name
    if "oval" in folder_name.lower():
        face_shape = "Oval"
    elif "square" in folder_name.lower():
        face_shape = "Square"
    elif "round" in folder_name.lower():
        face_shape = "Round"
    elif "heart" in folder_name.lower():
        face_shape = "Heart"
    elif "oblong" in folder_name.lower():
        face_shape = "Oblong"

    # Assign category and material based on file name (customize based on your naming conventions)
    if "party" in file_name.lower():
        category = "Party"
    elif "traditional" in file_name.lower():
        category = "Traditional"
    elif "casual" in file_name.lower():
        category = "Casual"

    if "gold" in file_name.lower():
        material = "Gold"
    elif "silver" in file_name.lower():
        material = "Silver"
    elif "diamond" in file_name.lower():
        material = "Diamond"

    # Assign type based on file name
    if "necklace" in file_name.lower():
        jewelry_type = "Necklace"
    elif "ring" in file_name.lower():
        jewelry_type = "Ring"
    elif "earrings" in file_name.lower():
        jewelry_type = "Earrings"

    # Assign color based on file name (example logic)
    if "red" in file_name.lower():
        color = "Red"
    elif "blue" in file_name.lower():
        color = "Blue"
    elif "green" in file_name.lower():
        color = "Green"
    elif "yellow" in file_name.lower():
        color = "Yellow"
    elif "white" in file_name.lower():
        color = "White"

    return face_shape, category, material, jewelry_type, color

# Function to generate the dataset
def create_dataset(base_path, output_csv):
    rows = []
    for root, dirs, files in os.walk(base_path):
        folder_name = os.path.basename(root)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract relative file path
                file_path = os.path.relpath(os.path.join(root, file), base_path)
                
                # Extract attributes based on folder and file name
                face_shape, category, material, jewelry_type, color = extract_attributes(folder_name, file)
                
                # Add to the dataset row
                row = {
                    "image_path": file_path,
                    "face_shape": face_shape or "Unknown",
                    "category": category or "Unknown",
                    "material": material or "Unknown",
                    "type": jewelry_type or "Unknown",
                    "color": color or "Unknown"
                }
                rows.append(row)
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ["image_path", "face_shape", "category", "material", "type", "color"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Generate the dataset
create_dataset(base_path, output_csv)
print(f"Dataset created at {output_csv}")
