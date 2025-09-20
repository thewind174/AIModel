#!/usr/bin/env python3
"""
Automatic Image File Renamer Script
Automatically renames all image files in current directory to image_001.jpg, image_002.jpg, etc.
"""

import os
import re
from pathlib import Path
from datetime import datetime
import argparse
import sys

# External libraries - install with: pip install pillow
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL (Pillow) not available. Install with: pip install pillow")

class ImageRenamer:
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.ico'}
        self.renamed_count = 0
        self.skipped_count = 0
        
    def get_image_files(self, directory):
        """Get all image files from directory"""
        image_files = []
        for file_path in Path(directory).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        return sorted(image_files)
    
    def get_image_metadata(self, image_path):
        """Extract metadata from image using PIL"""
        if not PIL_AVAILABLE:
            return {}
        
        try:
            with Image.open(image_path) as img:
                metadata = {
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode
                }
                
                # Extract EXIF data
                exif = img.getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        if tag_name == 'DateTime':
                            metadata['date_taken'] = value
                        elif tag_name == 'Make':
                            metadata['camera_make'] = value
                        elif tag_name == 'Model':
                            metadata['camera_model'] = value
                
                return metadata
        except Exception as e:
            print(f"Warning: Could not read metadata from {image_path}: {e}")
            return {}
    
    def generate_new_name(self, file_path, pattern, counter=None, metadata=None):
        """Generate new filename based on pattern"""
        original_name = file_path.stem
        extension = file_path.suffix
        file_size = file_path.stat().st_size
        creation_time = datetime.fromtimestamp(file_path.stat().st_ctime)
        modification_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        # Default metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Pattern replacements
        replacements = {
            '{original}': original_name,
            '{counter}': str(counter).zfill(3) if counter else '',
            '{date}': creation_time.strftime('%Y%m%d'),
            '{time}': creation_time.strftime('%H%M%S'),
            '{datetime}': creation_time.strftime('%Y%m%d_%H%M%S'),
            '{mod_date}': modification_time.strftime('%Y%m%d'),
            '{mod_datetime}': modification_time.strftime('%Y%m%d_%H%M%S'),
            '{size}': str(file_size),
            '{width}': str(metadata.get('width', '')),
            '{height}': str(metadata.get('height', '')),
            '{resolution}': f"{metadata.get('width', '')}x{metadata.get('height', '')}" if metadata.get('width') else '',
            '{format}': metadata.get('format', ''),
            '{camera}': f"{metadata.get('camera_make', '')}_{metadata.get('camera_model', '')}" if metadata.get('camera_make') else ''
        }
        
        new_name = pattern
        for placeholder, value in replacements.items():
            new_name = new_name.replace(placeholder, str(value))
        
        # Clean up the filename
        new_name = re.sub(r'[<>:"/\\|?*]', '_', new_name)  # Replace invalid chars
        new_name = re.sub(r'_+', '_', new_name)  # Remove multiple underscores
        new_name = new_name.strip('_')  # Remove leading/trailing underscores
        
        return new_name + extension
    
    def rename_files_automatically(self, directory):
        """Automatically rename all image files to image_001.jpg, image_002.jpg, etc."""
        directory = Path(directory)
        if not directory.exists():
            print(f"Error: Directory '{directory}' does not exist")
            return
        
        image_files = self.get_image_files(directory)
        if not image_files:
            print(f"No image files found in '{directory}'")
            return
        
        print(f"Found {len(image_files)} image files to rename")
        
        # Create a list to track which files to rename (avoid conflicts)
        files_to_rename = []
        
        for i, file_path in enumerate(image_files, 1):
            # Use original extension but change to .jpg if needed
            original_ext = file_path.suffix.lower()
            new_name = f"{i:03d}.jpg"  # Always use .jpg extension
            new_path = file_path.parent / new_name
            
            files_to_rename.append((file_path, new_path, new_name))
        
        # Perform the renaming
        for file_path, new_path, new_name in files_to_rename:
            # Skip if target already exists and is not the same file
            if new_path.exists() and new_path != file_path:
                # Try with a different number
                counter = 1
                while new_path.exists():
                    temp_name = f"image_{len(image_files) + counter:03d}.jpg"
                    new_path = file_path.parent / temp_name
                    new_name = temp_name
                    counter += 1
            
            # Skip if names are the same
            if new_path.name == file_path.name:
                print(f"Skipped: {file_path.name} (already has correct name)")
                self.skipped_count += 1
                continue
            
            try:
                file_path.rename(new_path)
                print(f"Renamed: {file_path.name} -> {new_name}")
                self.renamed_count += 1
            except Exception as e:
                print(f"Error renaming {file_path.name}: {e}")
                self.skipped_count += 1
        
        print(f"\nSummary:")
        print(f"Renamed: {self.renamed_count} files")
        print(f"Skipped: {self.skipped_count} files")

def main():
    # Ask user for the image directory path
    print("Image File Renamer - Converts all images to 0001.jpg, 0002.jpg, 0003.jpg, etc.")
    print("\nExamples of paths you can use:")
    print("  ./images          (images folder in current directory)")
    print("  ../photos         (photos folder in parent directory)")
    print("  C:\\Users\\Name\\Pictures  (absolute Windows path)")
    print("  /home/user/pics   (absolute Linux/Mac path)")
    
    # Get directory path from user
    image_dir = input("\nEnter the path to your images folder: ").strip()
    
    # Handle empty input (use current directory)
    if not image_dir:
        image_dir = os.getcwd()
        print(f"Using current directory: {image_dir}")
    else:
        # Convert to absolute path
        image_dir = os.path.abspath(image_dir)
        print(f"Target directory: {image_dir}")
    
    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist!")
        return
    
    print("This will rename all image files to: 0001.jpg, 0002.jpg, 0003.jpg, etc.")
    
    # Ask for confirmation
    response = input("Do you want to continue? (y/N): ").strip().lower()
    if response != 'y' and response != 'yes':
        print("Operation cancelled.")
        return
    
    # Create renamer and rename files automatically
    renamer = ImageRenamer()
    renamer.rename_files_automatically(image_dir)

if __name__ == "__main__":
    main()