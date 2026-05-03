import os
import struct
import numpy as np
from PIL import Image
import json

def parse_etl1(file_path):
    record_size = 2052
    images = []
    labels = []

    with open(file_path, 'rb') as f:
        while True:
            record_data = f.read(record_size)
            if len(record_data) < record_size:
                break
            
            header = struct.unpack('>H2sH6BI4H4B4x2016s4x', record_data)
            
            # THE ROMAJI FIX:
            # Decode the 2-byte string into ASCII text and strip any spaces
            # Example: b'KA' becomes 'KA', b' A' becomes 'A'
            char_string = header[1].decode('ascii', errors='ignore').strip()
            
            img_data = header[18] 
            
            byte_array = np.frombuffer(img_data, dtype=np.uint8)
            high_bits = (byte_array >> 4) & 0x0F
            low_bits = byte_array & 0x0F
            
            pixels = np.column_stack((high_bits, low_bits)).flatten()
            img_matrix = pixels.reshape((63, 64))
            img_matrix = (img_matrix * 17).astype(np.uint8)
            
            img = Image.fromarray(img_matrix).resize((64, 64))
            
            images.append(np.asarray(img))
            labels.append(char_string)
            
    return np.array(images), np.array(labels)

def main():
    base_dir = r"C:\Users\mthar\Documents\Materi Kuliah - Copy\IPB\Semester 6\Visual Komputer Cerdas\katakanamodel3\parser"
    
    all_images = []
    all_labels = []

    print("Starting to parse ETL-1 Katakana dataset...")

    for i in range(7, 14):
        filename = f"ETL1C_{i:02d}" 
        file_path = os.path.join(base_dir, filename)
        
        if os.path.exists(file_path):
            print(f"Parsing {filename}...")
            imgs, lbls = parse_etl1(file_path)
            all_images.append(imgs)
            all_labels.append(lbls)

    raw_images = np.concatenate(all_images, axis=0)
    raw_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nFinished parsing. Total Katakana images: {len(raw_labels)}")

    # ==========================================
    # DYNAMIC ROMAJI MAPPING
    # ==========================================
    unique_romaji = np.unique(raw_labels)
    num_classes = len(unique_romaji)
    print(f"\nFound {num_classes} unique Katakana classes!")
    print(f"The Classes: {unique_romaji}")

    # Map the readable string (e.g., 'KA') to a PyTorch integer (e.g., 0)
    romaji_to_idx = {str(romaji): int(idx) for idx, romaji in enumerate(unique_romaji)}
    idx_to_romaji = {int(idx): str(romaji) for idx, romaji in enumerate(unique_romaji)}

    labels = np.array([romaji_to_idx[r] for r in raw_labels])

    # Save to disk
    # Save to a single compressed .npz archive
    np.savez_compressed(os.path.join(base_dir, 'etl1_katakana.npz'), images=raw_images, labels=labels)
    
    with open(os.path.join(base_dir, 'etl1_class_mapping.json'), 'w') as f:
        json.dump(idx_to_romaji, f)

    print("\nSuccess! Saved clean data ready.")

if __name__ == '__main__':
    main()