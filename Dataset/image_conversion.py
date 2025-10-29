import os
import numpy as np
import cv2

# Input and output folders
input_folder = r"C:\Users\Talifhani Nemaangani\Documents\Dataset_conversion\Dataset\L1-NOLATOB-275-GOOD"
output_folder = r"C:\Users\Talifhani Nemaangani\Documents\Dataset_conversion\Dataset_png\L1-NOLATOB-275ML-GOOD"

os.makedirs(output_folder, exist_ok=True)

# Image dimensions
WIDTH, HEIGHT = 1080, 1080

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".rg"):
        input_path = os.path.join(input_folder, filename)

        try:
            # Read raw binary file
            data = np.fromfile(input_path, dtype=np.uint8)

            # Ensure correct size
            if data.size != WIDTH * HEIGHT:

                # Reshape into 2D image
                bayer = data.reshape((800, 800))

               # Convert Bayer → RGB (assume RGGB pattern, common for sensors)
                rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)

                # Save as PNG
                output_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, rgb)

                print(f"[OK] Converted: {filename} -> {output_filename}")

                
                print(f"[ERROR] {filename}: unexpected size ({data.size} bytes)")
                continue

            # Reshape into 2D image
            bayer = data.reshape((HEIGHT, WIDTH))

            # Convert Bayer → RGB (assume RGGB pattern, common for sensors)
            rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)

            # Save as PNG
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, rgb)

            print(f"[OK] Converted: {filename} -> {output_filename}")

        except Exception as e:
            print(f"[ERROR] Could not process {filename}: {e}")
