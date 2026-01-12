# First time running the script will take longer because models need to download, subsequent runs will be faster
# Expecting PDFs to always be in the same resolution and format.
# Assumes each PDF only has one page
# Assuming there are either 0, 1 or 2 BOM tables in a single PDF
# If there are 2 BOMS in one pdf, assumes the second one (leftmost) does not surpass half the vertical length of the page

import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
from time import time
from tqdm import tqdm
import argparse

# These are specfic to our BOM tables and to default resolution
NEW_LINE_THRESH = 37

def norm(coords, width, height):
    return (int(coords[0]*width), int(coords[1]*height), int(coords[2]*width), int(coords[3]*height))

def load_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    page_image = images[0] # NOTE: we could add support for pdfs with multiple pages
    width, height = page_image.width, page_image.height
    right_crop_region = (0.70475, 0.045208, 0.9795, 0.81)
    left_crop_region = (0.43875, 0.044822, 0.70825, 0.5)

    cropped_right = page_image.crop(norm(right_crop_region, width, height))
    cropped_left = page_image.crop(norm(left_crop_region, width, height))

    # cropped_right.save("right_crop.png")
    # cropped_left.save("left_crop.png")

    return [np.array(cropped_right), np.array(cropped_left)]


def parse_bom(table_image):
    results = ocr.predict(table_image)
    
    qty_left, qty_right = None, None
    nd_left, nd_right = None, None
    desc_left, desc_right = None, None

    output = []
    current_y = None
    current_row_idx = None

    def is_between(coords, left, right):
        return coords[0] >= left and coords[2] <= right

    for res in results:
        # res.print()
        bom_title_found = False
        desc_found = False
        nd_found = False
        qty_found = False
        for i, text in enumerate(res["rec_texts"]):
            coords = res["rec_boxes"][i].tolist()
            
            if text in ["BILL OF MATERIALS", "BILL", "BILL OF"]: # Can be used to return multiple BOM tables
                bom_title_found = True
            
            if bom_title_found:
                # Start reading data after
                if text == "QTY":
                    qty_left = coords[0] - 30
                    qty_right = coords[2] + 30
                    qty_found = True
                    # qty_crop = Image.fromarray(table_image[:, qty_left:qty_right, :], "RGB")
                    # qty_crop.save("qty_crop.png")
                elif text == "ND":
                    nd_left = coords[0] - 60
                    nd_right = coords[2] + 60
                    nd_found = True
                    # nd_crop = Image.fromarray(table_image[:, nd_left:nd_right, :], "RGB")
                    # nd_crop.save("nd_crop.png")
                elif text == "DESCRIPTION":
                    desc_left = coords[0] - 230
                    desc_right = coords[2] + 230
                    desc_found = True
                    # desc_crop = Image.fromarray(table_image[:, desc_left:desc_right, :], "RGB")
                    # desc_crop.save("desc_crop.png")
            
            if qty_found and desc_found and nd_found:
                if current_row_idx is None:
                    current_y = coords[3]
                    current_row_idx = 0
                    output.append({"QTY": None, "ND": "", "DESC": ""})
                elif (coords[3] - current_y) > NEW_LINE_THRESH:
                    # We have hit a new row
                    current_row_idx += 1
                    output.append({"QTY": None, "ND": "", "DESC": ""})

                if is_between(coords, qty_left, qty_right):
                    # Found QTY entry
                    output[current_row_idx]["QTY"] = float(text.replace("M", ""))
                    current_y = coords[3]
                elif is_between(coords, nd_left, nd_right):
                    # Found one (of possibly many) ND entries, accumulate
                    output[current_row_idx]["ND"] += text # Accumulate entries
                    current_y = coords[3]
                elif is_between(coords, desc_left, desc_right):
                    # Found one (of possibly many) DESC entries, accumulate
                    output[current_row_idx]["DESC"] += text # Accumulate entries
                    current_y = coords[3]


    # Clean up any entries that have missing QTY
    cleaned_output = [dic for dic in output if dic["QTY"] is not None and dic["DESC"] != ""]

    # Fix common OCR reading mistakes for "ND" column
    for dic in cleaned_output:
        new_string = dic["ND"].replace("{", '″').replace("X", "x").replace('\"', "″").replace('”', "″").replace("$", "").replace("^", "").replace("\\", "").strip()
        if not new_string.endswith("″"):
            new_string += "″"
        dic["ND"] = new_string

    return cleaned_output


# START PROGRAM:
# pdf_directory = "test"
# output_dir = "test_csv_out"

parser = argparse.ArgumentParser()
parser.add_argument("pdf_directory", default="./pdfs")
args = parser.parse_args()
pdf_directory = args.pdf_directory
output_dir = "./csv"
if pdf_directory is not None and pdf_directory != "":
    print(f"Processing pdfs contained in directory: {pdf_directory}")
    output_dir = f"{pdf_directory}_csv_out"

start_time = time()
# Initialize OCR for detection only (faster)
ocr = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False)#,
    #det_limit_side_len=4000)

# For running on specific files
filename1 = '60-LC-4316-2_00.pdf' 
filename2 = '60-PO-4233_00.pdf'
select_files = None # [filename2]

global_output = []
os.makedirs(output_dir, exist_ok=True)
file_count = 0
for filename in select_files or tqdm(os.listdir(pdf_directory)):
    file_count += 1
    file_path = f"{pdf_directory}/{filename}"

    table_crops = load_pdf(file_path)
    file_output = []
    for i, table_crop in enumerate(table_crops):
        output = parse_bom(table_crop)
        if len(output) == 0:
            # print("No BOM found")
            pass
        else:
            # print("Detected BOM:", output)
            file_output.extend(output)
        
    df = pd.DataFrame(file_output)
    df.index = df.index + 1
    df.to_csv(f"{output_dir}/{filename.replace(".pdf", "")}_BOM.csv")

    global_output.extend(file_output)


# Finally save the global BOM of all files in directory

# Find all descs that match if we remove whitespaces, and rename them all to one of the versions with whitespaces for readability
no_whitespace_lookup = {} # [no_spaces]: [first version with spaces]
for dic in global_output:
    no_spaces = dic["DESC"].replace(" ", "")
    if no_whitespace_lookup.get(no_spaces, None) is not None:
        dic["DESC"] = no_whitespace_lookup[no_spaces] # Rename to first version with spaces
    else:
        no_whitespace_lookup[no_spaces] = dic["DESC"]

df = pd.DataFrame(global_output)
df.to_csv(f"{output_dir}/Global_BOM_raw.csv", index = False)

total_df = df.groupby(['DESC', 'ND'], as_index=False)['QTY'].sum()
total_df.to_csv(f"{output_dir}/Global_BOM_totals.csv", index=False, columns=["QTY", "ND", "DESC"])

print(f"DONE {file_count} files in {time()-start_time:.4f}s")

# LOGIC:
# Load pdf into image
# Crop out both potential sections where the tables can be
#   The left table will have a higher crop, we will assume the second table never gets too long
# We feed the right table to the OCR
# If BOM keyword is detected, we flag that we should repeat next steps for left table too
#   Locate x min&max for QTY column
#   Locate x min&max for ND column
#   Locate x min&max for DESC column
#   Search for next entry that fits in either QTY, ND, or DESC columns
#       If found, search for *all* next entries that fits in DESC col and match together (for ND too) if QTY col, just search for one entry
#           But we also need to check y dist, only allow words that count as second input line (same block), but not new row
# Convert resluting dict to csv
