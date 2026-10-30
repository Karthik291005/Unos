import os
import shutil

csv_files = {
    "KIDPAN_DATA": r"D:\UNOS_STAR_FILES\New folder\KIDPAN_DATA_named.csv",
    "KIDPAN_ADDTL_HLA": r"D:\UNOS_STAR_FILES\New folder\KIDPAN_ADDTL_HLA_named.csv",
    "KIDPAN_IMMUNO_DISCHARGE_DATA": r"D:\UNOS_STAR_FILES\New folder\KIDPAN_IMMUNO_DISCHARGE_DATA_named.csv",
    "KIDPAN_IMMUNO_FOLLOWUP_DATA": r"D:\UNOS_STAR_FILES\New folder\KIDPAN_IMMUNO_FOLLOWUP_DATA_named.csv",
    "KIDNEY_FOLLOWUP_DATA": r"D:\UNOS_STAR_FILES\New folder\KIDNEY_FOLLOWUP_DATA_named.csv",
    "KIDPAN_FOLLOWUP_DATA": r"D:\UNOS_STAR_FILES\New folder\KIDPAN_FOLLOWUP_DATA_named.csv",
    "KIDNEY_MALIG_FOLLOWUP_DATA": r"D:\UNOS_STAR_FILES\New folder\KIDNEY_MALIG_FOLLOWUP_DATA_named.csv",
    "KIDPAN_MALIG_FOLLOWUP_DATA": r"D:\UNOS_STAR_FILES\New folder\KIDPAN_MALIG_FOLLOWUP_DATA_named.csv",
    "KIDPAN_PRA_CROSSMATCH_DATA": r"D:\UNOS_STAR_FILES\New folder\KIDPAN_PRA_CROSSMATCH_DATA_named.csv",
}

output_dir = "downloaded_csvs"
os.makedirs(output_dir, exist_ok=True)

for name, src_path in csv_files.items():
    dst_path = os.path.join(output_dir, f"{name}.csv")
    print(f"Copying {name}: {src_path} -> {dst_path}")
    shutil.copy2(src_path, dst_path)

print("Done.")