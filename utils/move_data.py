import os
import shutil
from tqdm import tqdm


def copy_nii_files_with_progress(src_folder, dest_folder):
    """
    Copy all .nii.gz files from src_folder to dest_folder with tqdm progress bar.

    Args:
        src_folder (str): Source folder containing .nii.gz files.
        dest_folder (str): Destination folder to copy files to.
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    name_list=  ['hepaticvessel_086.nii.gz', 'hepaticvessel_097.nii.gz', 'hepaticvessel_369.nii.gz', 'hepaticvessel_431.nii.gz', 'hepaticvessel_189.nii.gz', 'hepaticvessel_150.nii.gz']
    # name_list = ['hepaticvessel_083.nii.gz', 'hepaticvessel_293.nii.gz', 'hepaticvessel_102.nii.gz', 'hepaticvessel_274.nii.gz', 'hepaticvessel_131.nii.gz', 'hepaticvessel_296.nii.gz', 'hepaticvessel_386.nii.gz', 'hepaticvessel_194.nii.gz', 'hepaticvessel_321.nii.gz', 'hepaticvessel_092.nii.gz', 'hepaticvessel_023.nii.gz', 'hepaticvessel_200.nii.gz', 'hepaticvessel_096.nii.gz', 'hepaticvessel_307.nii.gz', 'hepaticvessel_079.nii.gz', 'hepaticvessel_340.nii.gz', 'hepaticvessel_279.nii.gz', 'hepaticvessel_398.nii.gz', 'hepaticvessel_385.nii.gz', 'hepaticvessel_019.nii.gz', 'hepaticvessel_286.nii.gz', 'hepaticvessel_309.nii.gz', 'hepaticvessel_089.nii.gz', 'hepaticvessel_425.nii.gz', 'hepaticvessel_262.nii.gz', 'hepaticvessel_443.nii.gz', 'hepaticvessel_256.nii.gz', 'hepaticvessel_039.nii.gz', 'hepaticvessel_409.nii.gz', 'hepaticvessel_384.nii.gz', 'hepaticvessel_050.nii.gz', 'hepaticvessel_160.nii.gz', 'hepaticvessel_290.nii.gz', 'hepaticvessel_422.nii.gz', 'hepaticvessel_407.nii.gz', 'hepaticvessel_369.nii.gz', 'hepaticvessel_280.nii.gz', 'hepaticvessel_284.nii.gz', 'hepaticvessel_433.nii.gz', 'hepaticvessel_456.nii.gz', 'hepaticvessel_423.nii.gz', 'hepaticvessel_111.nii.gz', 'hepaticvessel_208.nii.gz', 'hepaticvessel_322.nii.gz', 'hepaticvessel_027.nii.gz', 'hepaticvessel_011.nii.gz', 'hepaticvessel_234.nii.gz', 'hepaticvessel_248.nii.gz', 'hepaticvessel_406.nii.gz', 'hepaticvessel_053.nii.gz', 'hepaticvessel_282.nii.gz', 'hepaticvessel_225.nii.gz', 'hepaticvessel_255.nii.gz', 'hepaticvessel_368.nii.gz', 'hepaticvessel_044.nii.gz', 'hepaticvessel_110.nii.gz', 'hepaticvessel_161.nii.gz', 'hepaticvessel_094.nii.gz', 'hepaticvessel_215.nii.gz', 'hepaticvessel_240.nii.gz', 'hepaticvessel_165.nii.gz', 'hepaticvessel_291.nii.gz', 'hepaticvessel_217.nii.gz', 'hepaticvessel_013.nii.gz', 'hepaticvessel_442.nii.gz', 'hepaticvessel_085.nii.gz', 'hepaticvessel_336.nii.gz', 'hepaticvessel_371.nii.gz', 'hepaticvessel_416.nii.gz', 'hepaticvessel_259.nii.gz', 'hepaticvessel_061.nii.gz', 'hepaticvessel_411.nii.gz', 'hepaticvessel_177.nii.gz', 'hepaticvessel_404.nii.gz', 'hepaticvessel_098.nii.gz', 'hepaticvessel_420.nii.gz', 'hepaticvessel_258.nii.gz', 'hepaticvessel_042.nii.gz', 'hepaticvessel_195.nii.gz', 'hepaticvessel_314.nii.gz', 'hepaticvessel_051.nii.gz', 'hepaticvessel_223.nii.gz', 'hepaticvessel_270.nii.gz', 'hepaticvessel_184.nii.gz', 'hepaticvessel_159.nii.gz', 'hepaticvessel_196.nii.gz', 'hepaticvessel_318.nii.gz', 'hepaticvessel_088.nii.gz', 'hepaticvessel_246.nii.gz', 'hepaticvessel_341.nii.gz']
    # name_list = ['PuJian_20241109_022_0001.nii.gz', 'PuJian_20241109_088_0002.nii.gz', 'PuJian_20241109_006_0001.nii.gz', 'PuJian_20241109_001_0002.nii.gz', 'PuJian_20241109_021_0002.nii.gz', 'PuJian002_0002.nii.gz', 'PuJian044_0001.nii.gz', 'PuJian033_0001.nii.gz', 'PuJian043_0002.nii.gz', 'PuJian014_0002.nii.gz', 'PuJian_20241109_067_0001.nii.gz', 'PuJian_20241109_010_0002.nii.gz', 'PuJian_20241109_047_0001.nii.gz', 'PuJian037_0002.nii.gz', 'PuJian004_0002.nii.gz', 'PuJian_20241109_034_0001.nii.gz', 'PuJian_20241109_072_0001.nii.gz', 'PuJian021_0002.nii.gz', 'PuJian_20241109_026_0002.nii.gz', 'PuJian_20241109_019_0002.nii.gz', 'PuJian_20241109_080_0002.nii.gz', 'PuJian_20241109_090_0002.nii.gz', 'PuJian005_0002.nii.gz', 'PuJian018_0002.nii.gz', 'PuJian_20241109_046_0002.nii.gz', 'PuJian_20241109_050_0002.nii.gz', 'PuJian_20241109_039_0001.nii.gz', 'PuJian006_0001.nii.gz', 'PuJian053_0002.nii.gz', 'PuJian_20241109_005_0001.nii.gz', 'PuJian012_0002.nii.gz', 'PuJian_20241109_073_0002.nii.gz', 'PuJian_20241109_076_0001.nii.gz', 'PuJian_20241109_058_0002.nii.gz', 'PuJian_20241109_051_0002.nii.gz', 'PuJian022_0002.nii.gz', 'PuJian029_0001.nii.gz', 'PuJian_20241109_028_0001.nii.gz', 'PuJian006_0002.nii.gz', 'PuJian031_0001.nii.gz', 'PuJian032_0002.nii.gz', 'PuJian049_0002.nii.gz', 'PuJian027_0001.nii.gz', 'PuJian_20241109_082_0001.nii.gz', 'PuJian_20241109_070_0002.nii.gz', 'PuJian_20241109_044_0001.nii.gz', 'PuJian019_0001.nii.gz', 'PuJian_20241109_006_0002.nii.gz', 'PuJian_20241109_091_0001.nii.gz', 'PuJian_20241109_041_0001.nii.gz', 'PuJian_20241109_064_0001.nii.gz', 'PuJian_20241109_073_0001.nii.gz', 'PuJian053_0001.nii.gz', 'PuJian_20241109_078_0002.nii.gz', 'PuJian040_0001.nii.gz', 'PuJian_20241109_025_0002.nii.gz', 'PuJian_20241109_036_0001.nii.gz', 'PuJian_20241109_063_0001.nii.gz', 'PuJian_20241109_074_0001.nii.gz', 'PuJian007_0001.nii.gz', 'PuJian046_0002.nii.gz', 'PuJian_20241109_067_0002.nii.gz', 'PuJian028_0001.nii.gz', 'PuJian044_0002.nii.gz', 'PuJian_20241109_085_0002.nii.gz', 'PuJian010_0002.nii.gz', 'PuJian_20241109_043_0002.nii.gz', 'PuJian_20241109_091_0002.nii.gz', 'PuJian025_0002.nii.gz', 'PuJian052_0001.nii.gz', 'PuJian_20241109_012_0002.nii.gz', 'PuJian_20241109_079_0001.nii.gz', 'PuJian048_0002.nii.gz', 'PuJian_20241109_054_0002.nii.gz', 'PuJian_20241109_005_0002.nii.gz', 'PuJian035_0001.nii.gz', 'PuJian_20241109_007_0001.nii.gz', 'PuJian_20241109_070_0001.nii.gz', 'PuJian_20241109_050_0001.nii.gz', 'PuJian_20241109_033_0001.nii.gz', 'PuJian040_0002.nii.gz', 'PuJian_20241109_033_0002.nii.gz', 'PuJian_20241109_087_0001.nii.gz', 'PuJian_20241109_019_0001.nii.gz', 'PuJian_20241109_012_0001.nii.gz', 'PuJian051_0001.nii.gz', 'PuJian_20241109_040_0002.nii.gz', 'PuJian_20241109_000_0001.nii.gz', 'PuJian_20241109_077_0002.nii.gz', 'PuJian_20241109_030_0002.nii.gz']
    # Find all .nii.gz files in the source folder
    nii_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(src_folder)
        for file in files
        if file.endswith(".nii.gz") and file in name_list
    ]

    # Copy files with progress bar
    for file_path in tqdm(nii_files, desc="Copying files", unit="file"):
        shutil.copy(file_path, os.path.join(dest_folder, os.path.basename(file_path)))


# Example usage
src_folder = "/data7/hhy/temp/vessel-venous"  
# src_folder = "/data7/hhy/temp/vessel-arterial"
dest_folder = "/data1/zzh/VesselSegModel/old/IRCADb"
copy_nii_files_with_progress(src_folder, dest_folder)
