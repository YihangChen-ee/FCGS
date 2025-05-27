import os
from huggingface_hub import hf_hub_download
from argparse import ArgumentParser
import sys

parser = ArgumentParser(description="dataset_param")
parser.add_argument("--odir", default="./DL3DV-GS-960P", type=str, help="output directory")
parser.add_argument("--file_type", default="3DGS", type=str, choices=["3DGS", "imgs_undist"], help="The file type to download. 1) 3DGS: pre-trained 3DGS. 2) imgs_undist: both undistorted images and camera poses")
parser.add_argument("--subset", default="/", type=str, choices=["/", "1K", "2K", "3K", "4K", "5K", "6K", "7K"], help="The subset of the benchmark to download. A / by default indicates downloading all subsets one by one.")
parser.add_argument("--hash", default="/", type=str, help="If set hash, this is the hash code of the scene to download")
parser.add_argument("--split", default="train", type=str, help="The training or testing split. For the testing split, chkpnt30000.pth is also provided.")
args = parser.parse_args(sys.argv[1:])

file_type = args.file_type

with open(f"all_hash_name_{args.split}.txt") as fhn:
    lines = fhn.readlines()
    lines = [line.strip() for line in lines]

lines = list(sorted(lines))

# Create download directory if it doesn't exist
download_dir = args.odir
os.makedirs(download_dir, exist_ok=True)

for line in lines:

    if args.subset in line and args.hash in line:

        print("downloading: ", line)
        try:
            K = line.split('/')[1]
            hash_name = line.split('/')[2]
            zip_filename = hash_name + '.zip'

            # Create subdirectory for K if it doesn't exist
            os.makedirs(os.path.join(download_dir, f"{file_type}", K), exist_ok=True)

            # Local paths
            local_zip_path = os.path.join(download_dir, f"{file_type}", K, zip_filename)
            local_extract_path = os.path.join(download_dir, f"{file_type}", K, hash_name)

            # Skip if already downloaded and extracted
            if os.path.exists(local_extract_path):
                print(f"✅ {line} already downloaded and extracted, skipping")
                continue

            # Download the zip file
            hf_hub_download(
                repo_id="DL3DV/DL3DV-GS-960P",
                filename=os.path.join(f"{file_type}", K, zip_filename),
                repo_type="dataset",
                local_dir=download_dir,
                local_dir_use_symlinks=False,
            )

            # Extract the zip file
            # shutil.unpack_archive(local_zip_path, os.path.join(download_dir, K))
            os.system(f"cd {os.path.join(download_dir, f'{file_type}', K)} && unzip {zip_filename}")

            # Remove the zip file after extraction
            os.remove(local_zip_path)

        except KeyboardInterrupt:
            print("\nDownload interrupted by user. Exiting...")
            exit()
        except Exception as e:
            print(f"failed: {line}, error: {str(e)}")
            with open(f"all_hash_name_{args.split}_fail_download_{file_type}.txt", 'a') as fhn:
                fhn.write(line)
                fhn.write('\n')