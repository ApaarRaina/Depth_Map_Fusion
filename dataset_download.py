import os
import shutil
import urllib.request
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image

DATASET_ROOT = "./dataset"

TRAIN_IMG_DIR   = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_DEPTH_DIR = os.path.join(DATASET_ROOT, "train", "depth")
TEST_IMG_DIR    = os.path.join(DATASET_ROOT, "test",  "images")
TEST_DEPTH_DIR  = os.path.join(DATASET_ROOT, "test",  "depth")

TMP_DIR = "./tmp_downloads"

KITTI_DOWNLOAD_RAW = False

def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        print(f"\r  {pct:5.1f}%  {downloaded/1e6:.1f} / {total_size/1e6:.1f} MB",
              end="", flush=True)

def make_dirs():
    for d in [TRAIN_IMG_DIR, TRAIN_DEPTH_DIR, TEST_IMG_DIR, TEST_DEPTH_DIR, TMP_DIR]:
        os.makedirs(d, exist_ok=True)

def download_file(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  Already downloaded: {os.path.basename(dest)}")
        return

    print(f"  Downloading: {os.path.basename(dest)}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=show_progress)
        print()
    except urllib.error.HTTPError as e:
        print(f"\nHTTP {e.code} error for: {url}")
        raise

def extract_zip(zip_path: str, extract_to: str):
    print(f"  Extracting: {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print("  Done.")

NYU_MAT_URL    = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
NYU_SPLITS_URL = "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"

NYU_SPLITS_MIRRORS = [
    NYU_SPLITS_URL,
    "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/splits.mat",
]

def download_splits_mat(dest: str):
    if os.path.exists(dest):
        print(f"  Already downloaded: splits.mat")
        return
    for url in NYU_SPLITS_MIRRORS:
        try:
            print(f"  Downloading splits.mat from: {url}")
            urllib.request.urlretrieve(url, dest, reporthook=show_progress)
            print()
            return
        except Exception as e:
            print(f"\nFailed ({e}), trying next mirror ...")
    raise RuntimeError("All splits.mat mirrors failed. See comments in script.")


def process_nyu():
    print("\n━━━ NYU Depth V2 (indoor) ━━━")

    mat_path   = os.path.join(TMP_DIR, "nyu_depth_v2_labeled.mat")
    split_path = os.path.join(TMP_DIR, "splits.mat")

    download_file(NYU_MAT_URL, mat_path)
    download_splits_mat(split_path)

    # Install deps if needed
    try:
        import h5py
        import scipy.io
    except ImportError:
        print("  Installing h5py and scipy ...")
        os.system("pip install h5py scipy -q")
        import h5py
        import scipy.io

    print("  Loading .mat file into memory (may take ~30s) ...")
    with h5py.File(mat_path, "r") as f:
        images = np.array(f["images"])
        depths = np.array(f["depths"])

    split      = scipy.io.loadmat(split_path)
    train_key  = "trainNdxs" if "trainNdxs" in split else "train"
    test_key   = "testNdxs"  if "testNdxs"  in split else "test"
    train_idxs = split[train_key].flatten() - 1
    test_idxs  = split[test_key].flatten()  - 1

    print(f"  Split loaded: {len(train_idxs)} train | {len(test_idxs)} test")

    def save_split(idxs, img_dir, depth_dir, split_name):
        print(f"  Saving NYU {split_name} ({len(idxs)} samples) ...")
        for i, idx in enumerate(idxs):
            tag = f"indoor_{i+1:05d}"

            rgb = images[idx].transpose(1, 2, 0).astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(img_dir, f"{tag}.png"))

            depth_mm = (depths[idx] * 1000.0).astype(np.uint16)
            Image.fromarray(depth_mm).save(os.path.join(depth_dir, f"{tag}.png"))

            if (i + 1) % 100 == 0 or (i + 1) == len(idxs):
                print(f"    {i+1}/{len(idxs)}")

    save_split(train_idxs, TRAIN_IMG_DIR, TRAIN_DEPTH_DIR, "train")
    save_split(test_idxs,  TEST_IMG_DIR,  TEST_DEPTH_DIR,  "test")

    print("NYU indoor samples saved.")
    print("To recover meters: depth_meters = depth_uint16 / 1000.0")

KITTI_URLS = {
    "rgb":       "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip",
    "annotated": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip",
    "velodyne":  "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip",
}

def process_kitti():
    print("\n━━━ KITTI Depth (outdoor) ━━━")

    rgb_zip = os.path.join(TMP_DIR, "data_depth_selection.zip")
    ann_zip = os.path.join(TMP_DIR, "data_depth_annotated.zip")

    download_file(KITTI_URLS["rgb"],       rgb_zip)
    download_file(KITTI_URLS["annotated"], ann_zip)

    if KITTI_DOWNLOAD_RAW:
        download_file(KITTI_URLS["velodyne"], os.path.join(TMP_DIR, "data_depth_velodyne.zip"))

    kitti_tmp = os.path.join(TMP_DIR, "kitti")
    os.makedirs(kitti_tmp, exist_ok=True)

    extract_zip(rgb_zip, kitti_tmp)
    extract_zip(ann_zip, kitti_tmp)

    def collect_pairs(root, split):
        root = Path(root)
        rgb_files   = sorted(root.rglob("*.png") )
        rgb_files   = [p for p in rgb_files if split in str(p)
                       and ("image" in str(p) or "rgb" in str(p))
                       and "depth" not in str(p)]
        depth_files = [p for p in sorted(root.rglob("*.png"))
                       if split in str(p) and "groundtruth" in str(p)]
        depth_map = {p.name: p for p in depth_files}
        pairs = []
        for r in rgb_files:
            d = depth_map.get(r.name)
            if d:
                pairs.append((r, d))
        return pairs

    train_pairs = collect_pairs(kitti_tmp, "train")
    val_pairs   = collect_pairs(kitti_tmp, "val")

    if not train_pairs and not val_pairs:
        print("  Using fallback matcher ...")
        all_pngs    = list(Path(kitti_tmp).rglob("*.png"))
        depth_files = [p for p in all_pngs if "groundtruth" in str(p)]
        rgb_files   = [p for p in all_pngs if "groundtruth" not in str(p)]
        depth_map   = {p.name: p for p in depth_files}
        for r in rgb_files:
            d = depth_map.get(r.name)
            if d:
                split = "val" if "val" in str(r) else "train"
                if split == "train":
                    train_pairs.append((r, d))
                else:
                    val_pairs.append((r, d))

    def save_kitti_pairs(pairs, img_dir, depth_dir, split_name):
        print(f"  Saving KITTI {split_name} ({len(pairs)} pairs) ...")
        for i, (rgb_path, depth_path) in enumerate(pairs):
            tag = f"outdoor_{i+1:05d}"
            shutil.copy2(rgb_path,   os.path.join(img_dir,   f"{tag}.png"))
            shutil.copy2(depth_path, os.path.join(depth_dir, f"{tag}.png"))
            if (i + 1) % 500 == 0 or (i + 1) == len(pairs):
                print(f"    {i+1}/{len(pairs)}")

    save_kitti_pairs(train_pairs, TRAIN_IMG_DIR, TRAIN_DEPTH_DIR, "train")
    save_kitti_pairs(val_pairs,   TEST_IMG_DIR,  TEST_DEPTH_DIR,  "test")

    print("KITTI outdoor samples saved.")
    print("To recover meters: depth_meters = depth_uint16 / 256.0")

def print_summary():
    def count(d): return len(list(Path(d).glob("*.png")))
    ti, td = count(TRAIN_IMG_DIR), count(TRAIN_DEPTH_DIR)
    ei, ed = count(TEST_IMG_DIR),  count(TEST_DEPTH_DIR)
    print("\n" + "=" * 55)
    print("All done! Final dataset:")
    print(f"""
  dataset/
  ├── train/
  │   ├── images/  ({ti} files)
  │   └── depth/   ({td} files)
  └── test/
      ├── images/  ({ei} files)
      └── depth/   ({ed} files)

  Naming:
    indoor_XXXXX.png  → NYU Depth V2
    outdoor_XXXXX.png → KITTI

  Depth recovery:
    indoor  → uint16 / 1000  = meters
    outdoor → uint16 / 256   = meters
    """)

def main():
    print("=" * 55)
    print("  Depth Dataset Downloader & Organizer")
    print("  NYU Depth V2 (indoor) + KITTI (outdoor)")
    print("=" * 55)

    make_dirs()
    process_nyu()
    process_kitti()

    print("\n  Cleaning up tmp downloads ...")
    shutil.rmtree(TMP_DIR)

    print_summary()

if __name__ == "__main__":
    main()