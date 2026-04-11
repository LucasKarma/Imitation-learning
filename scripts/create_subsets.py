import h5py
import numpy as np
import os

FULL_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "low_dim_v141.hdf5"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

SUBSETS = [20, 50, 100]
SEED = 42

def create_subset(n_demos):
    np.random.seed(SEED)

    with h5py.File(FULL_DATA, "r") as f_in:
        all_demos = sorted(f_in["data"].keys())
        total = len(all_demos)
        print(f"Full dataset has {total} demos")

        selected = sorted(np.random.choice(all_demos, size=n_demos, replace=False))
        out_path = os.path.join(OUTPUT_DIR, f"low_dim_{n_demos}demos.hdf5")

        with h5py.File(out_path, "w") as f_out:
            # Copy file-level attributes
            for attr_key in f_in.attrs:
                f_out.attrs[attr_key] = f_in.attrs[attr_key]

            grp = f_out.create_group("data")

            # Copy data group attributes (this includes env_args!)
            for attr_key in f_in["data"].attrs:
                grp.attrs[attr_key] = f_in["data"].attrs[attr_key]

            # Copy selected demos
            for new_idx, demo_name in enumerate(selected):
                new_name = f"demo_{new_idx}"
                f_in.copy(f"data/{demo_name}", grp, name=new_name)

            # Update total count
            grp.attrs["total"] = n_demos

            # Create mask
            if "mask" in f_in:
                mask_grp = f_out.create_group("mask")
                demo_names = [f"demo_{i}" for i in range(n_demos)]
                mask_grp.create_dataset("train", data=np.array(demo_names, dtype="S"))

        print(f"Created {out_path} with {n_demos} demos")

if __name__ == "__main__":
    for n in SUBSETS:
        create_subset(n)
    print("\nDone!")
