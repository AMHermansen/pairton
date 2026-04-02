import argparse
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot
import yaml


def load_variable_structure(config_file):
    """Load variable structure from YAML config file"""
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config["variable_structure"]


def compute_matchability(arr: np.ndarray):
    matched = arr != 0
    return np.sum(2 ** (arr[matched] - 1))


def compute_n_matched_jets(arr: np.ndarray):
    return np.sum(arr != 0)


def write_df_to_hdf5(
    ak_array: ak.Array,
    variable_structure: dict[str, list[str]],
    hdf_path: Path,
    *,
    matchability_key: str = "matchability",
    n_matched_jets_key: str = "n_matched_jets",
):
    mode = "a" if hdf_path.exists() else "w"
    with h5py.File(hdf_path, mode) as hdf_file:
        # Store variable names as attributes (only once when creating the file)
        if mode == "w":
            for key, vars in variable_structure.items():
                # Create a separate dataset for variable names
                hdf_file.create_dataset(
                    f"{key}_variables",
                    data=np.array(vars, dtype="S"),  # Store as byte strings
                )

        # Process the data arrays
        for key, vars in variable_structure.items():
            data = ak.to_dataframe(ak_array[vars]).to_numpy()
            maxshape = (None,) + data.shape[1:]
            if key not in hdf_file:
                hdf_file.create_dataset(
                    key,
                    data=data,
                    maxshape=maxshape,
                    chunks=True,
                )
            else:
                # Check for empty data.
                if data.shape[0] == 0:
                    continue
                hdf_file[key].resize(hdf_file[key].shape[0] + data.shape[0], axis=0)
                hdf_file[key][-data.shape[0] :] = data

        # Handle matchability data
        if matchability_key not in hdf_file:
            matchability_data = (
                ak.to_dataframe(ak_array["jet_truthmatch"])
                .groupby(level="entry")
                .apply(lambda x: compute_matchability(x["values"].to_numpy()))
                .to_numpy()
                .squeeze()
            )
            maxshape = (None,)
            hdf_file.create_dataset(
                matchability_key,
                data=matchability_data,
                maxshape=maxshape,
                chunks=True,
            )
        else:
            matchability_data = (
                ak.to_dataframe(ak_array["jet_truthmatch"])
                .groupby(level="entry")
                .apply(lambda x: compute_matchability(x["values"].to_numpy()))
                .to_numpy()
                .squeeze()
            )
            # Check for empty data.
            if matchability_data.shape[0] == 0:
                return
            hdf_file[matchability_key].resize(
                hdf_file[matchability_key].shape[0] + matchability_data.shape[0],
                axis=0,
            )
            hdf_file[matchability_key][-matchability_data.shape[0] :] = (
                matchability_data
            )

        # Handle n_matched_jets data
        if n_matched_jets_key not in hdf_file:
            n_matched_jets_data = (
                ak.to_dataframe(ak_array["jet_truthmatch"])
                .groupby(level="entry")
                .apply(lambda x: compute_n_matched_jets(x["values"].to_numpy()))
                .to_numpy()
                .squeeze()
            )
            maxshape = (None,)
            hdf_file.create_dataset(
                n_matched_jets_key,
                data=n_matched_jets_data,
                maxshape=maxshape,
                chunks=True,
            )
        else:
            n_matched_jets_data = (
                ak.to_dataframe(ak_array["jet_truthmatch"])
                .groupby(level="entry")
                .apply(lambda x: compute_n_matched_jets(x["values"].to_numpy()))
                .to_numpy()
                .squeeze()
            )
            # Check for empty data.
            if n_matched_jets_data.shape[0] == 0:
                return
            hdf_file[n_matched_jets_key].resize(
                hdf_file[n_matched_jets_key].shape[0] + n_matched_jets_data.shape[0],
                axis=0,
            )
            hdf_file[n_matched_jets_key][-n_matched_jets_data.shape[0] :] = (
                n_matched_jets_data
            )


def read_variable_names(hdf_path: Path, key: str):
    """Read variable names from HDF5 file"""
    with h5py.File(hdf_path, "r") as hdf_file:
        if f"{key}_variables" in hdf_file:
            var_names = hdf_file[f"{key}_variables"][:]
            # Convert byte strings back to regular strings
            return [name.decode("utf-8") for name in var_names]
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert ROOT files to HDF5 format")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="YAML config file containing variable structure",
    )
    parser.add_argument("infile", help="Input ROOT file to process")
    parser.add_argument(
        "--tree",
        "-t",
        default="Delphes",
        help="Tree name in ROOT file (default: Delphes)",
    )
    parser.add_argument(
        "--step-size",
        "-s",
        default="50MB",
        help="Chunk size for processing (default: 50MB)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output HDF5 file path (default: infile with .h5 extension)",
    )

    args = parser.parse_args()

    # Load variable structure from config
    variable_structure = load_variable_structure(args.config)

    # Generate output filename
    infile_path = Path(args.infile)
    if args.output:
        outfile = Path(args.output)
    else:
        outfile = infile_path.with_suffix(".h5")

    # Check if input file exists
    if not infile_path.exists():
        raise FileNotFoundError(f"Input file '{infile_path}' not found")

    print("Starting conversion...")
    print(f"Input:  {infile_path}")
    print(f"Output: {outfile}")
    print(f"Tree:   {args.tree}")
    print(f"Chunk size: {args.step_size}")
    print(f"Variable groups: {list(variable_structure.keys())}")

    # Open ROOT file and process
    try:
        root_file = uproot.open(infile_path)
        tree = root_file[args.tree]

        chunk_count = 0
        for ak_array in tree.iterate(step_size=args.step_size):
            chunk_count += 1
            print(f"Processing chunk {chunk_count}...")
            write_df_to_hdf5(ak_array, variable_structure, outfile)

        print(f"Conversion completed. Processed {chunk_count} chunks.")

        # Print variable information
        print("\nVariable groups created:")
        for key in variable_structure.keys():
            var_names = read_variable_names(outfile, key)
            if var_names:
                print(f"  {key}: {var_names}")

    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()
