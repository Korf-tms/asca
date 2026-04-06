import pathlib as pl

def create_folder(folder_path):
    folder = pl.Path(folder_path)

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

def get_unique_path(base_name, output_file=None, data_folder=None, name=None, suffix="hdf5"):
    if output_file is not None:
        output_path = pl.Path(output_file)
        if not output_path.parent.exists() and data_folder is not None:
            output_path = pl.Path(data_folder) / output_path
    else:
        output_name = f"{base_name}_{name}"
        output_path = pl.Path(data_folder) / f"{output_name}.{suffix}"

    file_num = 0
    unique_path = output_path

    while unique_path.exists():
        unique_path = output_path.with_name(f"{output_path.stem}_{file_num}{output_path.suffix}")
        file_num += 1

    return unique_path