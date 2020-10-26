"""Converts all .h5 model files to .h5 model weight files"""

from tensorflow.keras.models import load_model
from pathlib import Path
import shutil
from tqdm import tqdm


def create_output_folder(working_dir):

    output_dir = working_dir / (working_dir.stem + "_weights")
    if output_dir.is_dir():
        print("Output folder exists...")
        try:
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        except OSError as err:
            print("Error: %s : %s" % (output_dir, err.strerror))
            print("Overwriting...")
    else:
        print("Output folder created")
        output_dir.mkdir()

    return output_dir


def main():
    working_dir = Path.cwd()
    output_dir = create_output_folder(working_dir)
    mod_files = list(working_dir.glob("**/*.h5"))

    for file in tqdm(mod_files):
        try:
            model = load_model(str(file))
            model.save_weights(str(output_dir / (file.stem + "_w.h5")))

        # Add exception if .h5 weight file also exits in the working dir
        except expression as err:
            pass


if __name__ == "__main__":
    main()
