# Crate Classification: Semantic Segmentation using Multi-exposure images

## !Under Construction

## Introduction: motivation and basic overview in 2-50 sentences

## Setup: instructions on how to run the code / simulation

1. short version for fast setup (start with default settings)
2. long version for detailed setup (different params, custom images etc.)

## Conventions

* File Naming Convention
  * Followed by both source files and outputs of pkg_1b.
  * img_file

    ```python
    ['img_000001_a.bmp',..., 'img_000512_c.bmp',...]
    ```

  * ann_file

    ```python
    ['img_000001_a.cmp',..., 'img_000512_c.cmp',...]
    ```

* Processed image and annotation files

## Structure

### pkg_0: Annotated images

### pkg_0 -> pkg_1: Generte the dataset for the NN

1. Data Loader: Reads images and annotations from source folder.
2. Dataset Check: Checks for missing annotations files.
3. Statistics: Finds the distribution of classes by pixel density.
4. Augmenter: Genertes augmented images and corressponding annotations.

## Unit tests

* Unit test look for consistency of code and the compatibility of the defined configurations `pkg_1_config.json`, `pkg_2_config.json`
* Unit tests should be run from `crate_classifier/`:

    ```bash
    pytest tests/
    ```

## Installation: instructions on how to install the environment or software

1. Prerequisites: Python 3.7, NVIDIA GPU (with compute capability >= 7.5)
2. Installation: Install `requirements.txt` into a new python environment as follows:

   ```bash
   pip install -r requirements.txt
   ```

3. Instructions:
   Create a new environment using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [virturalenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment). Using either works, although conda is recommended.

   1. For conda, we can create a new environment and install all requirements.

    ```bash
    conda env create environment.yml
    ```

   2. For virtualenv, create and activate the environment, then install from requirements.txt.

    ```bash
    virtualenv crate_classifier     # creating new env
    mypthon\Scripts\activate        # activate env
    pip install -r requirements.txt
    ```

## Documentation: detailed explanation of important things. (! don't forget to add comments to your code)

## Development: instructions on how to expand the code and add new value to the software

1. Develop the moudules within `packages/`. They should be run from the parent directory `crate_classifier/`, using the `-m` flag.

```bash
# Example
python -m modules/pkg_1a
```

## Additional: Lecture and more about the topic

## Contact: possibility to contact administrators of the project
