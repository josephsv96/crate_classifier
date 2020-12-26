from pathlib import Path
from utils import create_output_folder, read_image, write_image


def get_working_dir(working_dir):
    working_dir = Path(input("Enter path of image folder: "))
    output_dir = create_output_folder(working_dir)
    print(output_dir)
    return working_dir, output_dir


def process_api_images(working_dir, output_dir):
    bmp_files = list(working_dir.glob('**/*.bmp'))
    jpg_files = list(working_dir.glob('**/*.jpg'))
    png_files = list(working_dir.glob('**/*.jpeg'))
    jpeg_files = list(working_dir.glob('**/*.png'))
    img_files = bmp_files + jpg_files + png_files + jpeg_files

    for i, img_file in enumerate(img_files):
        src = read_image(img_file)
        new_file_path = output_dir / ("IMG_" + str(i) + (".bmp"))
        print(new_file_path)
        write_image(src, str(new_file_path))

    return True


def main():
    working_dir, output_dir = get_working_dir()
    process_api_images(working_dir, output_dir)


if __name__ == "__main__":
    main()
