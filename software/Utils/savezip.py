import zipfile, os


def get_all_file_paths(directory):
    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    # returning all file paths
    return file_paths


def main(directory):
    # calling function to get all file paths in the directory
    file_paths = get_all_file_paths(directory)

    # writing files to a zipfile
    with zipfile.ZipFile(directory+'.zip', 'w') as zip:
        # writing each file one by one
        for file in file_paths:
            zip.write(file)
        zip.close()

    print('All files zipped successfully!')


if __name__ == "__main__":
    start_dir = os.getcwd()
    for path in os.listdir(start_dir):
        temp_path = start_dir + '/' + path
        print(path)
        if os.path.isdir(temp_path):
            main(path)