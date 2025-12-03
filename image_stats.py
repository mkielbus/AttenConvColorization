import os
import cv2
import numpy as np
import requests
from tqdm import tqdm
import argparse
from skimage.measure import shannon_entropy


def getImagesPaths(root_folder, extensions={'.jpg', '.jpeg', '.png', '.bmp'}):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if os.path.splitext(file)[1].lower() in extensions:
                image_paths.append(os.path.join(dirpath, file))
    return image_paths


def getImagesAmount(query: str = "*", qf: list[str] = [""], theme: str = ""):
    url = "https://api.europeana.eu/record/v2/search.json"
    params = {
             "query": query,
             "qf": qf,
             "theme": theme,
             "wskey": "strityber"
             }
    response = requests.get(url, params=params)
    data = response.json()
    return data["totalResults"]


def computeStats(image_paths):
    pixel_counts = []
    for path in tqdm(image_paths, desc="Image number"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f" Unable to load: {path}")
            continue
        height, width = img.shape[:2]
        pixel_counts.append(width * height)
    pixel_counts = np.array(pixel_counts)
    mean = pixel_counts.mean()
    std = pixel_counts.std()
    return mean, std


def estimateSNRNoReference(image):
    mean = np.mean(image)
    std = np.std(image)
    snr = mean / std
    return snr


def computeMeanStdSNR(image_paths):
    snr = []
    for path in tqdm(image_paths, desc="Image number"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f" Unable to load: {path}")
            continue
        snr.append(estimateSNRNoReference(img))
    snr = np.asarray(snr)
    mean = snr.mean()
    std = snr.std()
    return mean, std


def computeMeanStdLaplacian(image_paths):
    laplacian_std = []
    laplacian_mean = []
    for path in tqdm(image_paths, desc="Image number"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f" Unable to load: {path}")
            continue
        laplacian_mean.append(cv2.Laplacian(img, ddepth=cv2.CV_64F).mean())
        laplacian_std.append(cv2.Laplacian(img, ddepth=cv2.CV_64F).std())
    laplacian_mean = np.asarray(laplacian_mean)
    laplacian_std = np.asarray(laplacian_std)
    mean_mean = laplacian_mean.mean()
    mean_std = laplacian_std.mean()
    std_mean = laplacian_mean.std()
    std_var = laplacian_std.std()
    return mean_mean, mean_std, std_mean, std_var


def computeMeanStdShannonEntropy(image_paths):
    entropy = []
    for path in tqdm(image_paths, desc="Image number"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f" Unable to load: {path}")
            continue
        entropy.append(shannon_entropy(img))
    entropy = np.asarray(entropy)
    mean = entropy.mean()
    std = entropy.std()
    return mean, std


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_folder_path", type=str, help="Folder zawierający badane obrazy")
    args = parser.parse_args()
    return args


def main():
    args = getCMDArgs()
    sizes = ("small", "medium", "large", "extra_large")
    gray_images_amount = getImagesAmount(qf=["IMAGE_GREYSCALE:true", "TYPE:IMAGE"])
    print(f"Ilość obrazów w skali szarości (Europeana): {gray_images_amount}")
    for size in sizes:
        gray_images_amount = getImagesAmount(qf=["IMAGE_GREYSCALE:true", "TYPE:IMAGE", f"IMAGE_SIZE:{size}"])
        print(f"Ilość obrazów w skali szarości o rozmiarze {size} (Europeana): {gray_images_amount}")
    color_images_amount = getImagesAmount(qf=["IMAGE_COLOUR:true", "TYPE:IMAGE"])
    print(f"Ilość obrazów kolorowych (Europeana): {color_images_amount}")
    for size in sizes:
        color_images_amount = getImagesAmount(qf=["IMAGE_COLOUR:true", "TYPE:IMAGE", f"IMAGE_SIZE:{size}"])
        print(f"Ilość obrazów kolorowych o rozmiarze {size} (Europeana): {color_images_amount}")
    folder = args.images_folder_path
    image_paths = getImagesPaths(folder)
    pixel_mean, pixel_std = computeStats(image_paths)
    print(f"Średni rozmiar w pikselach: {round(pixel_mean, 2)}, Odchylenie standardowe rozmiaru w pikselach: \
{round(pixel_std, 2)}")
    snr_mean, snr_std = computeMeanStdSNR(image_paths)
    print(f"Średnia miara SNR obrazów: {round(snr_mean, 2)}, Odchylenie standardowe SNR obrazów: \
{round(snr_std, 2)}")
    laplace_mean_mean, laplace_std_mean, laplace_mean_std, laplace_std_std =\
        computeMeanStdLaplacian(image_paths)
    print(f"Średnia miara średniego Laplasjanu obrazów: {round(laplace_mean_mean, 2)}\n\
Średnia miara odchylenia standardowego Laplasjanu obrazów: {round(laplace_std_mean, 2)}\n\
Odchylenie standardowe średniego Laplasjanu obrazów: {round(laplace_mean_std, 2)}\n\
Odchylenie standardowe odchylenia standardowego Laplasjanu obrazów: {round(laplace_std_std, 2)}")
    entropy_mean, entropy_std = computeMeanStdShannonEntropy(image_paths)
    print(f"Średnia miara Entropii Shannona obrazów: {round(entropy_mean, 2)}, Odchylenie standardowe \
Entropii Shannona obrazów: {round(entropy_std, 2)}")


if __name__ == "__main__":
    main()
