import requests
import numpy as np
import cv2
from tqdm import tqdm
import argparse


def checkGreyScale(image) -> bool:
    b, g, r = cv2.split(image)
    return np.allclose(r, g) and np.allclose(g, b)


def saveImagesDataset(urls: list[str], theme: str) -> None:
    iterator = 0
    for url in tqdm(urls, desc="Downloading images"):
        try:
            response = requests.get(url, timeout=60)
        except Exception:
            continue
        if response.status_code != 200:
            continue
        image_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is not None and checkGreyScale(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"./images/{theme}/{iterator}.jpg", img)
            iterator += 1


def getImagesUrls(data: dict, number_of_gathered_urls: int, number_of_images_urls: int) -> list[str]:
    images_urls = []
    added = 0
    for record in data["items"]:
        if number_of_gathered_urls + added >= number_of_images_urls:
            return images_urls
        images_urls.append(record["edmIsShownBy"][0])
        added += 1
    return images_urls


def gatherImagesUrls(query: str = "*", qf: list[str] = [""], theme: str = "ww1",
                     number_of_images_urls: int = float("inf")) -> list[str]:
    images_urls = []
    number_of_gathered_urls = 0
    url = "https://api.europeana.eu/record/v2/search.json"
    params = {
             "query": query,
             "qf": qf,
             "theme": theme,
             "media": "true",
             "wskey": "strityber",
             "cursor": "*",
             "rows": 100,
             }
    cont = True
    next_cursor = params["cursor"]
    with tqdm(total=number_of_images_urls, desc="Downloading urls") as progress_bar:
        while cont and next_cursor:
            response = requests.get(url, params=params)
            data = response.json()
            if "nextCursor" not in data.keys():
                next_cursor = ""
            else:
                next_cursor = data["nextCursor"]
            params["cursor"] = next_cursor
            prev_length = len(images_urls)
            images_urls.extend(getImagesUrls(data, number_of_gathered_urls, number_of_images_urls))
            number_of_gathered_urls += (len(images_urls) - prev_length)
            if number_of_gathered_urls >= number_of_images_urls:
                cont = False
            progress_bar.update(len(images_urls) - prev_length)
    return images_urls


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('query_path', type=str, help="Plik zawierający zapytania")
    parser.add_argument('theme', type=str, help="Motyw obrazów")
    parser.add_argument('number_of_images', type=str, help="Liczba obrazów do pobrania")
    args = parser.parse_args()
    return args


def readQueryFile(filename: str) -> dict:
    query = {"q": [], "qf": []}
    with open(filename, "r") as file_handle:
        for line in file_handle:
            line = line.strip()
            query_type, query_value = line.split(",")
            query[query_type].append(query_value)
    return query


def main():
    args = getCMDArgs()
    query_dict = readQueryFile(args.query_path)
    images_urls = gatherImagesUrls(query=query_dict["q"], qf=query_dict["qf"], theme=args.theme,
                                   number_of_images_urls=int(args.number_of_images))
    saveImagesDataset(images_urls, args.theme)


if __name__ == "__main__":
    main()
