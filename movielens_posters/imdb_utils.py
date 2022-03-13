import logging
import os
import csv
import time
from tqdm import tqdm
from typing import List
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup


def generate_item_to_imdb_url_csv(ml100k_path: str, imdb_domain: str, output_csv_path: str) -> List[str]:
    """
    Generates a csv containing item id and IMDB url correspondence
    Args:
        ml100k_path: The main path of the Movielens100K
        imdb_domain: The IMDB domain
        output_csv_path: The output csv path

    Returns:
        A list of ids the method couldn't get the IMDB url for
    """
    error_urls = []
    with open(os.path.join(ml100k_path, 'u.item'), 'r', encoding="ISO-8859-1") as f:
        reader = csv.DictReader(f, fieldnames=["item_id", "item_name"], delimiter='|')
        for row in tqdm(reader, total=1682):
            item_id, item_name = row['item_id'], row['item_name']
            search_url = imdb_domain + f"/find?q={urllib.parse.quote_plus(item_name)}"

            with urllib.request.urlopen(search_url) as response:
                try:
                    html = response.read()
                    soup = BeautifulSoup(html, "html.parser")
                    title = soup.find('table', class_='findList').tr.a['href']
                    movie_url = imdb_domain + f"/{title}"
                    with open(output_csv_path, 'a', newline='') as out_csv:
                        writer = csv.writer(out_csv, delimiter=',')
                        writer.writerow([item_id, movie_url])
                except AttributeError:
                    logging.info("No content found for the url provided")
                    error_urls.append(item_id)
                except ConnectionError:
                    logging.info("Connection reset by peer. Waiting 5 seconds")
                    time.sleep(5)
                    error_urls.append(item_id)
    return error_urls


def generate_item_to_poster_url_csv(item_imdb_url_path: str, output_csv_path: str) -> List[str]:
    """
    Generates a csv containing item id and poster url correspondence. It also saves all images
    into posters/ folder.
    Args:
        item_imdb_url_path: The path fot the IMDB url to id csv
        output_csv_path: The output path

    Returns:
        A list of ids the method couldn't get the poster url for
    """
    no_poster_urls = []
    extension = ".jpg"

    with open(item_imdb_url_path, "r", newline="") as in_csv:
        reader = csv.DictReader(in_csv, fieldnames=["item_id", "item_url"])
        for row in tqdm(reader, total=1682):
            item_id, item_url = row['item_id'], row['item_url']
            with urllib.request.urlopen(item_url) as response:
                try:
                    html = response.read()
                    soup = BeautifulSoup(html, "html.parser")
                    image_url = soup.find("img").get("src").partition("_")[0] + extension
                    filename = "../posters/" + item_id + extension
                    with urllib.request.urlopen(image_url) as response_image:
                        with open(filename, 'wb') as out_image:
                            out_image.write(response_image.read())
                        with open(output_csv_path, 'a', newline='') as out_csv:
                            writer = csv.writer(out_csv, delimiter=',')
                            writer.writerow([item_id, image_url])
                except AttributeError:
                    logging.info("No content found for the url provided")
                    no_poster_urls.append(item_id)
                except ConnectionError:
                    logging.info("Connection reset by peer. Waiting 5 seconds")
                    time.sleep(5)
                    no_poster_urls.append(item_id)
    return no_poster_urls
