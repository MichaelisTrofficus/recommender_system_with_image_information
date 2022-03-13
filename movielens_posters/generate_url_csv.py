from imdb_utils import generate_item_to_imdb_url_csv, generate_item_to_poster_url_csv

GENERATE_IMDB_URL_CSV = False
GENERATE_POSTER_URL_CSV = True

if GENERATE_IMDB_URL_CSV:
    error_ids = generate_item_to_imdb_url_csv(ml100k_path="../ml-100k", imdb_domain="http://www.imdb.com",
                                              output_csv_path="./item_imdb_url.csv")
    print("Ids with errors: ", error_ids)

if GENERATE_POSTER_URL_CSV:
    error_ids = generate_item_to_poster_url_csv(item_imdb_url_path="./item_imdb_url.csv",
                                                output_csv_path="./item_poster_url.csv")

    print("Ids with errors: ", error_ids)

# import csv
# import urllib
# from bs4 import BeautifulSoup
#
#
# with open("./item_imdb_url.csv", "r", newline="") as in_csv:
#     reader = csv.DictReader(in_csv, fieldnames=["item_id", "item_url"])
#     for row in reader:
#         item_id, item_url = row['item_id'], row['item_url']
#         print(item_id, item_url)
#         with urllib.request.urlopen(item_url) as response:
#             html = response.read()
#             soup = BeautifulSoup(html, "html.parser")
#             image_url = soup.find('div', class_='poster')
#             image_url = "".join(image_url.partition("_")[0]) + ".jpg"
#             print(image_url)
