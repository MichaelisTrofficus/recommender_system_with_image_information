from imdb_utils import generate_item_to_imdb_url_csv, generate_item_to_poster_url_csv

GENERATE_IMDB_URL_CSV = False
GENERATE_POSTER_URL_CSV = True

if GENERATE_IMDB_URL_CSV:
    error_ids = generate_item_to_imdb_url_csv(ml100k_path="../ml-100k", imdb_domain="http://www.imdb.com",
                                              output_csv_path="./item_imdb_url.csv")
    print("Ids with errors: ", error_ids)

if GENERATE_POSTER_URL_CSV:
    error_ids = generate_item_to_poster_url_csv(item_imdb_url_path="./item_imdb_url.csv",
                                                output_csv_path="./missing_item_poster_url.csv",
                                                item_list=['1666', '1667'])

    print("Ids with errors: ", error_ids)
