import feedparser
import argparse
import requests
import os
import json
from bs4 import BeautifulSoup
from utilities.common_utils import make_sure_path_exists

root_url = 'https://web.archive.org/'

def crawl_individual_page(page_datetime_or_url):
    try:
        # To see if the argument is just the datetime, or a full URL. If it's the datetime, it's parsable as an int.
        datetime_int = int(page_datetime_or_url)
        true_url = 'https://web.archive.org/web/%s/http://stackoverflow.com/jobs/feed' % (str(page_datetime_or_url))
    except ValueError as e:
        true_url = page_datetime_or_url

    response = requests.get(true_url)
    return response.text

def parse_individual_page(rss_text):
    parsed_text = feedparser.parse(rss_text)
    individual_ads = parsed_text['entries']
    for entry in individual_ads:
        entry['title_pub'] = entry['title'] + '___' + entry['published']
        entry['title_pub_updated'] = entry['title'] + '___' + entry['published'] + '___' + entry['updated']
    return individual_ads

def crawl_entire_page(filenames):
    """
    Parses all the RSS feeds referenced from a list of pages.
    :param filename: A list of the names of files containing the calendar divs.
    :return: A dictionary mapping each datetime (string) to the rss content of the page.
    """
    all_pages = dict()
    for filename in filenames:
        page_contents = '\n'.join(open(filename, mode='r', encoding='utf8').readlines())
        soup = BeautifulSoup(page_contents)
        urls = [root_url+x['href'] for x in soup.find_all('a') if '/web/' in str(x)]
        all_pages.update({current_url.split('/web/')[1].split('/http')[0]:
                         crawl_individual_page(current_url) for current_url in urls})
    return all_pages

def parse_all_pages(all_pages, keep_all_updates=False):
    """
    Parses the RSS content of all pages, turning it into a list of ads.
    :param all_pages: The dictionary mapping each datetime string into its RSS content (raw string).
    :param keep_all_updates: Whether to keep all updates of each ad in the final list, or to only keep the final
    update.
    :return: A dict of all the ads.
    """

    all_ads = dict()

    for current_datetime in sorted(list(all_pages.keys())):
        page_content = all_pages[current_datetime]
        current_ads = parse_individual_page(page_content)
        if keep_all_updates:
            current_ads = {entry['title_pub_updated']: entry for entry in current_ads}
        else:
            current_ads = {entry['title_pub']: entry for entry in current_ads}
        all_ads.update(current_ads)

    return all_ads

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--page_files', type=str, required=True)
    parser.add_argument('--keep_all_updates', action='store_true')
    parser.add_argument('--save_originals', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    list_of_filenames = [os.path.join(args.page_files, x) for x in os.listdir(args.page_files)]
    all_pages = crawl_entire_page(list_of_filenames)
    parsed_ads = parse_all_pages(all_pages, args.keep_all_updates)
    make_sure_path_exists(args.output_dir)

    with open(os.path.join(args.output_dir, 'all_ads.json'), 'w') as f:
        json.dump(parsed_ads, f)

    if args.save_originals:
        with open(os.path.join(args.output_dir, 'jobs_rss_raw.json'), 'w') as f:
            json.dump(all_pages, f)


if __name__ == '__main__':
    main()