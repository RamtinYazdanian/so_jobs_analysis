import feedparser
import argparse
import requests
from bs4 import BeautifulSoup

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
    :return: A list of all the ads.
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

    return list(all_ads.values())

def main():

    pass

if __name__ == '__main__':
    main()