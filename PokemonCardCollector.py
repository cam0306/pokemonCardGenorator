""""
Author: Cameron Knight
Description: Collects data from pokemon website for pokemon cards
"""

import bs4 as bs
import urllib
import time
import os
from urllib import request
directory = "data/cards/"
start_url = "http://www.pokemon.com/us/pokemon-tcg/pokemon-cards/"
extended_url = "?cardName=&cardText=&evolvesFrom=&card-grass=on&card-fire=on&card-water=on&card-lightning=on&card-psychic=on&card-fighting=on&card-darkness=on&card-metal=on&card-colorless=on&card-fairy=on&card-dragon=on&format=unlimited&hitPointsMin=0&hitPointsMax=250&retreatCostMin=0&retreatCostMax=5&totalAttackCostMin=0&totalAttackCostMax=5&particularArtist=&sort=name&sort=name"

num_pages = 516
def safe_soup_page(url):
    """
    safely retrives soup page
    :param url: url to fetch
    :return: the bs page in lxml format
    """
    response = None
    attempts = 0
    while response is None and attempts < 5:
        try:
            response = urllib.request.urlopen(url).read()
        except Exception as e:
            time.sleep(5)
            attempts += 1

    if response is None:
        return None

    soup = bs.BeautifulSoup(response, 'lxml')

    return soup


def collect_data_from_page():
    """
    Collects the tab from tabs ultimate format
    :param url: url to fetch
    :return: None
    """

    card_num = 0
    url = start_url + extended_url
    for i in range(num_pages):
        soup = safe_soup_page(url)
        cards = soup.find_all('ul', attrs={'class': 'cards-grid clear'})

        for line in str(cards).split('\n'):
            if 'img' in line:
                img_link = line.split('src="')[-1].strip('"/>') # get image link
                img_name = str(card_num) + "_" + line.split('alt="')[-1].split('"')[0] + ".png"     #build image string
                print("Collecting: " + img_link + " as " + img_name)
                recorded = False
                attempts_left = 3
                while not recorded:
                    attempts_left -= 1
                    try:
                        with open(directory + img_name, 'wb') as f:
                            f.write(request.urlopen(img_link).read())
                            recorded = True
                    except:
                        if attempts_left == 0:
                            recorded = True
                            print("FAILED!")
                        else:
                            print('Failed Trying again...')
                card_num += 1

        url = start_url + str(i+2) + extended_url


if __name__ == '__main__':
    collect_data_from_page()