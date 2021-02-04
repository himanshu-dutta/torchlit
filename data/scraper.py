import os
import math
import logging
from tqdm import *
import urllib.request
from typing import Union, List
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


def init_driver(fetcher):
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    tab = webdriver.Chrome(
        ChromeDriverManager(log_level=logging.WARNING).install(), options=options
    )

    def fetcher_with_tab(*args, **kwargs):
        return fetcher(*args, **kwargs, driver=tab)

    return fetcher_with_tab


@init_driver
def google_images_scraper(
    search_term: str,
    min_image_count: int = 10,
    driver: webdriver.chrome.webdriver.WebDriver = None,
) -> list:
    base_url = "https://www.google.com/imghp?hl=en"
    driver.get(base_url)
    search_form = driver.find_element_by_name("q")
    search_form.send_keys(search_term)
    search_form.submit()
    urls = []
    last_height = driver.execute_script("return document.body.scrollHeight")

    reached_page_end = False
    while len(urls) < min_image_count:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        urls = driver.execute_script(
            "return Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));"
        )

        while None in urls:
            urls.remove(None)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if last_height == new_height:
            break
        else:
            last_height = new_height

    return urls


def download_images_from_url(
    urls: Union[str, List[str]],
    image_tag: str,
    format: str = "png",
    save_location: str = ".",
) -> None:
    dir = os.path.join(save_location, image_tag)
    if not os.path.exists(dir):
        os.mkdir(dir)

    if isinstance(urls, str):
        urllib.request.urlretrieve(urls, os.path.join(dir, "0001" + "." + format))
        print(f"Downloaded 1 image in {dir}.")

    elif isinstance(urls, list):
        pad = math.ceil(math.log10(len(urls)))
        for idx in tqdm(range(len(urls))):
            urllib.request.urlretrieve(
                urls[idx], os.path.join(dir, str(idx + 1).zfill(pad) + "." + format)
            )
        print(f"Downloaded {len(urls)} images in {dir}.")

    else:
        raise Exception('Invalid argument type for "urls"')
