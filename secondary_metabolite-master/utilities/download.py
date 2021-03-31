""" scraper.py -- download Gene clusters from antiSMASH or MiBIG """

import os
import requests

from dataclasses import dataclass
from bs4 import BeautifulSoup
from requests.models import HTTPError


@dataclass
class Scraper(object):
    """
    """

    identifier: str = None


    def mibig_download(self):
        """
        """

        url: str = 'https://mibig.secondarymetabolites.org/repository/{}/{}.1.region001.gbk'.format(self.identifier, self.identifier)
        page = requests.get(url)

        soup = BeautifulSoup(page.text, 'html.parser')
        self.write_to_file(soup, url=None, href=None)
    

    def antismash_download(self, url: str):
        """
        """

        try:
            href: str
            line: str = 'https://antismash-db.secondarymetabolites.org/output/{}/index.html#r1c2'.format(url.split('/')[7])
            page = requests.get(line) 
            if page.status_code == 200:
                soup = BeautifulSoup(page.text, 'html.parser')
                links = [tag for tag in soup.findAll('a') if tag.string == 'Download region GenBank file']
                href = links[0]['href']
                download_page: str = 'https://antismash-db.secondarymetabolites.org/output/{}/{}'.format(url.split('/')[7], href)
                page = requests.get(download_page)
                if page.status_code == 200:
                    soup = BeautifulSoup(page.text, 'html.parser')
            else:
                return False
        except requests.ConnectionError:
            raise('Connection Error')
            
        self.write_to_file(soup, url=url, href=href)

        return href


    

    def write_to_file(self, data, url, href):
        """
        """

        if url != None or href != None:
            # only use when dealing with antismash
            self.identifier = self.identifier if self.identifier else url.split('/')
            self.identifier = self.identifier[7] + "_" + href

        file_path: str = os.path.join(os.getcwd(), 'tmp\\gbk\\{}.gbk'.format(self.identifier))

        with open(file_path, 'wb') as temp_file:
            temp_file.write(data.encode('utf-8'))
    
