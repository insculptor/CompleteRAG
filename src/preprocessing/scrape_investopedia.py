"""
####################################################################################
#####                    File name: preprocess_data.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 03/23/2024                              #####
#####         Scrape the Contents of Investopedia Website and save as PDF      #####
####################################################################################
"""

## Load Environment Variables
import os
from pathlib import Path
import re
import requests
from time import perf_counter as timer
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from preprocess_data import preprocess_text,preprocess_text_math

from dotenv import load_dotenv
load_dotenv(Path('C:/Users/erdrr/OneDrive/Desktop/Scholastic/NLP/LLM/RAG/CompleteRAG/.env'))


headers = {
                    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
                }


class InvestopediaScrape:
    """
    Class to Scrape Investopedia Data and store into pdf.
    """
    def __init__(self, scrape_data_path):
        self.scrape_data_path = scrape_data_path
        print("Starting...", flush=True)
    
    def get_all_pagination(self):
        url = 'https://www.investopedia.com/'
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text,'lxml')
        all_lists = soup.find('ul', {'class': 'terms-bar__list'}).find_all('li')
        return all_lists
        
    def scrape(self, url):
        headers = {
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
        }
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text,'lxml')
        try:
            term_urls = soup.find('div', {'class': 'dictionary-top300-list__list-content'}).find_all('a')
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return
        for term_url in term_urls:
            url = term_url['href'].split('=')[0]
            file_path = os.path.join(self.scrape_data_path)
            os.makedirs(file_path, exist_ok=True)
            file_name = "Investopedia_" + url.split('/')[-2] + "_what_is_" + url.split('/')[-1].replace(".asp", "").replace("-","_")
            file_name = ''.join(x for x in file_name.title() if not x.isspace())
            pdf_path = os.path.join(file_path, f'{file_name}.pdf')

            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            Story = []

            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.content,'lxml')

            # Collecting text data and removing HTML tags by converting to text
            text_elements = soup.findAll('div', {'class': 'article-content'})
            cleaned_text = ' '.join(element.get_text(" ", strip=True) for element in text_elements)
            cleaned_text = preprocess_text(cleaned_text)
            cleaned_text = preprocess_text_math(cleaned_text)
            cleaned_text = cleaned_text.replace("\n"," ")
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

            if not cleaned_text.strip():
                print(f"Skipping empty PDF for URL: {term_url}")
                continue  # Skip this URL as it leads to an empty PDF

            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = [Paragraph(cleaned_text, styles["Normal"])]

            # Build the PDF only if there's content
            doc.build(story)
        print("[INFO]: Scraping finished.")
        
        
if __name__ == "__main__":
    preprocessed_data_path = Path(os.environ["PREPROCESSED_DATA_DIR"])
    os.makedirs(preprocessed_data_path, exist_ok=True)
    start_time = timer()
    crawler = InvestopediaScrape(preprocessed_data_path)
    all_pagination_links =crawler.get_all_pagination()
    for page in all_pagination_links:
        page_url = page.find('a')['href'].split('=')[0]
        print(f"[INFO]: Fetching terms for the URL: {page_url}", flush=True)
        crawler.scrape(page_url)
    end_time = timer()
    print(f"[INFO]: Total Time: {end_time-start_time:.5f} seconds.")
