
import csv
import requests
from bs4 import BeautifulSoup
import time
import logging
from urllib.parse import urljoin
import tempfile
import os


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager

#  Configuration
start_url = 'https://www.punjabitribuneonline.com/news/nation/'

output_csv_file = 'punjabi_tribune_nation_selenium_updated.csv'
source_website = 'Punjabi Tribune'

max_articles_per_run = 5000
article_scrape_delay = 3
load_more_delay = 5
max_load_more_attempts =1000

#  Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#  Header for CSV

csv_header = ['Headline', 'Article Text', 'URL', 'Source', 'Publication Date']

# CSS Selectors
load_more_button_selector = ".load-button"
article_container_selector = "div.listing"
article_link_selector = 'h2.is-title.post-title a'
headline_selector = 'h1.is-title.post-title'
article_text_container_selector = 'div.entry-content'
date_selector = 'time.post-date'


requests_headers = {
    'User-Agent': 'MyPoliticalNewsScraper/1.0+Selenium (Contact: your-email@example.com)'
}


def scrape_article(article_url):
    """Fetches and scrapes headline, text, and date from a single article URL using Requests."""
    try:
        logging.info(f"[Requests] Requesting article: {article_url}")
        response = requests.get(article_url, headers=requests_headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        headline_element = soup.select_one(headline_selector)
        headline = headline_element.get_text(strip=True) if headline_element else "Headline not found"

        text_container = soup.select_one(article_text_container_selector)
        article_text = "Article text not found"
        if text_container:
            paragraphs = text_container.find_all('p', recursive=False)
            if not paragraphs or not any(p.get_text(strip=True) for p in paragraphs):
                 paragraphs = text_container.find_all('p')
            if paragraphs:
                 article_text = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            else:
                 article_text = text_container.get_text(strip=True, separator='\n')

        date_element = soup.select_one(date_selector)
        publication_date = date_element.get_text(strip=True) if date_element else "Date not found"

        # Return None if essential data missing
        if headline == "Headline not found" and article_text == "Article text not found":
             logging.warning(f"[Requests] Could not extract headline or body from {article_url}")
             return None

        logging.info(f"[Requests] Scraped: Headline='{headline[:30]}...', Date='{publication_date}'")
        # Return only the columns defined in the header
        return {
            'Headline': headline,
            'Article Text': article_text,
            'URL': article_url,
            'Source': source_website,
            'Publication Date': publication_date
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"[Requests] Network error scraping {article_url}: {e}")
    except Exception as e:
        logging.error(f"[Requests] Error parsing {article_url}: {e}")
    return None




existing_urls = set()
file_existed_before_run = os.path.exists(output_csv_file)
if file_existed_before_run:
    print(f"Reading existing URLs from {output_csv_file}...")
    try:
        with open(output_csv_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames and 'URL' in reader.fieldnames:
                 for row in reader:
                     if row.get('URL'):
                         existing_urls.add(row['URL'])

            elif os.path.getsize(output_csv_file) > 0:
                 print(f"Warning: CSV file {output_csv_file} exists but has no header or 'URL' column.")

        print(f"Found {len(existing_urls)} existing URLs.")
    except Exception as e:
        print(f"Error reading existing CSV: {e}. Starting fresh check.")
        existing_urls = set()
else:
     print(f"Output file {output_csv_file} not found. This will be the first run.")




print("--- Starting Selenium Scraper (Handles Existing URLs) ---")
print("Libraries: selenium, webdriver-manager, beautifulsoup4, requests")
print("Requires Google Chrome browser.")

driver = None
all_article_urls = set()
newly_scraped_data = []
articles_scraped_this_run = 0

try:

    print("Setting up Chrome WebDriver...")
    service = Service(ChromeDriverManager().install())

    user_data_dir = tempfile.mkdtemp(prefix="selenium_chrome_user_data_", dir="/tmp")
    print(f"--- Using unique temporary user data directory: {user_data_dir} ---")
    options = webdriver.ChromeOptions()

    chrome_binary_path = "/usr/bin/google-chrome"
    if chrome_binary_path and os.path.exists(chrome_binary_path):
        options.binary_location = chrome_binary_path
        print(f"--- Using specified Chrome binary: {chrome_binary_path} ---")
    elif chrome_binary_path:
         print(f"--- WARNING: Chrome binary path specified but not found: {chrome_binary_path} ---")
    else:
        print("--- Chrome binary path not specified. Relying on WebDriver to find Chrome. ---")


    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f'user-agent={requests_headers["User-Agent"]}')
    options.add_argument(f"--user-data-dir={user_data_dir}")
    options.add_argument("--enable-logging --v=1")

    print("Initializing WebDriver...")
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(10)
    print("WebDriver setup complete.")


    print(f"Loading initial page: {start_url}")
    driver.get(start_url)
    print("Waiting after initial page load...")
    time.sleep(5)


    print(f"Attempting to click 'Load More' button using selector: '{load_more_button_selector}'")
    for attempt in range(max_load_more_attempts):
        print(f"--- Load More Attempt {attempt + 1} / {max_load_more_attempts} ---")
        try:
            wait = WebDriverWait(driver, 20)
            load_more_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, load_more_button_selector)))
            print("Found 'Load More' button. Scrolling slightly and clicking...")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", load_more_button)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", load_more_button)
            print(f"Clicked 'Load More'. Waiting {load_more_delay} seconds for content...")
            time.sleep(load_more_delay)
        except TimeoutException:
            print("Could not find 'Load More' button after waiting (Timeout). Assuming all content loaded.")
            break
        except Exception as e:
            print(f"An unexpected error occurred during 'Load More' click: {e}")
            logging.error(f"Unexpected error clicking Load More: {e}", exc_info=True)
            break


    print("\nFinished clicking 'Load More'. Extracting final page source...")
    final_html = driver.page_source
    print("Parsing final HTML with BeautifulSoup...")
    soup = BeautifulSoup(final_html, 'html.parser')


    print("Finding all article link elements...")
    link_elements = []
    if article_container_selector:
        container = soup.select_one(article_container_selector)
        if container:
             print(f"Searching for links within container: '{article_container_selector}' using selector '{article_link_selector}'")
             link_elements = container.select(article_link_selector)
        else:
             print(f"Warning: Article container '{article_container_selector}' not found.")
             print(f"Searching for links ('{article_link_selector}') in the whole page.")
             link_elements = soup.select(article_link_selector)
    else:
         print(f"Searching for links ('{article_link_selector}') in the whole page.")
         link_elements = soup.select(article_link_selector)

    print(f"Found {len(link_elements)} potential article link elements in final HTML.")

    if not link_elements:
        print("--- WARNING: No link elements found. Check selectors or page load state. ---")

    for link_element in link_elements:
        if 'href' in link_element.attrs:
            href = link_element['href']
            article_url = urljoin(start_url, href)
            all_article_urls.add(article_url) # Collect all found URLs

    print(f"Collected {len(all_article_urls)} unique article URLs from Selenium phase.")

except Exception as e:
    print(f"--- An error occurred during Selenium setup or page processing: {e} ---")
    logging.error("Error during Selenium phase", exc_info=True)

finally:
    # --- Close the browser ---
    if driver:
        print("Closing WebDriver...")
        driver.quit()
        # Clean up the temporary user data directory
        try:
            if 'user_data_dir' in locals() and os.path.exists(user_data_dir):
                 import shutil
                 shutil.rmtree(user_data_dir)
                 print(f"Cleaned up temporary directory: {user_data_dir}")
        except Exception as cleanup_error:
             print(f"Warning: Could not clean up temp directory {user_data_dir}: {cleanup_error}")
        print("WebDriver closed.")


#  STEP 2: Filter and Scrape Only NEW Articles
urls_to_scrape = list(all_article_urls)
if urls_to_scrape:
    print(f"\n--- Checking {len(urls_to_scrape)} collected URLs against {len(existing_urls)} existing URLs... ---")

    for i, url in enumerate(urls_to_scrape):
        # THE CORE LOGIC TO AVOID DUPLICATES
        if url in existing_urls:

            continue


        if articles_scraped_this_run >= max_articles_per_run:
            print(f"\nReached maximum NEW article scraping limit ({max_articles_per_run}) for this run. Stopping.")
            break

        print(f"\n--- Processing NEW Article {articles_scraped_this_run + 1} / Target {max_articles_per_run} (URL {i+1} / {len(urls_to_scrape)}) ---")

        article_data = scrape_article(url)

        if article_data:

            filtered_data = {header: article_data.get(header) for header in csv_header if header != 'Sentiment (AI)'} # Exclude sentiment column here
            newly_scraped_data.append(filtered_data)
            articles_scraped_this_run += 1
        else:
             print(f"--- Failed to scrape or process NEW article {url} ---")


        print(f"--- Waiting for {article_scrape_delay} seconds... ---")
        time.sleep(article_scrape_delay)
else:
     print("\n--- No article URLs were collected by Selenium, cannot scrape individual articles. ---")




if newly_scraped_data:
    print(f"\n--- Appending {len(newly_scraped_data)} NEW articles to {output_csv_file} ---")
    try:

        with open(output_csv_file, 'a', newline='', encoding='utf-8') as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=[h for h in csv_header if h != 'Sentiment (AI)'], extrasaction='ignore')

            if not file_existed_before_run or os.path.getsize(output_csv_file) == 0:
                 writer.writeheader()
                 print("Writing header row.")

            writer.writerows(newly_scraped_data)
        print(f"--- SUCCESS: Appended data to {output_csv_file} ---")
    except IOError as e:
        print(f"--- ERROR: Error appending to CSV file {output_csv_file}: {e} ---")
        logging.error(f"Error writing CSV: {e}", exc_info=True)
    except Exception as e:
         print(f"--- ERROR: Unexpected error writing to CSV file {output_csv_file}: {e} ---")
         logging.error(f"Unexpected error writing CSV: {e}", exc_info=True)
else:
     print("\n--- No NEW articles were successfully scraped in this run to append to CSV. ---")


print("\n--- Script finished. ---")
