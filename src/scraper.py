import os
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import json

# Set the base URL
BASE_URL = "https://thapar.edu"
SITEMAP_URL = "https://thapar.edu/sitemap.xml"

def get_sitemap_links(sitemap_url):
    """
    Fetch all URLs from the sitemap and return them as a list.
    """
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        # Parse the XML
        root = ET.fromstring(response.content)
        urls = [elem.text for elem in root.iter() if elem.tag.endswith("loc")]
        print(f"Total URLs found in sitemap: {len(urls)}")
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

def scrape_content(url):
    """
    Scrape the main content of the webpage from the provided URL.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract text content from all <p> tags
        content = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
        return content if content else None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def main():
    """
    Main function to fetch URLs, scrape their content, and save it to a JSON file.
    """
    print("Fetching sitemap URLs...")
    urls = get_sitemap_links(SITEMAP_URL)
    print(f"Scraping content from {len(urls)} URLs...")

    scraped_data = []

    for idx, url in enumerate(urls):
        print(f"Scraping URL {idx+1}/{len(urls)}: {url}")
        content = scrape_content(url)
        if content:
            print(f"Scraped content from {url}: {content[:100]}")  # Debug: First 100 chars
            scraped_data.append({"url": url, "content": content})
        else:
            print(f"No content found for {url}")

    # Save the scraped data to a JSON file
    output_file = os.path.abspath("./data/raw/thapar_scraped.json")

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(scraped_data, file, ensure_ascii=False, indent=4)
        print(f"Scraped data saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
