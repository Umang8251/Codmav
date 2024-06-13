#Dataset creation - Web Scraping
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import pandas as pd
import requests

# Making a GET request
r = requests.get('https://app.samsungfood.com/recipes/101ba678e3bf5075a5126fd78f7dc45024c7061cbcd')

# check status code for response received
# success code - 200
print(r)

# print content of request
print(r.content)

soup = BeautifulSoup(r.content, 'html.parser')
print(soup.prettify())

s = soup.find('div')
#content = s.find_all('table')

print(s)
