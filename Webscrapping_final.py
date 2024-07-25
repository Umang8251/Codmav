import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.common.exceptions import StaleElementReferenceException

titles = []
category = []
allergen = []
serving = []
weight = []
energy_list = []
proteins_list = []
carbs_list = []
fats_list = []
fiber_list = []
all_ingredients = []

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

website = 'https://www.niwi.ai/'
service = Service()
driver = webdriver.Chrome(service=service)
driver.maximize_window()
driver.get(website)
wait = WebDriverWait(driver, 10)

links = driver.find_elements(By.XPATH, "//a[@href]")
for link in links:
    if "recipes" in link.get_attribute("innerHTML"):
        link.click()
        break

# Working code
def func(page_number):
    go = True
    while go:
        try:
            # Process the current page (scrape the data)
            print(f"Scraping page {page_number + 1}")

            # Click the 'next page' button to go to the next page
            for click in range(page_number):
                # Wait for the 'next page' button to be visible and clickable
                button = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@aria-label='Go to next page']")))
                button.click()

            # Optional: Add a small delay to handle loading times
            time.sleep(2)

            # Handle cookies popup if it exists
            try:
                cookies_popup = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[@class='cookies-popup']"))
                )
                # Click on the 'Accept' button or close the popup
                cookies_popup.find_element(By.XPATH, "//button[contains(text(), 'Accept')]").click()
            except Exception as e:
                print(f"No cookies popup found or failed to handle: {e}")

            # Wait for the div elements to be present and visible
            div_elements = WebDriverWait(driver, 5).until(
                EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@class, 'w-full lg:w-[90%] grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3 gap-4 items-center justify-items-center')]"))
            )

            # List to store the href attributes
            links = []

            # Iterate over each div element
            for div in div_elements:
                # Find all <a> tags within the current div element
                a_tags = div.find_elements(By.TAG_NAME, "a")

                # Print the href attribute of each <a> tag
                for a in a_tags:
                    href = a.get_attribute("href")
                    print(href)
                    links.append(href)  # Append each href to the links list

            # Iterate over the links and extract data
            for link in links:
                driver.get(link)
                time.sleep(2)  # Wait for the page to load

                # Extract the food title
                try:
                    food = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, "//h1"))
                    )
                    title = food.get_attribute("innerHTML").strip()
                except Exception as e:
                    print(f"Failed to extract title: {e}")
                    continue

                # Extract the food category
                try:
                    cat = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, "//p[@class='p-2 text-xs lowercase bg-emerald-600 rounded-md text-white']"))
                    )
                    cate = cat.get_attribute("innerHTML").strip()
                except Exception as e:
                    print(f"Failed to extract category: {e}")
                    continue

                # Extract the food allergen
                try:
                    alle = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, "//p[@class='p-2 text-xs lowercase bg-slate-600 rounded-md text-white']"))
                    )
                    allerg = alle.get_attribute("innerHTML").strip()
                except Exception as e:
                    print(f"Failed to extract allergen: {e}")
                    continue

                try:
                    details = driver.find_elements(By.XPATH, "//strong[@class='text-rose-600']")
                    if len(details) >= 2:  # Ensure there are enough nutrients listed
                        serving_size = details[1].get_attribute("innerHTML").strip()
                        weight_size = details[2].get_attribute("innerHTML").strip()
                    else:
                        print("Not enough details found")
                        continue

                    # Extract nutrients
                    nutrients = driver.find_elements(By.XPATH, "//p[@class='text-sm lg:text-lg font-bold']")
                    if len(nutrients) >= 5:  # Ensure there are enough nutrients listed
                        energy = nutrients[0].get_attribute("innerHTML").strip()
                        proteins = nutrients[1].get_attribute("innerHTML").strip()
                        carbs = nutrients[2].get_attribute("innerHTML").strip()
                        fats = nutrients[3].get_attribute("innerHTML").strip()
                        fiber = nutrients[4].get_attribute("innerHTML").strip()
                    else:
                        print("Not enough nutrients found")
                        continue

                    # Extract ingredients
                    ingredients = driver.find_elements(By.XPATH, "//p[@class='flex-1']")
                    ingredient_list = [' '.join(i.get_attribute("innerHTML").replace("\n", "").split()) for i in ingredients]

                    # Append extracted data to lists
                    titles.append(title)
                    category.append(cate)
                    allergen.append(allerg)
                    serving.append(serving_size)
                    weight.append(weight_size)
                    energy_list.append(energy)
                    proteins_list.append(proteins)
                    carbs_list.append(carbs)
                    fats_list.append(fats)
                    fiber_list.append(fiber)
                    all_ingredients.append(ingredient_list)

                except Exception as e:
                    print(f"Failed to extract some details: {e}")
                    continue

        except StaleElementReferenceException:
            print("StaleElementReferenceException occurred. Re-finding the element...")
            continue  # Retry finding the button if StaleElementReferenceException occurs

        except Exception as e:
            print(f"An error occurred: {e}")
            break  # Break the loop if any other exception occurs

        driver.get(website)  # Replace with the actual URL
        links = driver.find_elements(By.XPATH, "//a[@href]")

        for link in links:
            if "recipes" in link.get_attribute("innerHTML"):
                link.click()
                break
        go = False

# Iterate through each page
for page_num in range(82, 100 , 1):
    func(page_num)

# Check the lengths of all lists
print(f"Titles: {len(titles)}, Categories: {len(category)}, Allergens: {len(allergen)}, Serving sizes: {len(serving)}, Weights: {len(weight)}, Energy: {len(energy_list)}, Proteins: {len(proteins_list)}, Carbs: {len(carbs_list)}, Fats: {len(fats_list)}, Fiber: {len(fiber_list)}, Ingredients: {len(all_ingredients)}")

# Generate csv file
flattened_ingredients = ['; '.join(ingredients) for ingredients in all_ingredients]
df = pd.DataFrame({
    'food': titles,
    'category': category,
    'allergen': allergen,
    'serving': serving,
    'weight': weight,
    'energy': energy_list,
    'proteins': proteins_list,
    'carbs': carbs_list,
    'fats': fats_list,
    'fiber': fiber_list,
    'Ingredients': flattened_ingredients
})
df.to_csv("foooood.csv", index=False)
print(df)

# To close browser
input("Press Enter to close the browser...")
