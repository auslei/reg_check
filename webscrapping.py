#%%
import time #for delayed action
import numpy as np # for randomisation
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium import webdriver

_wd_path = "/Users/auslei/Programming/python/chromedriver"
_output_dir = "/Users/auslei/Programming/python/webscrapping/output"


#%% Functions
# get webdriver: chrome driver
def get_webdriver() -> webdriver:
    opt = Options()
    opt.add_argument('--disable-blink-features=AutomationControlled') 
    return webdriver.Chrome(executable_path=_wd_path,options=opt)

# highlight a specific elements
def highlight(element, effect_time = 3, color = "red", border = 5):
    """Highlights (blinks) a Selenium Webdriver element"""
    driver = element._parent
    def apply_style(s):
        driver.execute_script("arguments[0].setAttribute('style', arguments[1]);",
                              element, s)
    original_style = element.get_attribute('style')
    apply_style("border: {0}px solid {1};".format(border, color))
    time.sleep(effect_time)
    apply_style(original_style)


driver = get_webdriver()

page = driver.get("https://ncc.abcb.gov.au/editions/ncc-2022/adopted/housing-provisions")


# %%
