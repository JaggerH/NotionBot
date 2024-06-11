import os
from utils.set_env import set_env
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
set_env()

def init_driver():
    """
    初始化WebDriver
    """
    driver_path = os.getenv("CHROME_DRIVER")
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service)
    return driver

def open_page(driver, url):
    """
    打开目标网页
    """
    driver.get(url)
    driver.implicitly_wait(10)  # 等待加载时间，视情况调整

def get_table_html(driver, selector):
    """
    提取HTML表格内容
    """
    table_element = driver.find_element(By.CSS_SELECTOR, selector)
    return table_element.get_attribute('outerHTML')

def close_driver(driver):
    """
    关闭WebDriver
    """
    driver.quit()
