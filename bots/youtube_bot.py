from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
import time
import random
import math
import sys

def get_random_wait():
	return (1 - math.sqrt(1 - random.random()))*10 + 1

def get_random_int(list_length):
	return math.floor(abs(random.random() - random.random()) * (1 + list_length))

def click_progress_bar(driver):
	progress_bar = driver.find_element_by_class_name('ytp-progress-list')	
	width = progress_bar.size['width']
	action = ActionChains(driver)
	rand = get_random_int(64)
	action.move_to_element(progress_bar).move_by_offset(- width / (rand + 4), 0)
	action.click()
	action.perform()


def fullscreen(driver, player):
	#Double click
	actions = ActionChains(driver)
	actions.move_to_element(player)
	actions.double_click(player)
	actions.perform()

def watch_videos(driver):
	try:
		#search = driver.find_element_by_name("search_query")
		#search.clear()
		#search.send_keys("ur mom gay lol")
		#search.send_keys(Keys.RETURN)
		driver.get("https://www.youtube.com")
		time.sleep(get_random_wait())
		element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "img"))) # 10 seconds implicit wait
		element.click()
		#player = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "player"))) # 10 seconds implicit wait
		#fullscreen(driver, player)
		while(True):
			# while video isn't over
			while driver.execute_script("return document.getElementById('movie_player').getPlayerState()"):
				rand = get_random_wait()

				print("rand: " + str(rand))
				if rand > 8.5: # sometimes choose another video before current is over
					if random.choice([0, 1, 2, 3]):
						print("skipping video")
						break
					print("going to another point in video")
					click_progress_bar(driver)
				time.sleep(10) # wait for 10s
			print("video ended! rand: " + str(rand))

			# get list of videos
			video_list = driver.find_elements_by_xpath("//ytd-compact-video-renderer[@class='style-scope ytd-watch-next-secondary-results-renderer']")
			next_video = video_list[get_random_int(len(video_list)/2)]
			next_video.click()
			rand = get_random_wait()
	except:
		return watch_videos(driver)

if len(sys.argv) > 1:
	if sys.argv[1] == 'f':	
		prof = webdriver.FirefoxProfile()
		prof.add_extension(extension="extensions/addblock_firefox.xpi")
		prof.set_preference("extensions.ublock_origin.currentVersion", "1.16.0")
		driver = webdriver.Firefox(firefox_profile = prof)

else:
	opt = webdriver.ChromeOptions()
	opt.add_extension("extensions/addblock_chrome.crx")
	driver = webdriver.Chrome(chrome_options = opt)


watch_videos(driver)