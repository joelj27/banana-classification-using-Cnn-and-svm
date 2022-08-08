from instascrape import Reel
import time

# session id
SESSIONID = "1659556956374"

# Header with session id
headers = {
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
	AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.74 \
	Safari/537.36 Edg/79.0.309.43",
	"cookie": f'sessionid={1659556956374};'
}

# Passing Instagram reel link as argument in Reel Module
insta_reel = Reel('https://www.instagram.com/p/Cgf55oMJ3B7/?utm_source=ig_web_copy_link')

# Using scrape function and passing the headers
insta_reel.scrape(headers=headers)

# Giving path where we want to download reel to the
# download function
insta_reel.download(fp=f"data.mp4")

# printing success Message
print('Downloaded Successfully.')
