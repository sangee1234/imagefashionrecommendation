from bs4 import BeautifulSoup
import requests
import logging
import urllib.request
import os

print("hello")
# logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148','From':'saransang97@gmail.com',
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
}

def download_image(url, category,  img_id):
    urllib.request.urlretrieve(url, "images_new/"+category+"/"+img_id+".jpg")

def retreive_category(category):
    with open("file_"+category+".csv","a") as f :
        
        URL = "your url"
        urls = [URL]
        for i in range(2,11):
            urls.append(URL+"&page="+str(i))

        for url in urls:

            webpage = requests.get(url, headers=HEADERS)
            soup = BeautifulSoup(webpage.content, "html.parser")
            print(soup)
            try:
                outer_divs = soup.find_all("div", attrs={"data-component-type":"s-search-result"})
                print(outer_divs)
                for outer_div in outer_divs:

                    img_div = outer_div.find("img", attrs={"class":"s-image"})
                    img_src = img_div['src']
                    print(img_src)
                    img_id = outer_div["data-asin"]
                    l = []
                    if os.path.exists(img_id):
                        with open("nlist_file_"+category+".txt","r") as fl :
                            l = [product_id for product_id  in fl]
                    if img_id not in l:
                        f.write(img_id+";"+img_src+";"+img_div['alt']+'\n')
                        download_image(img_src, category, img_id)
                        with open("nlist_file_"+category+".txt","a") as fl :
                            fl.write(img_id+'\n')
            except Exception as e:
                print(e)
                logging.error("There was an error in extracting webpage content in url "+url)
     
for category in ["Short sleeve top","Long sleeve top","Short sleeve outwear", "Long sleeve outwear", "Vest",
"Sling", "Shorts", "Trousers", "Skirt", "Short sleeve dress", "Long sleeve dress", "Vest dress","Sling dress"]:
    os.mkdir("images_new/"+category)
    retreive_category(category)




