import urllib.request               #urllib :library that extract data from web

url = "http://uta.pw/shodou/img/28/214"         #input image's url and fix png name
savename = "test.png"
def hellourl():
    urllib.request.urlretrieve(url,savename)        
    print("Saved!")                     #saved in D:\study (executed point)

    mem = urllib.request.urlopen.read()         #read() = Generally, after open(), then read()// urlopen set on stack data then read it

    with open(savename, mode = "wb") as f:      #with open : load saved file in disk ,and select mode (read,write,both)
        f.write(mem)
        print("Saved!!")

#import urllib.request          first : import module that read url link
def use_api ():
    url =  	"http://openapi.customs.go.kr/openapi/service/newTradestatistics/getitemtradeList"             #second : read url
    urlopen = urllib.request.urlopen(url)
    data = urlopen.read()
    #text = data.decode("utf-8")         # data.decode("utf-8") : decode binary data to text data (maybe to read)
import urllib.parse
def use_api_kma():
    api = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"     #load api

    value = {'stnld':'159'}         #set parameter : parameter to set area 
    params = urllib.parse.urlencode(value)      #parse.urlencode : encode to use parameters type of tuple
    url = api+"?"+params                        
    #load new url with stnld parameter (api + ? + parameter(encoded to use api))
    #ㄴ input "?" tail of url and input ((key) = (value)) 
    data = urllib.request.urlopen(url).read()
    text = data.decode("utf-8")
    #print(text)             #printed too many data
    #filtering neccessary data
