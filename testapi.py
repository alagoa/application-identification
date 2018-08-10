import requests
r = requests.put('http://localhost:5200/api/predictions/1', json={'prediction':'twasdfasdfasdfitch'})
print(r)