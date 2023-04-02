import requests

url = 'http://localhost:5000/predict'
data = {'text': 'god bless you'}
response = requests.post(url, json=data)

print(response.json())