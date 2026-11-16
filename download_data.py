
import requests

# Dataset download link
url = "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main/freMTPL2freq.csv"

#Download the dataset
response=requests.get(url)
#Check if download was successful
response.raise_for_status()
#Save the dataset
with open("data/insurance_claims.csv",'wb') as file:
    file.write(response.content)

print("Dataset downloaded successfully")

