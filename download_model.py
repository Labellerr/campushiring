# download_model.py
import requests

url = "https://drive.google.com/file/d/1g3FVJ2oEsipyAjxswi8t3it1iCvM_5Jq/view?usp=sharing"
r = requests.get(url)
with open("best.pt", "wb") as f:
    f.write(r.content)
