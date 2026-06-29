import gdown
import zipfile

url = "https://drive.google.com/uc?id=1b5FIh31obzF6dbex4s9C68s2tcBAhsaq"
output = "poses.zip"

gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall(".")