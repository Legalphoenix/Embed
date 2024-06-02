import requests
from bs4 import BeautifulSoup
import os

# The base URL for accessing each case
base_url = "https://dpcuria.eu/case?reference="

# Corrected list of all 100 unique case references
case_references = [
    "C-741/21",
    "C-604/22",
    "C-118/22",
    "C-687/21",
    "C-634/21",
    "C-26/22",
    "C-807/21",
    "C-683/21",
    "C-333/22",
    "C-319/22",
    "C-307/22",
    "C-252/21",
    "C-579/21",
    "C-300/21",
    "C-487/21",
    "C-34/21",
    "C-268/21",
    "C-349/21",
    "C-453/21",
    "C-560/21",
    "C-205/21",
    "C-132/21",
    "C-154/21",
    "C-694/20",
    "C-460/20",
    "C-180/21",
    "C-37/20",
    "C-129/21",
    "C-77/21",
    "C-306/21",
    "C-793/19",
    "C-339/20",
    "C-184/20",
    "C-534/20",
    "C-817/19",
    "C-319/20",
    "C-140/20",
    "C-245/20",
    "C-175/20",
    "C-102/20",
    "C-439/19",
    "C-597/19",
    "C-645/19",
    "C-746/18",
    "C-61/19",
    "C-511/18",
    "C-623/17",
    "C-311/18",
    "C-272/19",
    "C-708/18",
    "C-673/17",
    "C-507/17",
    "C-136/17",
    "C-40/17",
    "C-345/17",
    "C-207/16",
    "C-25/17",
    "C-210/16",
    "C-434/16",
    "C-73/16",
    "C-13/16",
    "C-536/15",
    "C-398/15",
    "C-203/15",
    "C-582/14",
    "C-191/15",
    "C-362/14",
    "C-230/14",
    "C-201/14",
    "C-446/12",
    "C-212/13",
    "C-141/12",
    "C-131/12",
    "C-293/12",
    "C-486/12",
    "C-473/12",
    "C-291/12",
    "C-342/12",
    "C-119/12",
    "C-461/10",
    "C-70/10",
    "C-468/10",
    "C-543/09",
    "C-92/09",
    "C-553/07",
    "C-557/07",
    "C-524/06",
    "C-73/07",
    "C-275/06",
    "C-101/01",
    "C-465/00",
    "C-350/21",
    "C-376/22",
    "C-667/21"
]

# Directory to save the case files
output_dir = "dpcuria_cases"
os.makedirs(output_dir, exist_ok=True)

# Function to download and save case text
def download_case(reference):
    url = base_url + reference
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Locate the h1 tag with class 'heading2' and extract all following text until <br>
        heading = soup.find('h1', class_='heading2')
        if heading:
            case_text_elements = heading.find_all_next(string=True)
            case_text = "\n".join(case_text_elements).split("[Signatures]")[0]  # Extract text until "[Signatures]"
            filename = f"{reference.replace('/', '_')}.txt"  # Replace / with _ for filenames
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(case_text.strip())
            print(f"Saved case {reference}")
        else:
            print(f"Case heading not found for {reference}")
    else:
        print(f"Failed to retrieve case {reference}")

# Download each case
for case in case_references:
    download_case(case)
