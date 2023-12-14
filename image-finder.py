import requests
import os
from PIL import Image
from io import BytesIO
import hashlib

# Your Bing Search API subscription key
subscription_key = "78865a655c24410281f00b8ef08951ca"

# Endpoint URL for the Bing Image Search API
search_url = "https://api.bing.microsoft.com/v7.0/images/search"

# Search terms
search_terms = ["highland heath", "hilltop castle", "himalayan peaks",
    "hot springs", "iceberg lagoon", "igneous rock formations", "island archipelago",
    "jungle river", "kaleidoscope sky", "karst landscape", "kelp forest underwater",
    "lacustrine plain", "lagoon islands", "lake aurora", "lavender fields",
    "lichen rocks", "lofty peaks", "lush delta", "mangrove forest",
    "marble caves", "marshland", "meandering river", "mediterranean coast",
    "mesa plateau", "meteor shower sky", "misty fjords", "monsoon forest",
    "moonlight bay", "mountain pass", "mountain ridge", "mountain tarn",
    "muddy wetlands", "mystical forest", "northern lights sky", "oasis desert",
    "ocean cliff", "ocean floor", "old growth forest", "olive grove",
    "opal mine", "pacific atoll", "palm grove", "peat bog",
    "peninsula coast", "permafrost tundra", "pine forest snow", "pink sand beach",
    "plains bison", "plateau mountains", "plumeria grove", "pond lilies",
    "prairie sunset", "prehistoric forest", "primeval forest", "pristine beach",
    "quartzite cliffs", "rainforest canopy", "rainforest waterfall", "ravine gorge",
    "reef barrier", "rock arches", "rocky shore", "rolling hills",
    "sagebrush steppe", "salt flats", "sandstone arch", "sandy cay",
    "savanna sunset", "scenic byway", "scree slope", "sea arch",
    "sea stacks", "seagrass meadow", "secluded beach", "serene lake",
    "shale hills", "sierra mountains", "silhouette sunset", "silver falls",
    "smoky mountains", "snowy forest", "solar eclipse", "spring meadow",
    "stone forest", "stony brook", "subalpine meadow", "sulfur springs",
    "sunflower field", "sunset crater", "swamp bayou", "taiga forest",
    "teal ocean", "temperate rainforest", "thermal geyser", "tidal marsh",
    "tidepool life", "towering redwoods", "tranquil cove", "tree ferns",
    "tropical foliage", "tropical moon", "tundra autumn", "turquoise sea",
    "twilight sky", "underground cave", "underwater coral", "valley meadow",
    "volcanic lake", "waterfall grotto", "wetland marsh", "white birch forest",
    "wildflower valley", "willow trees", "windblown desert", "winter alpine",
    "wisteria tunnel", "woodland creek", "woodland path", "zen garden"
    ]
headers = {"Ocp-Apim-Subscription-Key": subscription_key}

# Number of images to download for each search term
num_images = 1000

# Directory to save the downloaded images
image_dir = 'downloaded_images'
os.makedirs(image_dir, exist_ok=True)

# Set to store image hashes and avoid duplicates
downloaded_image_hashes = set()

for term in search_terms:
    # Set parameters for search
    params = {
        "q": term,
        "license": "public",
        "imageType": "photo",
        "count": num_images,
    }

    # Make the search request to the Bing Image API, and get the results
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    # Download and save each of the images returned by the API
    for idx, img in enumerate(search_results["value"]):
        try:
            # Make a request to download the image file
            img_data = requests.get(img["contentUrl"])
            img_data.raise_for_status()
            
            # Calculate the image's hash and skip download if it's a duplicate
            image_hash = hashlib.sha256(img_data.content).hexdigest()
            if image_hash in downloaded_image_hashes:
                print(f"Duplicate image skipped: {img['contentUrl']}")
                continue
            
            # Add the hash to the set of known hashes
            downloaded_image_hashes.add(image_hash)

            # Open the image file and save it to the directory
            image = Image.open(BytesIO(img_data.content))
            image_path = os.path.join(image_dir, f"{term.replace(' ', '_')}_{idx}.jpg")
            with open(image_path, "wb") as f:
                image.save(f, "JPEG")
        except Exception as e:
            # If there's any issue with one image, we can simply skip it
            print(f"Could not download {img['contentUrl']} - {e}")

print("Image download complete.")
y