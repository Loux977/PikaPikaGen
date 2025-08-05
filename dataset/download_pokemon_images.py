
# -------------------------------
# Automated Pokémon image downloader + deduplicator
# -------------------------------

import os
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from PIL import Image
import imagehash

# --------------- CONFIG ---------------

pokemon_list = ['Abra', 'Aerodactyl', 'Alakazam', 'Arbok', 'Arcanine', 
                'Articuno', 'Beedrill', 'Bellsprout', 'Blastoise', 'Bulbasaur', 
                'Butterfree', 'Caterpie', 'Chansey', 'Charizard', 'Charmander', 
                'Charmeleon', 'Clefable', 'Clefairy', 'Cloyster', 'Cubone', 
                'Dewgong', 'Diglett', 'Ditto', 'Dodrio', 'Doduo',
                'Dragonair', 'Dragonite', 'Dratini', 'Drowzee', 'Dugtrio',
                'Eevee', 'Ekans', 'Electabuzz', 'Electrode', 'Exeggcute',
                'Exeggutor', 'Farfetchd', 'Fearow', 'Flareon', 'Gastly', 
                'Gengar', 'Geodude', 'Gloom', 'Golbat', 'Goldeen',
                'Golduck', 'Golem', 'Graveler', 'Grimer', 'Growlithe',
                'Gyarados', 'Haunter', 'Hitmonchan', 'Hitmonlee', 'Horsea',
                'Hypno', 'Ivysaur', 'Jigglypuff', 'Jolteon', 'Jynx', 'Kabuto', 
                'Kabutops', 'Kadabra', 'Kakuna', 'Kangaskhan', 'Kingler', 'Koffing', 
                'Krabby', 'Lapras', 'Lickitung', 'Machamp', 'Machoke', 'Machop', 
                'Magikarp', 'Magmar', 'Magnemite', 'Magneton', 'Mankey', 'Marowak', 
                'Meowth', 'Metapod', 'Mew', 'Mewtwo', 'Moltres', 'MrMime', 'Muk', 
                'Nidoking', 'Nidoqueen', 'Nidorina', 'Nidorino', 'Ninetales', 'Oddish', 
                'Omanyte', 'Omastar', 'Onix', 'Paras', 'Parasect', 'Persian', 'Pidgeot', 
                'Pidgeotto', 'Pidgey', 'Pikachu', 'Pinsir', 'Poliwag', 'Poliwhirl', 
                'Poliwrath', 'Ponyta', 'Porygon', 'Primeape', 'Psyduck', 'Raichu', 
                'Rapidash', 'Raticate', 'Rattata', 'Rhydon', 'Rhyhorn', 'Sandshrew', 
                'Sandslash', 'Scyther', 'Seadra', 'Seaking', 'Seel', 'Shellder', 
                'Slowbro', 'Slowpoke', 'Snorlax', 'Spearow', 'Squirtle', 'Starmie', 
                'Staryu', 'Tangela', 'Tauros', 'Tentacool', 'Tentacruel', 'Vaporeon', 
                'Venomoth', 'Venonat', 'Venusaur', 'Victreebel', 'Vileplume', 'Voltorb', 
                'Vulpix', 'Wartortle', 'Weedle', 'Weepinbell', 'Weezing', 'Wigglytuff', 
                'Zapdos', 'Zubat', 'Lucario'] # Lucario added

print(len(pokemon_list))

keywords_extra = [
    "official anime art",
    "anime screenshot",
    "anime official"
]

base_folder = "dataset/images_scrapped"
max_per_keyword = 50  # Images per keyword per crawler

# --------------- IMAGE DOWNLOAD + DEDUPLICATION ---------------

for pokemon in pokemon_list:
    pokemon_folder = os.path.join(base_folder, pokemon)
    os.makedirs(pokemon_folder, exist_ok=True)
    print(f"#########    Downloading images for: {pokemon}   ###########")

    # Define all keyword variations
    keyword_variations = [f"{pokemon} Pokémon"] + [f"{pokemon} Pokémon {extra}" for extra in keywords_extra]

    # Google
    google_crawler = GoogleImageCrawler(storage={"root_dir": pokemon_folder})
    for kw in keyword_variations:
        try:
            google_crawler.crawl(keyword=kw, max_num=max_per_keyword, file_idx_offset='auto')
        except Exception as e:
            print(f"Google crawler error for {kw}: {e}")

    # Bing
    bing_crawler = BingImageCrawler(storage={"root_dir": pokemon_folder})
    for kw in keyword_variations:
        try:
            bing_crawler.crawl(keyword=kw, max_num=max_per_keyword, file_idx_offset='auto')
        except Exception as e:
            print(f"Bing crawler error for {kw}: {e}")

    # --------------- DEDUPLICATION ---------------

    print(f" Removing duplicates for {pokemon}...")
    hashes = {}
    duplicates = []

    for fname in os.listdir(pokemon_folder):
        img_path = os.path.join(pokemon_folder, fname)
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            h = imagehash.phash(img)
            if h in hashes:
                duplicates.append(img_path)
            else:
                hashes[h] = img_path
        except Exception as e:
            print(f"Could not process {img_path}: {e}")
            duplicates.append(img_path)

    for dup_path in duplicates:
        try:
            os.remove(dup_path)
        except Exception as e:
            print(f"Error removing {dup_path}: {e}")

    print(f"{pokemon}: {len(duplicates)} duplicates removed.\n")

print("#################  All done!  #######################")
