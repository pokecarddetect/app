"""
Sample Pokemon card data for testing and authentication system initialization.
This creates a basic database of known authentic Pokemon cards for comparison.
"""

import json
import os
from app import app, db
from models import OfficialCard, PokemonSet

def create_sample_card_data():
    """
    Create sample Pokemon card data for testing the authenticity system.
    This simulates the data structure from the Pokemon TCG API.
    """
    
    # Sample Pokemon sets
    sample_sets = [
        {
            "id": "base1",
            "name": "Base Set",
            "series": "Base",
            "printedTotal": 102,
            "total": 102,
            "legalities": {"unlimited": "Legal"},
            "ptcgoCode": "BS",
            "releaseDate": "1999/01/09",
            "updatedAt": "2020/08/14 09:35:00",
            "images": {
                "symbol": "https://images.pokemontcg.io/base1/symbol.png",
                "logo": "https://images.pokemontcg.io/base1/logo.png"
            }
        },
        {
            "id": "base2",
            "name": "Jungle",
            "series": "Base",
            "printedTotal": 64,
            "total": 64,
            "legalities": {"unlimited": "Legal"},
            "ptcgoCode": "JU",
            "releaseDate": "1999/06/16",
            "updatedAt": "2020/08/14 09:35:00",
            "images": {
                "symbol": "https://images.pokemontcg.io/base2/symbol.png",
                "logo": "https://images.pokemontcg.io/base2/logo.png"
            }
        },
        {
            "id": "xy1",
            "name": "XY",
            "series": "XY",
            "printedTotal": 146,
            "total": 146,
            "legalities": {"unlimited": "Legal", "expanded": "Legal"},
            "ptcgoCode": "XY",
            "releaseDate": "2014/02/05",
            "updatedAt": "2020/08/14 09:35:00",
            "images": {
                "symbol": "https://images.pokemontcg.io/xy1/symbol.png",
                "logo": "https://images.pokemontcg.io/xy1/logo.png"
            }
        }
    ]
    
    # Sample Pokemon cards with detailed data
    sample_cards = [
        {
            "id": "base1-4",
            "name": "Charizard",
            "supertype": "Pokémon",
            "subtypes": ["Stage 2"],
            "level": "76",
            "hp": "120",
            "types": ["Fire"],
            "evolvesFrom": "Charmeleon",
            "attacks": [
                {
                    "name": "Energy Burn",
                    "cost": ["Fire"],
                    "convertedEnergyCost": 1,
                    "damage": "",
                    "text": "As often as you like during your turn (before your attack), you may turn all Energy attached to Charizard into Fire Energy for the rest of the turn. This power can't be used if Charizard is Asleep, Confused, or Paralyzed."
                },
                {
                    "name": "Fire Spin",
                    "cost": ["Fire", "Fire", "Fire", "Fire"],
                    "convertedEnergyCost": 4,
                    "damage": "100",
                    "text": "Discard 2 Energy cards attached to Charizard in order to use this attack."
                }
            ],
            "weaknesses": [
                {
                    "type": "Water",
                    "value": "×2"
                }
            ],
            "resistances": [
                {
                    "type": "Fighting",
                    "value": "-30"
                }
            ],
            "retreatCost": ["Colorless", "Colorless", "Colorless"],
            "convertedRetreatCost": 3,
            "set": {
                "id": "base1",
                "name": "Base Set",
                "series": "Base",
                "printedTotal": 102,
                "total": 102,
                "legalities": {"unlimited": "Legal"},
                "ptcgoCode": "BS",
                "releaseDate": "1999/01/09",
                "updatedAt": "2020/08/14 09:35:00",
                "images": {
                    "symbol": "https://images.pokemontcg.io/base1/symbol.png",
                    "logo": "https://images.pokemontcg.io/base1/logo.png"
                }
            },
            "number": "4",
            "artist": "Mitsuhiro Arita",
            "rarity": "Rare Holo",
            "flavorText": "Spits fire that is hot enough to melt boulders. Known to cause forest fires unintentionally.",
            "nationalPokedexNumbers": [6],
            "legalities": {"unlimited": "Legal"},
            "images": {
                "small": "https://images.pokemontcg.io/base1/4.png",
                "large": "https://images.pokemontcg.io/base1/4_hires.png"
            },
            "tcgplayer": {
                "url": "https://prices.pokemontcg.io/tcgplayer/base1-4",
                "updatedAt": "2022/05/20",
                "prices": {
                    "holofoil": {
                        "low": 89.95,
                        "mid": 124.95,
                        "high": 750.0,
                        "market": 104.02,
                        "directLow": 95.0
                    }
                }
            }
        },
        {
            "id": "base1-25",
            "name": "Pikachu",
            "supertype": "Pokémon",
            "subtypes": ["Basic"],
            "level": "12",
            "hp": "40",
            "types": ["Lightning"],
            "attacks": [
                {
                    "name": "Gnaw",
                    "cost": ["Colorless"],
                    "convertedEnergyCost": 1,
                    "damage": "10",
                    "text": ""
                },
                {
                    "name": "Thunder Jolt",
                    "cost": ["Lightning", "Colorless"],
                    "convertedEnergyCost": 2,
                    "damage": "30",
                    "text": "Flip a coin. If tails, Pikachu does 10 damage to itself."
                }
            ],
            "weaknesses": [
                {
                    "type": "Fighting",
                    "value": "×2"
                }
            ],
            "retreatCost": ["Colorless"],
            "convertedRetreatCost": 1,
            "set": {
                "id": "base1",
                "name": "Base Set",
                "series": "Base",
                "printedTotal": 102,
                "total": 102,
                "legalities": {"unlimited": "Legal"},
                "ptcgoCode": "BS",
                "releaseDate": "1999/01/09",
                "updatedAt": "2020/08/14 09:35:00",
                "images": {
                    "symbol": "https://images.pokemontcg.io/base1/symbol.png",
                    "logo": "https://images.pokemontcg.io/base1/logo.png"
                }
            },
            "number": "25",
            "artist": "Mitsuhiro Arita",
            "rarity": "Common",
            "flavorText": "When several of these Pokémon gather, their electricity could build and cause lightning storms.",
            "nationalPokedexNumbers": [25],
            "legalities": {"unlimited": "Legal"},
            "images": {
                "small": "https://images.pokemontcg.io/base1/25.png",
                "large": "https://images.pokemontcg.io/base1/25_hires.png"
            },
            "tcgplayer": {
                "url": "https://prices.pokemontcg.io/tcgplayer/base1-25",
                "updatedAt": "2022/05/20",
                "prices": {
                    "normal": {
                        "low": 4.25,
                        "mid": 8.5,
                        "high": 25.0,
                        "market": 7.25
                    }
                }
            }
        },
        {
            "id": "xy1-1",
            "name": "Venusaur-EX",
            "supertype": "Pokémon",
            "subtypes": ["Basic", "EX"],
            "hp": "180",
            "types": ["Grass"],
            "attacks": [
                {
                    "name": "Poison Powder",
                    "cost": ["Grass", "Colorless", "Colorless"],
                    "convertedEnergyCost": 3,
                    "damage": "60",
                    "text": "Your opponent's Active Pokémon is now Poisoned."
                },
                {
                    "name": "Jungle Hammer",
                    "cost": ["Grass", "Grass", "Colorless", "Colorless"],
                    "convertedEnergyCost": 4,
                    "damage": "90",
                    "text": "Heal 30 damage from this Pokémon."
                }
            ],
            "weaknesses": [
                {
                    "type": "Fire",
                    "value": "×2"
                }
            ],
            "retreatCost": ["Colorless", "Colorless", "Colorless", "Colorless"],
            "convertedRetreatCost": 4,
            "set": {
                "id": "xy1",
                "name": "XY",
                "series": "XY",
                "printedTotal": 146,
                "total": 146,
                "legalities": {"unlimited": "Legal", "expanded": "Legal"},
                "ptcgoCode": "XY",
                "releaseDate": "2014/02/05",
                "updatedAt": "2020/08/14 09:35:00",
                "images": {
                    "symbol": "https://images.pokemontcg.io/xy1/symbol.png",
                    "logo": "https://images.pokemontcg.io/xy1/logo.png"
                }
            },
            "number": "1",
            "artist": "Eske Yoshinob",
            "rarity": "Rare Holo EX",
            "nationalPokedexNumbers": [3],
            "legalities": {"unlimited": "Legal", "expanded": "Legal"},
            "images": {
                "small": "https://images.pokemontcg.io/xy1/1.png",
                "large": "https://images.pokemontcg.io/xy1/1_hires.png"
            },
            "tcgplayer": {
                "url": "https://prices.pokemontcg.io/tcgplayer/xy1-1",
                "updatedAt": "2022/05/20",
                "prices": {
                    "holofoil": {
                        "low": 2.25,
                        "mid": 4.5,
                        "high": 12.99,
                        "market": 3.87
                    }
                }
            }
        },
        {
            "id": "base2-7",
            "name": "Kangaskhan",
            "supertype": "Pokémon",
            "subtypes": ["Basic"],
            "level": "40",
            "hp": "90",
            "types": ["Colorless"],
            "attacks": [
                {
                    "name": "Fetch",
                    "cost": ["Colorless"],
                    "convertedEnergyCost": 1,
                    "damage": "",
                    "text": "Draw a card."
                },
                {
                    "name": "Comet Punch",
                    "cost": ["Colorless", "Colorless", "Colorless", "Colorless"],
                    "convertedEnergyCost": 4,
                    "damage": "20×",
                    "text": "Flip 4 coins. This attack does 20 damage times the number of heads."
                }
            ],
            "weaknesses": [
                {
                    "type": "Fighting",
                    "value": "×2"
                }
            ],
            "resistances": [
                {
                    "type": "Psychic",
                    "value": "-30"
                }
            ],
            "retreatCost": ["Colorless", "Colorless"],
            "convertedRetreatCost": 2,
            "set": {
                "id": "base2",
                "name": "Jungle",
                "series": "Base",
                "printedTotal": 64,
                "total": 64,
                "legalities": {"unlimited": "Legal"},
                "ptcgoCode": "JU",
                "releaseDate": "1999/06/16",
                "updatedAt": "2020/08/14 09:35:00",
                "images": {
                    "symbol": "https://images.pokemontcg.io/base2/symbol.png",
                    "logo": "https://images.pokemontcg.io/base2/logo.png"
                }
            },
            "number": "7",
            "artist": "Mitsuhiro Arita",
            "rarity": "Rare Holo",
            "flavorText": "The infant rarely ventures out of its mother's protective pouch until it is 3 years old.",
            "nationalPokedexNumbers": [115],
            "legalities": {"unlimited": "Legal"},
            "images": {
                "small": "https://images.pokemontcg.io/base2/7.png",
                "large": "https://images.pokemontcg.io/base2/7_hires.png"
            },
            "tcgplayer": {
                "url": "https://prices.pokemontcg.io/tcgplayer/base2-7",
                "updatedAt": "2022/05/20",
                "prices": {
                    "holofoil": {
                        "low": 15.0,
                        "mid": 25.0,
                        "high": 75.0,
                        "market": 22.5
                    }
                }
            }
        }
    ]
    
    return sample_sets, sample_cards

def populate_sample_database():
    """
    Populate the database with sample Pokemon card data for testing.
    """
    with app.app_context():
        # Create tables if they don't exist
        db.create_all()
        
        sample_sets, sample_cards = create_sample_card_data()
        
        # Add sample sets
        sets_added = 0
        for set_data in sample_sets:
            existing_set = PokemonSet.query.get(set_data['id'])
            if not existing_set:
                pokemon_set = PokemonSet(
                    id=set_data['id'],
                    name=set_data['name'],
                    series=set_data['series'],
                    printed_total=set_data['printedTotal'],
                    total=set_data['total'],
                    legalities=json.dumps(set_data['legalities']),
                    ptcgo_code=set_data['ptcgoCode'],
                    release_date=set_data['releaseDate'],
                    updated_at=set_data['updatedAt'],
                    symbol_url=set_data['images']['symbol'],
                    logo_url=set_data['images']['logo'],
                    data_json=json.dumps(set_data)
                )
                db.session.add(pokemon_set)
                sets_added += 1
        
        # Add sample cards
        cards_added = 0
        for card_data in sample_cards:
            existing_card = OfficialCard.query.get(card_data['id'])
            if not existing_card:
                images = card_data.get('images', {})
                set_data = card_data.get('set', {})
                
                official_card = OfficialCard(
                    id=card_data['id'],
                    name=card_data['name'],
                    set_id=set_data.get('id'),
                    set_name=set_data.get('name'),
                    number=card_data.get('number'),
                    rarity=card_data.get('rarity'),
                    artist=card_data.get('artist'),
                    hp=int(card_data.get('hp', 0)) if card_data.get('hp') and str(card_data.get('hp')).isdigit() else None,
                    types=json.dumps(card_data.get('types', [])),
                    attacks=json.dumps(card_data.get('attacks', [])),
                    weaknesses=json.dumps(card_data.get('weaknesses', [])),
                    resistances=json.dumps(card_data.get('resistances', [])),
                    retreat_cost=len(card_data.get('retreatCost', [])) if card_data.get('retreatCost') else None,
                    converted_energy_cost=card_data.get('convertedEnergyCost'),
                    tcgplayer_url=card_data.get('tcgplayer', {}).get('url'),
                    image_url_small=images.get('small'),
                    image_url_large=images.get('large'),
                    data_json=json.dumps(card_data)
                )
                db.session.add(official_card)
                cards_added += 1
        
        # Commit all changes
        db.session.commit()
        
        print(f"Sample database populated successfully!")
        print(f"Added {sets_added} sets and {cards_added} cards")
        return sets_added, cards_added

def get_card_comparison_data(card_name: str):
    """
    Get official card data for comparison during authenticity checking.
    
    Args:
        card_name: Name of the Pokemon card to look up
        
    Returns:
        List of matching official cards
    """
    with app.app_context():
        # Search for cards with similar names
        cards = OfficialCard.query.filter(
            OfficialCard.name.ilike(f'%{card_name}%')
        ).all()
        
        return cards

if __name__ == "__main__":
    sets_added, cards_added = populate_sample_database()
    print(f"Database initialized with {sets_added} sets and {cards_added} cards")