"""
jaipur_game.py

This module contains the implementation of the Jaipur game logic.
The Jaipur class is responsible for managing the game state, handling actions, and providing observations and rewards.
"""

from jaipur_rl.game.deck import Deck
from jaipur_rl.game.market import Market
from jaipur_rl.game.tokens import TokenBank
from typing import Optional
import random


class JaipurGame:
    def __init__(self, seed: Optional[int] = None):
        self.deck = Deck(seed=seed)
        self.token_bank = TokenBank(seed=seed)
        self.market = Market(self.deck)

    def reset(self, seed: Optional[int] = None):
        deck_seed, token_seed = (random.Random(seed).randint(0, 2 ** 32 - 1), random.Random(seed).randint(0, 2 ** 32 - 1)) \
            if seed else (None, None)
        self.deck = Deck(seed=deck_seed)
        self.token_bank.reset()
        self.market.reset(self.deck)


if __name__ == "__main__":
    game = JaipurGame()
    game.reset(seed=456)
    print("Deck:", game.deck.cards)
    print("Market:", game.market.market_cards)
    print("Token Bank:", game.token_bank.goods_tokens)
    print("Bonus Tokens:", game.token_bank.bonus_tokens)
