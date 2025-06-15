"""
deck.py

This module contains the implementation of the deck for the Jaipur game.
The deck is responsible for managing the cards used in the game.
"""

import random
import numpy as np
from typing import List, Optional
from collections import Counter
from jaipur_rl.config import CardType, TOTAL_CARD_COUNTS, INITIAL_MARKET_CAMELS, CARD_IDX, NUM_CARD_TYPES


class Deck:
    def __init__(self, seed: Optional[int] = None):
        """Initialize and build the deck"""
        self._rng = random.Random(seed)
        self.cards = self._build_deck()

    def _build_deck(self) -> List[CardType]:
        """Build the deck of cards for the game."""
        _card_counts = TOTAL_CARD_COUNTS.copy()
        _card_counts[CardType.CAMEL] -= INITIAL_MARKET_CAMELS

        _deck = []
        [_deck.extend([card] * amount) for card, amount in _card_counts.items()]

        self._rng.shuffle(_deck)
        return _deck

    def draw(self) -> Optional[CardType]:
        """Draw a card from the deck."""
        return self.cards.pop() if self.cards else None

    def get_deck_percentages(self) -> np.ndarray:
        """Calculate the percentage of each card type in the deck."""
        deck_len = len(self.cards)
        if deck_len == 0:
            return np.zeros(NUM_CARD_TYPES, dtype=np.float32)

        pct = np.zeros(NUM_CARD_TYPES, dtype=np.float32)
        counts = dict(Counter(self.cards))

        for card, index in CARD_IDX.items():
            pct[index] = counts.get(card, 0) / deck_len

        return pct

    def is_empty(self) -> bool:
        """Check if the deck is empty."""
        return not self.cards

    def __len__(self):
        """Return the number of cards in the deck."""
        return len(self.cards)


if __name__ == "__main__":
    deck = Deck(seed=42)
    print(f"Deck size: {len(deck)}")
    print(f"Deck percentages: {deck.get_deck_percentages()}")
    print(f"deck: {[card.value for card in deck.cards]}")
    print(f"Drawing a card: {deck.draw()}")
    print(f"Deck size after drawing a card: {len(deck)}")
    print(f"Is deck empty? {deck.is_empty()}")