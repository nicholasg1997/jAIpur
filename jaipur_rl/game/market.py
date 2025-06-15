"""
market.py

This module contains the implementation of the market for the Jaipur game.
The market is responsible for managing the cards available for players to take.
"""

import numpy as np
from typing import List, Optional
from jaipur_rl.config import CardType, INITIAL_MARKET_CAMELS, MARKET_SIZE, NUM_CARD_TYPES, CARD_IDX
from jaipur_rl.game.deck import Deck



class Market:
    def __init__(self, deck: Deck):
        self.deck = deck
        self.market_cards: List[CardType] = []
        self._build_initial_market()

    def _build_initial_market(self) -> None:
        self.market_cards = [CardType.CAMEL] * INITIAL_MARKET_CAMELS
        self.fill()

    def fill(self) -> None:
        while len(self.market_cards) < MARKET_SIZE:
            c = self.deck.draw()
            if c:
                self.market_cards.append(c)
            else:
                break

    def take_card_by_index(self, i: int) -> Optional[CardType]:
        if 0 <= i < len(self.market_cards):
            return self.market_cards.pop(i)
        return None

    def take_all_camels(self) -> List[CardType]:
        ct = [c for c in self.market_cards if c == CardType.CAMEL]
        if ct:
            self.market_cards = [c for c in self.market_cards if c != CardType.CAMEL]
        return ct

    def get_cards(self) -> List[CardType]:
        return list(self.market_cards)

    def reset(self, d: Deck) -> None:
        self.deck = d
        self._build_initial_market()

    def get_market_matrix(self) -> np.ndarray:
        matrix = np.zeros((MARKET_SIZE, NUM_CARD_TYPES), dtype=np.float32)
        for slot, card in enumerate(self.market_cards):
            if slot >= MARKET_SIZE:
                break
            if card is not None:
                card_index = CARD_IDX.get(card, -1)
                if card_index != -1:
                    matrix[slot, card_index] = 1.0
        return matrix

    def add_cards_to_market(self, cards: List[CardType]):
        self.market_cards.extend(cards)

    def remove_cards_by_indices_from_market(self, itr: List[int]) -> List[CardType]:
        if not itr:  # itr = index to remove
            return []
        current_market_size = len(self.market_cards)
        for i_val in itr:
            if not (0 <= i_val < current_market_size):
                return []
        removed_cards = []
        temp_market = list(self.market_cards)
        sorted_indices_to_pop = sorted(list(set(itr)), reverse=True)  # index is reversed so correct card is taken
        try:
            for i_pop in sorted_indices_to_pop:
                removed_cards.append(temp_market.pop(i_pop))
        except IndexError:
            return []
        if len(removed_cards) == len(set(itr)):
            self.market_cards = temp_market
            return list(reversed(removed_cards))
        return []


if __name__ == "__main__":
    # Example usage
    deck = Deck()
    market = Market(deck)
    print("Initial market:", market.get_cards())
    print("Market matrix:\n", market.get_market_matrix())
    print("Taking card at index 0:", market.take_card_by_index(3))
    market.fill()
    print("Market after taking card:", market.get_cards())
    print("Taking all camels:", market.take_all_camels())
    market.fill()
    print("Market after taking all camels:", market.get_cards())
    print("Market matrix after taking all camels:\n", market.get_market_matrix())