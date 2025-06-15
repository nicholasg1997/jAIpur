"""
player.py

This module contains the implementation of the player for the Jaipur game.
The player is responsible for managing the player's hand, camels, and points.
"""

from jaipur_rl.config import *
from typing import List, Tuple, Dict, Set
import numpy as np
from jaipur_rl.game.tokens import TokenBank

class Player:
    def __init__(self, name: str):
        self.name = name
        self.hand: List[CardType] = []
        self.herd: int = 0
        self.tokens: List[Tuple[CardType, int]] = []
        self.bonus_tokens: List[int] = []
        self.total_score: int = 0
        self.known_opp_cards: Dict[CardType, int] = {g: 0 for g in GOOD_IDX_TO_CARD.values()}
        self.rewarded_set_bonus: Dict[CardType, Set[int]] = {g: set() for g in GOOD_IDX_TO_CARD.values()}
        self._CARD_TO_GOOD_IDX = {card: idx for idx, card in GOOD_IDX_TO_CARD.items()}

    def reset(self) -> None:
        self.hand = []
        self.herd = 0
        self.tokens = []
        self.bonus_tokens = []
        self.total_score = 0
        self.known_opp_cards = {g: 0 for g in GOOD_IDX_TO_CARD.values()}
        self.rewarded_set_bonus = {g: set() for g in GOOD_IDX_TO_CARD.values()}

    def add_card(self, card: CardType) -> bool:
        if card == CardType.CAMEL:
            self.herd += 1
            return True

        if len(self.hand) < HAND_LIMIT and card in self._CARD_TO_GOOD_IDX:
            self.hand.append(card)
            return True

        return False

    def add_cards(self, cards: List[CardType]) -> int:
        total_cards_added = 0
        for card in cards:
            added = self.add_card(card)
            if added:
                total_cards_added += 1
        return total_cards_added

    def remove_goods_of_type(self, good_type: CardType, c: int) -> List[CardType]:
        removed_cards = []
        curr_hand = list(self.hand)
        for _ in range(c):
            if good_type in curr_hand:
                curr_hand.remove(good_type)
                removed_cards.append(good_type)
            else:
                return []
        self.hand = curr_hand
        return removed_cards

    def remove_cards_by_indices_from_hand(self, itr: List[int]) -> List[CardType]:

        if not itr:  # itr = indices to remove from hand
            return []

        current_hand_size = len(self.hand)
        for i_val in itr:
            if not (0 <= i_val < current_hand_size):  # Check if index is in current hand length
                return []

        removed_cards = []
        sorted_indexes = sorted(list(set(itr)), reverse=True)
        curr_hand = list(self.hand)
        try:
            for i_pop in sorted_indexes:
                removed_cards.append(curr_hand.pop(i_pop))
        except IndexError:
            return []
        self.hand = curr_hand

        return list(reversed(removed_cards))

    def count_good(self, good: CardType) -> int:
        return self.hand.count(good)

    def hand_size(self) -> int:
        return len(self.hand)

    def can_sell(self, good: CardType) -> bool:
        if good == CardType.CAMEL or good not in MIN_SELL:
            return False

        return self.count_good(good) >= MIN_SELL[good]

    def record_sale(self, good: CardType, token_points: int, bonus_points: int, ns: int) -> None:
        if token_points > 0:
            self.tokens.append((good, token_points))

        if bonus_points > 0:
            self.bonus_tokens.append(bonus_points)

    def calculate_final_score(self, camel_bonus: int = 0) -> int:
        total_from_tokens = sum(p for _, p in self.tokens)
        total_bonus_points = sum(self.bonus_tokens)
        self.total_score = total_from_tokens + total_bonus_points + camel_bonus
        return self.total_score

    def get_hand_matrix(self) -> np.ndarray:
        matrix = np.zeros((HAND_LIMIT, NUM_GOOD_TYPES), dtype=np.float32)
        for slot, card in enumerate(self.hand):
            if slot >= HAND_LIMIT:
                break
            if card == CardType.CAMEL:
                continue

            good_idx = self._CARD_TO_GOOD_IDX.get(card)
            if good_idx is not None:
                matrix[slot, good_idx] = 1.0

        for empty_slots in range(len(self.hand), HAND_LIMIT):
            matrix[empty_slots, :] = -1.0

        return matrix

    def good_counts_in_hand(self):
        counts = {good: 0 for good in GOOD_IDX_TO_CARD.values()}
        for card in self.hand:
            if card != CardType.CAMEL:
                counts[card] += 1
        return counts.values()

    def goods_can_sell(self):
        goods = {good: 0 for good in GOOD_IDX_TO_CARD.values()}
        for card in set(self.hand):
            if card != CardType.CAMEL and self.can_sell(card):
                goods[card] = 1

        return goods.values()

    def get_known_opp_cards_vector(self) -> np.ndarray:
        v = np.zeros(NUM_GOOD_TYPES, dtype=np.float32)

        for idx, good_type in GOOD_IDX_TO_CARD.items():
            v[idx] = self.known_opp_cards.get(good_type, 0)
            total_counts = TOTAL_CARD_COUNTS.get(good_type, 1)
            v[idx] = min(1.0, v[idx] / total_counts)

        return v.astype(np.float32)

    def update_known_opp_card(self, card: CardType, amount: int):
        if card in self.known_opp_cards:
            self.known_opp_cards[card] = max(0, self.known_opp_cards[card] + amount)

    def get_heuristic_hand_value(self, token_bank: TokenBank, calculate_set_bonus_for_reward: bool = False) -> Tuple[
        float, float]:
        hand_goods_value = 0.0
        one_time_set_bonus_achieved_this_step = 0.0

        for good_type in GOOD_IDX_TO_CARD.values():
            count = self.count_good(good_type)
            if count > 0:
                potential_sell_value = sum(token_bank.peek_top_token_values(good_type, count))

                if self.can_sell(good_type):  # Meets MIN_SELL criteria
                    hand_goods_value += potential_sell_value * K1_SELLABLE_FACTOR
                else:  # Does not meet MIN_SELL (less valuable for immediate sale)
                    hand_goods_value += potential_sell_value * K2_UNSELLABLE_FACTOR

                if calculate_set_bonus_for_reward:
                    for set_size_threshold, bonus_value in BONUS_TOKEN_POTENTIAL_REWARDS.items():
                        if count >= set_size_threshold and set_size_threshold not in self.rewarded_set_bonus[good_type]:
                            one_time_set_bonus_achieved_this_step += bonus_value
                            self.rewarded_set_bonus[good_type].add(set_size_threshold)

        camels_value = self.herd * CAMEL_VALUE_IN_HAND_HEURISTIC
        total_value = hand_goods_value + camels_value
        return total_value, one_time_set_bonus_achieved_this_step

    def update_set_bonuses_after_sale(self, good_type: CardType, count_after_sale: int):
        """
        Called after a sale to reset any bonuses that are no longer met.
        """
        for set_size in range(3, 6):
            if set_size in self.rewarded_set_bonus[good_type] and count_after_sale < set_size:
                self.rewarded_set_bonus[good_type].remove(set_size)


if __name__ == "__main__":
    player = Player("TestPlayer")
    print(player.add_card(CardType.CAMEL))
    print(player.add_card(CardType.LEATHER))
    print(player.add_card(CardType.SPICE))
    print(player.add_card(CardType.CLOTH))
    print(player.add_card(CardType.SPICE))
    print(player.add_card(CardType.GOLD))
    print(player.add_card(CardType.DIAMOND))
    print(player.hand)
    print(player.get_hand_matrix())

    # test known opponent cards
    player.update_known_opp_card(CardType.LEATHER, 2)
    player.update_known_opp_card(CardType.SPICE, 1)
    print(player.get_known_opp_cards_vector())

