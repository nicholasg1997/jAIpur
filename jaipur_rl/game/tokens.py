"""
tokens.py

This module contains the implementation of the tokens for the Jaipur game.
The Tokens class is responsible for managing the tokens used in the game.
"""
import copy

from configs.game_configs import CardType, GOODS_TOKENS_VALUES, BONUS_TOKENS_CONFIG, NUM_GOOD_TYPES, GOOD_IDX_TO_CARD
import random
from typing import List, Tuple, Any, Optional
import numpy as np


class TokenBank:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.reset()

    def _shuffle_list(self, d: List[Any]):
        self._rng.shuffle(d)

    def reset(self) -> None:
        self.goods_tokens = copy.deepcopy(GOODS_TOKENS_VALUES)
        self.bonus_tokens = copy.deepcopy(BONUS_TOKENS_CONFIG)
        for k in self.bonus_tokens:
            self._shuffle_list(self.bonus_tokens[k])

        self._initial_token_counts = {good: len(tokens) for good, tokens in GOODS_TOKENS_VALUES.items()}

    def take_goods_tokens(self, good: CardType, amount: int) -> Tuple[int, int]:
        if good == CardType.CAMEL or good not in self.goods_tokens:
            return 0, 0

        stack = self.goods_tokens[good]
        num_taken = min(amount, len(stack))

        if num_taken == 0:
            return 0, 0

        tt = stack[:num_taken]
        self.goods_tokens[good] = stack[num_taken:]
        points = sum(tt)
        bonus = self._take_bonus_token(amount)
        return points, bonus

    def _take_bonus_token(self, amount: int) -> int:
        if amount < 3:
            return 0
        bonus_category = min(amount, 5)
        stack = self.bonus_tokens.get(bonus_category)
        if stack and len(stack) > 0:
            return stack.pop(0)
        return 0

    def count_depleted_stacks(self) -> int:
        return sum(1 for s in self.goods_tokens.values() if not s)

    def get_token_fullness(self) -> np.ndarray:
        fullness = np.zeros(NUM_GOOD_TYPES, dtype=np.float32)
        for good_idx, good_type in GOOD_IDX_TO_CARD.items():
            initial_count = self._initial_token_counts.get(good_type, -1)
            if initial_count > 0:
                fullness[good_idx] = len(self.goods_tokens.get(good_type, [])) / initial_count
            else:
                fullness[good_idx] = 0.0
        return fullness

    def peek_top_token_values(self, good: CardType, count: int) -> List[int]:
        if good == CardType.CAMEL or good not in self.goods_tokens:
            return []
        stack = self.goods_tokens[good]
        return stack[:min(count, len(stack))]

    def peek_bonus_token_value(self,
                               num_cards: int) -> int:
        if num_cards < 3:
            return 0
        bonus_category = min(num_cards, 5)
        stack = self.bonus_tokens.get(bonus_category)
        if stack and len(stack) > 0:
            return stack[0]
        return 0


if __name__ == "__main__":
    token_bank = TokenBank()
    print(f"Initial goods tokens: {token_bank.goods_tokens}")
    print(f"Initial bonus tokens: {token_bank.bonus_tokens}")
    print(f"Taking 3 leather tokens: {token_bank.take_goods_tokens(CardType.LEATHER, 3)}")
    print(f"Goods tokens after taking: {token_bank.goods_tokens}")
    print(f"Bonus tokens after taking: {token_bank.bonus_tokens}")
    print(f"Depleted stacks count: {token_bank.count_depleted_stacks()}")
    print(f"Token fullness: {token_bank.get_token_fullness()}")