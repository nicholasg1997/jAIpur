"""
jaipur_env.py

This module contains the implementation of the Gym environment for the Jaipur game.
The environment is responsible for managing the game state, handling actions, and providing observations and rewards.
"""

import gymnasium as gym
import numpy as np
import random
from typing import Optional, Tuple, Dict, List
from jaipur_rl.game.player import Player
from jaipur_rl.config import *
from jaipur_rl.game.jaipur_game import JaipurGame


class JaipurEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, seed: Optional[int] = None, max_steps: int = 200):
        super().__init__()
        self.seed_val = seed
        self.max_steps = max_steps
        self._rng = np.random.default_rng(self.seed_val)
        self.game = JaipurGame(seed=self.seed_val)
        self.players = [Player("P1"), Player("P2")]
        self.current_step = 0
        self.current_player_idx = 0

        self.action_space = gym.spaces.Dict({
            "main_action_type": gym.spaces.Discrete(3),  # 0:TakeCamels, 1:Sell, 2:TakeSpecificOrExchange
            "sell_good_type_index": gym.spaces.Discrete(NUM_GOOD_TYPES),
            # For action 2 (TakeSpecificOrExchange)
            "exchange_market_take_mask": gym.spaces.MultiBinary(MARKET_SIZE),  # Agent selects which market cards
            "exchange_hand_give_indices": gym.spaces.MultiBinary(HAND_LIMIT)  # Agent selects which hand cards (by slot)

        })

        self.observation_space = gym.spaces.Dict({
            "market": gym.spaces.Box(low=0, high=1, shape=(MARKET_SIZE, NUM_CARD_TYPES), dtype=np.float32),
            "camels_in_market": gym.spaces.Box(low=0, high=1, shape=(MARKET_SIZE,), dtype=np.float32),
            "hand": gym.spaces.Box(low=0, high=1, shape=(HAND_LIMIT, NUM_GOOD_TYPES), dtype=np.float32),
            "score": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "good_counts_in_hand": gym.spaces.Box(low=0, high=1, shape=(NUM_GOOD_TYPES,), dtype=np.float32),
            "goods_can_sell": gym.spaces.Box(low=0, high=1, shape=(NUM_GOOD_TYPES,), dtype=np.float32),
            "herd_count": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "token_fullness": gym.spaces.Box(low=0, high=1, shape=(NUM_GOOD_TYPES,), dtype=np.float32),
            "empty_token_stacks": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "deck_pct": gym.spaces.Box(low=0, high=1, shape=(NUM_CARD_TYPES,), dtype=np.float32),
            "known_opp_cards": gym.spaces.Box(low=0, high=1, shape=(NUM_GOOD_TYPES,), dtype=np.float32),
            "hand_size": gym.spaces.Box(low=0, high=1, shape=(HAND_LIMIT + 1,), dtype=np.float32),
            "hand_good_fullness": gym.spaces.Box(low=0, high=1, shape=(NUM_GOOD_TYPES,), dtype=np.float32),
            "opp_hand_size": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "opp_herd_size": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "opp_score": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "action_mask": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
            "take_mask": gym.spaces.Box(low=0, high=1, shape=(MARKET_SIZE,), dtype=np.int8),
        })

        self.current_player: Player = self._get_current_player()
        self._consecutive_invalid_attempts = 0
        self.episode_action_counts = np.zeros(3, dtype=int)

    def _get_current_player(self) -> Player:
        return self.players[self.current_player_idx]

    def _get_opponent(self) -> Player:
        return self.players[1 - self.current_player_idx]

    def _deal_starting_hands(self) -> None:
        for p in self.players:
            p.reset()
            [p.add_card(c) for _ in range(5) if (c := self.game.deck.draw())]

    def _reset_game_state(self) -> None:
        self.game.reset(seed=self.seed_val)
        self._deal_starting_hands()
        self.current_player_idx = self._rng.choice([0, 1])
        self.current_player = self._get_current_player()
        self.current_step = 0
        self._consecutive_invalid_attempts = 0
        self.episode_action_counts = np.zeros(3, dtype=int)
        for p in self.players:
            p.rewarded_set_bonus = {g: set() for g in GOOD_IDX_TO_CARD.values()}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed_val = seed
            self._rng = np.random.default_rng(self.seed_val)
        self._reset_game_state()

        observation = self._get_obs()
        _info = self._get_info()
        return observation, _info

    def _get_heuristic_market_value(self) -> float:
        market_value = 0.0
        for card in self.game.market.get_cards():
            if card != CardType.CAMEL and card in GOODS_TOKENS_VALUES:
                top_tokens = self.game.token_bank.peek_top_token_values(card, 1)
                if top_tokens:
                    market_value += top_tokens[0]
        return market_value

    def _validate_and_execute_take_or_exchange(self, ad: Dict) -> Tuple[bool, str, float]:
        player = self.current_player
        opponent = self._get_opponent()
        current_market_cards = self.game.market.get_cards()

        market_take_mask = ad["exchange_market_take_mask"]  # ad = action dictionary
        indices_to_take_from_market: List[int] = [i for i, take in enumerate(market_take_mask) if
                                                  take == 1 and i < len(current_market_cards)]
        cards_to_take: List[CardType] = []
        for idx in indices_to_take_from_market:
            card = current_market_cards[idx]
            if card == CardType.CAMEL:
                return False, "Cannot take camels with Take/exchange.", PENALTY_INVALID_ACTION
            cards_to_take.append(card)
        num_market_cards_taken = len(cards_to_take)

        if num_market_cards_taken == 0:
            return False, "Must select >=1 card for Take/Exchange.", PENALTY_INVALID_ACTION

        action_specific_base_reward = 0.0

        if num_market_cards_taken == 1:  # No exchange required for single take
            if player.hand_size() >= HAND_LIMIT:
                return False, "Hand limit for single take.", PENALTY_INVALID_ACTION

            card_taken_instance = cards_to_take[0]
            idx_to_take = indices_to_take_from_market[0]
            removed_card = self.game.market.remove_cards_by_indices_from_market([idx_to_take])
            if not removed_card or removed_card[0] != card_taken_instance:
                if removed_card:
                    self.game.market.add_cards_to_market(removed_card)  # Return removed card
                return False, "Failed to take single card from market.", PENALTY_INVALID_ACTION

            if not player.add_card(card_taken_instance):
                self.game.market.add_cards_to_market(removed_card)  # Return removed card
                return False, "Failed to add taken card to hand.", PENALTY_INVALID_ACTION

            new_deck_card = self.game.deck.draw()
            if new_deck_card:
                self.game.market.add_cards_to_market([new_deck_card])
            self.game.market.fill()
            opponent.update_known_opp_card(card_taken_instance, 1)
            action_specific_base_reward = BASE_REWARD_ACTION_SUCCESS

        else:  # N-for-N Exchange (num_market_cards_taken > 1)
            hand_give_indices_mask = ad["exchange_hand_give_indices"]

            valid_hand_mask = hand_give_indices_mask[:player.hand_size()]
            indices_of_hand_cards_to_give: List[int] = [i for i, give in enumerate(valid_hand_mask) if
                                                        give == 1 and i < player.hand_size()]
            num_goods_given_from_hand = len(indices_of_hand_cards_to_give)

            camels_needed_to_give = num_market_cards_taken - num_goods_given_from_hand

            if camels_needed_to_give < 0:
                return False, f"N-N Imbalance: Trying to give {num_goods_given_from_hand} goods for {num_market_cards_taken} market cards (too many goods given).", PENALTY_INVALID_ACTION

            if camels_needed_to_give > player.herd:
                return False, f"Not enough camels for N-N: Need {camels_needed_to_give}, Have {player.herd}.", PENALTY_INVALID_ACTION

            if (player.hand_size() + camels_needed_to_give) > HAND_LIMIT:
                return False, "N-N Exceeds hand limit.", PENALTY_INVALID_ACTION

            action_specific_base_reward += VALID_N_FOR_N_EXCHANGE_CONSTRUCTION_BONUS

            # Execute the N-for-N exchange
            taken_from_market = self.game.market.remove_cards_by_indices_from_market(indices_to_take_from_market)
            if len(taken_from_market) != num_market_cards_taken:
                if taken_from_market:
                    self.game.market.add_cards_to_market(taken_from_market)
                return False, "N-N Market execution fail.", action_specific_base_reward

            given_from_hand = player.remove_cards_by_indices_from_hand(
                indices_of_hand_cards_to_give) if num_goods_given_from_hand > 0 else []
            if len(given_from_hand) != num_goods_given_from_hand:
                if given_from_hand:
                    player.add_cards(given_from_hand)  # Return cards from hand (if any given)
                self.game.market.add_cards_to_market(taken_from_market)  # Return taken cards to market
                return False, "N-N Hand exececution fail.", action_specific_base_reward

            player.herd -= camels_needed_to_give
            player.add_cards(taken_from_market)
            self.game.market.add_cards_to_market(given_from_hand + [CardType.CAMEL] * camels_needed_to_give)
            self.game.market.fill()
            for card_taken in taken_from_market:
                opponent.update_known_opp_card(card_taken, 1)
            for card_given in given_from_hand:
                opponent.update_known_opp_card(card_given, -1)
            action_specific_base_reward += BASE_REWARD_N_FOR_N_EXCHANGE_SUCCESS

        return True, "Take/Exchange successful.", action_specific_base_reward

    def step(self, ad: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        term, trunc = False, False
        inf = {"action_main_type": ad["main_action_type"]}
        ap = self.current_player  # Active player
        pp = self._get_opponent()  # Passive player (opponent)
        mat = ad["main_action_type"]
        action_valid = True
        action_error_reason = ""

        hv_before_action, _ = ap.get_heuristic_hand_value(self.game.token_bank, calculate_set_bonus_for_reward=False)
        market_value_at_step_start = self._get_heuristic_market_value()

        points_from_sale_this_step = 0.0
        action_base_success_bonus_this_step = 0.0

        hand_mask = np.zeros(HAND_LIMIT, dtype=np.int8)
        hand_mask[:ap.hand_size()] = 1

        if "exchange_hand_give_indices" in ad:
            ad["exchange_hand_give_indices"] *= hand_mask  # Ensure indices are valid

        if mat == 0:  # Take Camels
            num_camels_in_market = sum(1 for card in self.game.market.get_cards() if card == CardType.CAMEL)
            if num_camels_in_market == 0:
                action_valid = False
                action_error_reason = "Invalid: No camels."

            else:
                camels_taken = self.game.market.take_all_camels()
                ap.add_cards(camels_taken)
                self.game.market.fill()
                action_base_success_bonus_this_step = len(
                    camels_taken) * REWARD_FACTOR_CAMEL_TAKEN + BASE_REWARD_ACTION_SUCCESS

        elif mat == 1:  # Sell Goods
            sell_index = ad["sell_good_type_index"]
            if not (0 <= sell_index < NUM_GOOD_TYPES):
                action_valid = False
                action_error_reason = "Invalid sell index."

            else:
                good_to_sell = GOOD_IDX_TO_CARD[sell_index]
                if not ap.can_sell(good_to_sell):
                    action_valid = False
                    action_error_reason = f"Invalid: Cannot sell {good_to_sell.value}."

                else:
                    amount_to_sell = ap.count_good(good_to_sell)
                    pp.update_known_opp_card(good_to_sell, -amount_to_sell)
                    cards_sold = ap.remove_goods_of_type(good_to_sell, amount_to_sell)
                    if not cards_sold and amount_to_sell > 0:
                        action_valid = False
                        action_error_reason = "Error executing sell."
                    else:
                        token_points, bonus_points = self.game.token_bank.take_goods_tokens(good_to_sell,
                                                                                            len(cards_sold))
                        ap.record_sale(good_to_sell, token_points, bonus_points, len(cards_sold))
                        points_from_sale_this_step = float(token_points + bonus_points)
                        action_base_success_bonus_this_step = BASE_REWARD_ACTION_SUCCESS
                        ap.update_set_bonuses_after_sale(good_to_sell, amount_to_sell)

        elif mat == 2:  # Take Specific or Exchange
            success, reason, base_take_exchange_reward = self._validate_and_execute_take_or_exchange(ad)
            action_base_success_bonus_this_step = base_take_exchange_reward
            if not success:
                action_valid = False
                action_error_reason = reason
        else:
            action_valid = False
            action_error_reason = f"Invalid main action type: {mat}"

        if not action_valid:
            self._consecutive_invalid_attempts += 1
            # current_step_reward = min(action_base_success_bonus_this_step, PENALTY_INVALID_ACTION)
            if action_base_success_bonus_this_step >= 0:
                current_step_reward = PENALTY_INVALID_ACTION
            else:
                current_step_reward = action_base_success_bonus_this_step

            inf["error"] = action_error_reason
            inf["consecutive_invalid_attempts"] = self._consecutive_invalid_attempts
            if self._consecutive_invalid_attempts >= MAX_CONSECUTIVE_INVALID_MOVES_PER_TURN:
                inf["forced_pass_due_to_invalid_streak"] = True
                self.current_step += 1
                self.current_player_idx = 1 - self.current_player_idx
                self.current_player = self._get_current_player()
                self._consecutive_invalid_attempts = 0
                if self.current_step >= self.max_steps:
                    trunc = True
            if 0 <= mat < 3:
                self.episode_action_counts[mat] += 1

            current_step_reward += -(market_value_at_step_start * MARKET_VALUE_PENALTY_FACTOR)
            return self._get_obs(), current_step_reward, term, trunc, inf

        self.current_step += 1
        self._consecutive_invalid_attempts = 0
        if 0 <= mat < 3:
            self.episode_action_counts[mat] += 1

        hv_after_action, set_bonus_this_step = ap.get_heuristic_hand_value(self.game.token_bank,
                                                                           calculate_set_bonus_for_reward=True)
        delta_hvc = hv_after_action - hv_before_action
        market_value_after_action = self._get_heuristic_market_value()

        if mat == 1:
            current_step_reward = (points_from_sale_this_step * REWARD_SCALING_FACTOR_SELL_POINTS)
            current_step_reward += (delta_hvc * REWARD_FACTOR_DELTA_HAND_HEURISTIC) + set_bonus_this_step
            current_step_reward += -(market_value_after_action * MARKET_VALUE_PENALTY_FACTOR)
            current_step_reward += BASE_REWARD_ACTION_SUCCESS
        else:
            current_step_reward = (points_from_sale_this_step * REWARD_SCALING_FACTOR_SELL_POINTS) + \
                                  (delta_hvc * REWARD_FACTOR_DELTA_HAND_HEURISTIC) + \
                                  set_bonus_this_step + \
                                  (-(market_value_after_action * MARKET_VALUE_PENALTY_FACTOR)) + \
                                  action_base_success_bonus_this_step

        current_step_reward -= ACTION_COST

        if self.game.token_bank.count_depleted_stacks() >= DEPLETED_STACKS_END_CONDITION or self.game.deck.is_empty():
            term = True
        if not term and self.current_step >= self.max_steps:
            trunc = True

        if term or trunc:
            final_score = 0

            if self.current_step > 0:
                inf["freq_action_0_camels"] = self.episode_action_counts[0] / self.current_step
                inf["freq_action_1_sell"] = self.episode_action_counts[1] / self.current_step
                inf["freq_action_2_take_exchange"] = self.episode_action_counts[2] / self.current_step
            else:
                inf["freq_action_0_camels"] = 0.0
                inf["freq_action_1_sell"] = 0.0
                inf["freq_action_2_take_exchange"] = 0.0

            if term:
                final_score, winner = self._calculate_final_scores_and_winner()
                inf["final_scores"] = final_score
                inf["winner"] = f"P{winner + 1}" if winner is not None else "Tie"
                if winner is not None:
                    current_step_reward += TERMINAL_WIN_BONUS if winner == self.current_player_idx \
                        else TERMINAL_LOSS_PENALTY

                    score_difference = final_score[self.current_player_idx] - final_score[1 - self.current_player_idx]
                    current_step_reward += score_difference * SCORE_DIFFERENCE_REWARD_FACTOR
            elif trunc:
                final_score, winner = self._calculate_final_scores_and_winner()
                inf["final_scores_on_truncate"] = final_score
                inf["winner_on_truncate"] = f"P{winner + 1}" if winner is not None else "Tie"
                score_difference = final_score[self.current_player_idx] - final_score[1 - self.current_player_idx]
                current_step_reward += score_difference * SCORE_DIFFERENCE_REWARD_FACTOR

            inf["P1_Final_Score"] = float(final_score[0])
            inf["P2_Final_Score"] = float(final_score[1])

        rew = current_step_reward

        if not term and not trunc:
            self.current_player_idx = 1 - self.current_player_idx
            self.current_player = self._get_current_player()

        return self._get_obs(), rew, term, trunc, inf

    def _calculate_final_scores_and_winner(self) -> Tuple[List[int], Optional[int]]:
        p1, p2 = self.players[0], self.players[1]

        camel_bonus1, camel_bonus2 = 0, 0
        if p1.herd > p2.herd:
            camel_bonus1 = 5
        elif p2.herd > p1.herd:
            camel_bonus2 = 5

        s1 = p1.calculate_final_score(camel_bonus1)
        s2 = p2.calculate_final_score(camel_bonus2)
        final_score = [s1, s2]
        winner = None
        if s1 > s2:
            winner = 0
        elif s2 > s1:
            winner = 1

        else:  # Deal with tie
            if len(p1.bonus_tokens) > len(p2.bonus_tokens):
                winner = 0
            elif len(p2.bonus_tokens) > len(p1.bonus_tokens):
                winner = 1
            else:
                if len(p1.tokens) > len(p2.tokens):
                    winner = 0
                elif len(p2.tokens) > len(p1.tokens):
                    winner = 1
        return final_score, winner

    def _get_obs(self) -> Dict:
        p = self.current_player
        opp = self._get_opponent()
        kov = p.get_known_opp_cards_vector()
        hn = p.herd / MAX_CAMELS_PLAYER_CAN_HAVE if MAX_CAMELS_PLAYER_CAN_HAVE > 0 else 0.0
        hand_good_fullness = np.zeros(NUM_GOOD_TYPES, dtype=np.float32)
        hand_fullness = np.zeros(HAND_LIMIT + 1, dtype=np.float32)
        hand_fullness[:len(p.hand)] = 1.0
        for i_good_idx, good_type_val in GOOD_IDX_TO_CARD.items():
            hand_good_fullness[i_good_idx] = p.count_good(good_type_val) / HAND_LIMIT if HAND_LIMIT > 0 else 0.0
        cim = np.zeros(MARKET_SIZE, dtype=np.float32)
        camels_count = sum([1 for c in self.game.market.market_cards if c == CardType.CAMEL])
        if camels_count > 0:
            cim[:camels_count-1] = 1.0

        return {"market": self.game.market.get_market_matrix().astype(np.float32),
                "camels_in_market": cim,
                "hand": p.get_hand_matrix().astype(np.float32),
                "score": np.array([p.total_score / 150.0], dtype=np.float32),
                "good_counts_in_hand": np.array(list(p.good_counts_in_hand()), dtype=np.float32),
                "goods_can_sell": np.array(list(p.goods_can_sell()), dtype=np.float32),
                "herd_count": np.array([hn], dtype=np.float32),
                "token_fullness": self.game.token_bank.get_token_fullness().astype(np.float32),
                "empty_token_stacks": np.array([self.game.token_bank.count_depleted_stacks() /
                                                DEPLETED_STACKS_END_CONDITION], dtype=np.float32),
                "deck_pct": self.game.deck.get_deck_percentages().astype(np.float32),
                "known_opp_cards": kov,
                "hand_size": hand_fullness,
                "hand_good_fullness": hand_good_fullness,
                "opp_hand_size": np.array([opp.hand_size() / HAND_LIMIT if HAND_LIMIT > 0 else 0.0], dtype=np.float32),
                "opp_herd_size": np.array(
                    [opp.herd / MAX_CAMELS_PLAYER_CAN_HAVE if MAX_CAMELS_PLAYER_CAN_HAVE > 0 else 0.0],
                    dtype=np.float32),
                "opp_score": np.array([opp.total_score / 150.0], dtype=np.float32),
                "action_mask": self._get_action_mask(),
                "take_mask": self._get_take_mask()}

    def _get_action_mask(self) -> np.ndarray:
        m = np.zeros(3, dtype=np.int8)
        p = self.current_player
        mc = self.game.market.get_cards()

        if CardType.CAMEL in mc:
            m[0] = 1

        if any(p.can_sell(GOOD_IDX_TO_CARD[i]) for i in GOOD_IDX_TO_CARD):
            m[1] = 1

        market_non_camels = [card for card in mc if card != CardType.CAMEL]
        if market_non_camels:
            if p.hand_size() < HAND_LIMIT:
                m[2] = 1
            elif len(market_non_camels) >= 2 and (p.hand_size() + p.herd >= 2):
                m[2] = 1
        return m

    def _get_take_mask(self) -> np.ndarray:
        take_mask = np.zeros(MARKET_SIZE, dtype=np.int8)
        market_cards = self.game.market.get_cards()
        for i, card in enumerate(market_cards):
            if card != CardType.CAMEL:
                take_mask[i] = 1
        return take_mask

    def _get_info(self) -> Dict:
        p = self.current_player
        o = self._get_opponent()
        info_dict = {"current_player_name": p.name, "current_player_idx": self.current_player_idx,
                     "current_step": self.current_step, "deck_size": len(self.game.deck),
                     f"{p.name}_score_est": sum(t[1] for t in p.tokens) + sum(p.bonus_tokens),
                     f"{o.name}_score_est": sum(t[1] for t in o.tokens) + sum(o.bonus_tokens),
                     f"{p.name}_known_opp_goods_count": {g.value: c for g, c in p.known_opp_cards.items() if c > 0},
                     "market_cards_list": [c.value for c in self.game.market.market_cards]}
        if hasattr(self, '_consecutive_invalid_attempts') and self._consecutive_invalid_attempts > 0:
            info_dict["consecutive_invalid_attempts"] = self._consecutive_invalid_attempts
        return info_dict

    def render(self, mode="human"):
        if mode == "human":
            obs = self._get_obs()
            p = self.current_player
            print("\n" + "=" * 50)
            print(f"--- Step: {self.current_step}, Player to Move: {p.name} (P{self.current_player_idx + 1}) ---")
            print("--- Market ---")
            market_str = []
            [market_str.append(f"  [{i}]: {card.value if card else 'Empty'}") for i, card in
             enumerate(self.game.market.get_cards())]
            print("\n".join(market_str) if market_str else "  Market is empty.")
            print("\n--- Players ---")
            for pl_idx, pl_obj in enumerate(self.players):
                is_current = "(Current)" if pl_obj == p else ""
                print(f"  {pl_obj.name} {is_current}:")
                hand_str = []
                [hand_str.append(f"    [{i}]: {card_in_hand.value}") for i, card_in_hand in enumerate(pl_obj.hand)]
                print("\n".join(hand_str) if hand_str else "    Hand is empty.")
                print(f"    Herd: {pl_obj.herd}")
                known_opp_str = {g.value: count for g, count in pl_obj.known_opp_cards.items() if count > 0}
                print(
                    f"    Known Opponent Goods (P{1 - pl_idx + 1}'s hand): {known_opp_str if known_opp_str else 'None'}")
                print(f"    Score (Tokens): {sum(t[1] for t in pl_obj.tokens) + sum(pl_obj.bonus_tokens)}")
            print("\n--- Game Info ---")
            print(f"Deck Size: {len(self.game.deck)}")
            print(f"Action Mask (main actions for {p.name}): {obs['action_mask']}")
            current_heuristic_val, set_bonus_potential = p.get_heuristic_hand_value(self.game.token_bank,
                                                                                    calculate_set_bonus_for_reward=False)
            print(f"Current Heuristic Hand Value for {p.name}: {current_heuristic_val:.2f}")
            if hasattr(self, '_consecutive_invalid_attempts') and self._consecutive_invalid_attempts > 0:
                print(f"Consecutive invalid moves by {p.name}: {self._consecutive_invalid_attempts}")
            print("=" * 50 + "\n")

    def close(self):
        pass


if __name__ == "__main__":
    env = JaipurEnv(seed=123)
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)

    # Example action: Take all camels
    action = {"main_action_type": 0}
    obs, reward, terminated, truncated, info = env.step(action)
    print("After Taking Camels:", obs, reward, terminated, truncated, info)

    # Example action: Sell a good
    action = {"main_action_type": 1, "sell_good_type_index": 0}
    obs, reward, terminated, truncated, info = env.step(action)
    print("After Selling Good:", obs, reward, terminated, truncated, info)

    # Example action: Take/Exchange
    action = {
        "main_action_type": 2,
        "exchange_market_take_mask": np.array([1, 0, 0, 0, 0], dtype=np.int8),  # Take first market card
        "exchange_hand_give_indices": np.array([0, 0, 0, 0, 0,0,0], dtype=np.int8)  # Give no hand cards
    }
    obs, reward, terminated, truncated, info = env.step(action)
    print("After Take/Exchange:", obs, reward, terminated, truncated, info)

    # Example action: Take multiple cards
    action = {
        "main_action_type": 2,
        "exchange_market_take_mask": np.array([1, 1, 0, 0, 0], dtype=np.int8),  # Take first two market cards
        "exchange_hand_give_indices": np.array([1, 0, 0, 0, 0,0,0], dtype=np.int8)  # Give one hand card
    }
    obs, reward, terminated, truncated, info = env.step(action)
    print("After Multiple Take/Exchange:", obs, reward, terminated, truncated, info)
