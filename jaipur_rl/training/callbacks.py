from stable_baselines3.common.callbacks import BaseCallback

class InfoCallback(BaseCallback):
    """
    A custom callback that logs episode information from the 'info' dict.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Check if any environments have finished an episode
        # self.locals['dones'] is a boolean array, True for each env that finished
        for i, done in enumerate(self.locals['dones']):
            if done:
                # For a vectorized env, the infos are in a list
                info = self.locals['infos'][i]

                # --- Log Final Game Stats ---
                # Safely get final scores, checking for both terminal and truncated keys
                final_scores = info.get("final_scores") or info.get("final_scores_on_truncate")
                if final_scores and isinstance(final_scores, list) and len(final_scores) == 2:
                    self.logger.record("episode_stats/player1_final_score", final_scores[0])
                    self.logger.record("episode_stats/player2_final_score", final_scores[1])
                    self.logger.record("episode_stats/score_difference", abs(final_scores[0] - final_scores[1]))

                # --- Log Action Frequencies for the Episode ---
                freq_camels = info.get("freq_action_0_camels")
                if freq_camels is not None:
                    self.logger.record("action_freq/camels", freq_camels)

                freq_sell = info.get("freq_action_1_sell")
                if freq_sell is not None:
                    self.logger.record("action_freq/sell", freq_sell)

                freq_exchange = info.get("freq_action_2_take_exchange")
                if freq_exchange is not None:
                    self.logger.record("action_freq/take_exchange", freq_exchange)

        return True