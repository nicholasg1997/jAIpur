from jaipur_rl.envs.jaipur_env import JaipurEnv
from sb3_plus import MultiOutputPPO
from pathlib import Path

env = JaipurEnv()
model_base_path = Path(__file__).resolve().parents[1] / "models"
model = MultiOutputPPO.load(model_base_path / "Jaipur_25.0M_scheduler_2025-06-20_18-23-16", env=None)
trained_agent = model

p1_scores = []
p2_scores = []
p1_wins = 0
p2_wins = 0
games = 100
games_finished = 0

for i in range(games):
    obs, _ = env.reset()
    print(f"Game {i+1} Start")
    for j in range(200):
        action_batch, _ = trained_agent.predict(obs, deterministic=False)

        obs, reward, done, truncated, info = env.step(action_batch)

        if done or truncated:
            games_finished += 1
            p1_score = info.get('P1_Final_Score')
            p1_scores.append(p1_score)
            p2_score = info.get('P2_Final_Score')
            p2_scores.append(p2_score)
            if info['winner'] == 'P1':
                p1_wins += 1
            elif info['winner'] == 'P2':
                p2_wins += 1
            break

print(f"Player 1 Score: {sum(p1_scores)/games_finished}, Player 2 Score: {sum(p2_scores)/games_finished}")
print(f"Player 1 Wins: {p1_wins}, Player 2 Wins: {p2_wins}")
print(f"Games Finished: {games_finished}/{games}")

