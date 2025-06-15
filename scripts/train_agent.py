from jaipur_rl.training.trainer import train
import multiprocessing as mp
mp.set_start_method('spawn', force=True)


if __name__ == "__main__":
    train(
        steps=25_000_000,
        n_envs=8,
        init_lr=1e-4,
        min_lr=1e-6,
        init_ent_coef=0.05,
        final_ent_coef=0.005,
        ent_coef_frac=0.4,
        pi_network=[256, 128],
        vf_network=[256, 256],
        tensorboard_log="./jaipur_tensorboard/",
        log_name="Jaipur_tb",
        save_path="./models",
    )