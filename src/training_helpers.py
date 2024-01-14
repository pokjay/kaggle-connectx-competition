import random

import numpy as np
from functools import partial

from kaggle_environments import evaluate
from stable_baselines3.common.callbacks import BaseCallback


def agent_from_model(model, obs, config):
    # Use the best model to select a column
    col, _ = model.predict(np.array(obs["board"]).reshape(1, 6, 7))
    # Check if selected column is valid
    is_valid = obs["board"][int(col)] == 0
    # If not valid, select random move.
    if is_valid:
        return int(col)
    else:
        return random.choice(
            [col for col in range(config.columns) if obs.board[int(col)] == 0]
        )


class EvaluateAgentSuccessCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, n_rounds=100, other_agents=None):
        super().__init__(verbose)

        self.n_rounds = n_rounds
        self.other_agents = ["random"] if other_agents is None else other_agents

    def _on_step(self) -> bool:
        our_agent = partial(agent_from_model, self.model)

        for other_agent in self.other_agents:
            perc_win, _, num_invalid_plays, _ = self._get_win_percentages(
                our_agent, other_agent
            )
            self.logger.record(f"win_perc_vs_{other_agent}", perc_win)
            self.logger.record(f"num_invalid_plays_vs_{other_agent}", num_invalid_plays)

        return True

    def _get_win_percentages(self, agent1, agent2):
        # Use default Connect Four setup
        config = {"rows": 6, "columns": 7, "inarow": 4}
        # Agent 1 goes first (roughly) half the time
        outcomes = evaluate(
            "connectx", [agent1, agent2], config, [], self.n_rounds // 2
        )
        # Agent 2 goes first (roughly) half the time
        outcomes += [
            [b, a]
            for [a, b] in evaluate(
                "connectx",
                [agent2, agent1],
                config,
                [],
                self.n_rounds - self.n_rounds // 2,
            )
        ]

        agent1_win_perc = np.round(outcomes.count([1, -1]) / len(outcomes), 2)
        agent2_win_perc = np.round(outcomes.count([-1, 1]) / len(outcomes), 2)
        agent1_invalid_plays = outcomes.count([None, 0])
        agent2_invalid_plays = outcomes.count([0, None])

        return (
            agent1_win_perc,
            agent2_win_perc,
            agent1_invalid_plays,
            agent2_invalid_plays,
        )
