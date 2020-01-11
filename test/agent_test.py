import os
import sys
os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(".")

import unittest
import shutil, tempfile
import agent
import numpy as np

class TestAgentMethods(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_random_seed(self):
        for _ in range(100):
            random_seed = np.random.randint(low=0, high=10000)

            first_agent = agent.Worker(
                worker_index=0,
                global_network=None,
                num_base_stations=1,
                num_users=1,
                arena_width=100,
                random_seed=random_seed,
                max_episodes=1,
                optimizer=None,
                update_frequency=None,
                entropy_coefficient=None,
                norm_clip_value=None,
                num_checkpoints=1,
                reward_queue=None,
                save_dir=self.test_dir
            )

            second_agent = agent.Worker(
                worker_index=1,
                global_network=None,
                num_base_stations=1,
                num_users=1,
                arena_width=100,
                random_seed=random_seed,
                max_episodes=1,
                optimizer=None,
                update_frequency=None,
                entropy_coefficient=None,
                norm_clip_value=None,
                num_checkpoints=1,
                reward_queue=None,
                save_dir=self.test_dir
            )

            self.assertTrue(
                np.all(np.equal(
                    np.ravel(first_agent.env.state), 
                    np.ravel(second_agent.env.state)
                ))
            )

if __name__ == '__main__':
    unittest.main()