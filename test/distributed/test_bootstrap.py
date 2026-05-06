"""
Test distributed bootstrap functionality.
"""
import unittest
import os
from tinygrad.distributed import init_distributed, DistributedConfig, get_world_info, get_all_capabilities, reset_global_config


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        """Clean up environment variables before each test."""
        reset_global_config()
        
        env_vars_to_reset = [
            "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
            "SLURM_PROCID", "SLURM_NTASKS", "SLURM_JOB_GPUS", "SLURM_JOB_ID"
        ]
        for var in env_vars_to_reset:
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up after each test."""
        reset_global_config()
    
    def test_get_world_info_single_node(self):
        """Test world info in single-node (non-distributed) mode."""
        info = get_world_info()
        self.assertEqual(info["rank"], 0)
        self.assertEqual(info["world_size"], 1)
        self.assertEqual(info["local_rank"], 0)
        self.assertEqual(info["num_nodes"], 1)
        self.assertTrue("node_name" in info)
    
    def test_get_all_capabilities_single_node(self):
        """Test all capabilities in single-node mode."""
        caps = get_all_capabilities()
        self.assertIsInstance(caps, list)
        self.assertEqual(len(caps), 1)
        self.assertTrue("rank" in caps[0])
        self.assertTrue("gpus" in caps[0])
    
    def test_init_distributed_single_node(self):
        """Test init_distributed in single-node mode."""
        config = init_distributed()
        self.assertIsNone(config)  # Should return None when no distributed backend
    
    def test_env_bootstrap(self):
        """Test bootstrap using environment variables (stub)."""
        # Save original env vars
        original_env = {
            "RANK": os.getenv("RANK"),
            "WORLD_SIZE": os.getenv("WORLD_SIZE"),
            "LOCAL_RANK": os.getenv("LOCAL_RANK"),
            "MASTER_ADDR": os.getenv("MASTER_ADDR"),
            "MASTER_PORT": os.getenv("MASTER_PORT"),
        }
        
        try:
            # Set env vars
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "2"
            os.environ["LOCAL_RANK"] = "0"
            
            # This should return a DistributedConfig with env backend
            config = init_distributed(backend="env")
            
            # Verify config
            self.assertIsNotNone(config)
            self.assertEqual(config.rank, 0)
            self.assertEqual(config.world_size, 2)
            self.assertEqual(config.local_rank, 0)
            self.assertEqual(config.backend, "env")
            
        finally:
            # Restore original env vars
            for k, v in original_env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]


if __name__ == "__main__":
    unittest.main()
