from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

# Exercise 5 + 6: instantiate and print out model using defaults config

@hydra.main(version_base = None, config_path = "configs", config_name = "defaults")
def run(cfg: DictConfig):
    model = instantiate(cfg.model)
    print(model)

if __name__ == "__main__":
    run()