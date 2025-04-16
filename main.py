import dotenv
import hydra
from omegaconf import DictConfig
from src.utils.runner import run, instantiate


dotenv.load_dotenv()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    score, predictions = run(*instantiate(config))
    return score


if __name__ == "__main__":
    main()
