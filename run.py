import hydra
from omegaconf import DictConfig, OmegaConf
from manager import manager

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="run"
)
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg)) # 打印配置
    # print(cfg.model_name)
    manager(cfg)

if __name__ == "__main__":
    main()
