from ruamel.yaml import YAML

class ParamsConfig:

    yaml = YAML(typ="safe")  

    def __init__(self) -> None:
        pass    

    @staticmethod
    def get_params_config(params_path: str='params.yaml'):
        return ParamsConfig.yaml.load(open(params_path, encoding="utf-8"))
    
    @staticmethod
    def get_architecture_params(latent_dim, use_maxpool, base_dir='.'):
        root_dir = f'{base_dir}/data/arch_configs'

        params_path = f'latent{latent_dim}'
        params_path += '_maxpool.yaml' if use_maxpool else '.yaml'

        return ParamsConfig.yaml.load(open(f"{root_dir}/{params_path}", encoding="utf-8"))