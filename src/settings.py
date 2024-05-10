from ruamel.yaml import YAML

class ParamsConfig:

    yaml = YAML(typ="safe")  

    def __init__(self) -> None:
        pass    

    @staticmethod
    def get_params_config(params_path: str='params.yaml'):
        return ParamsConfig.yaml.load(open(params_path, encoding="utf-8"))