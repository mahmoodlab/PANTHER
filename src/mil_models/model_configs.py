from dataclasses import dataclass, asdict
from typing import Optional, Union, Callable
import logging
import json
import os
logger = logging.getLogger(__name__)


@dataclass
class PretrainedConfig:
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.
        Args:
            json_file_path: Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        config_dict = {k: v for k, v in asdict(self).items()}
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(
                config_dict, indent=2, sort_keys=False) + "\n")

    @classmethod
    def from_pretrained(cls, config_path: Union[str, os.PathLike], update_dict={}):
        config_dict = json.load(open(config_path))
        for key in update_dict:
            if key in config_dict:
                config_dict[key] = update_dict[key]
        config = cls(**config_dict)
        return config

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.
        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
        """
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, "config.json")

        self.to_json_file(output_config_file)
        logger.info(f"Configuration saved in {output_config_file}")

@dataclass
class ABMILConfig(PretrainedConfig):
    gate: bool = True
    in_dim: int = 768
    n_classes: int = 2
    embed_dim: int = 512
    attn_dim: int = 384
    n_fc_layers: int = 1
    dropout: float = 0.25

@dataclass
class OTConfig(PretrainedConfig):
    in_dim: int = 768
    n_classes: int = 2
    n_filters: int = 2048
    len_motifs: int = 1
    subsamplings: int = 1
    kernel_args: int = 0.4
    weight_decay: float = 0.0001
    ot_eps: float = 3.0
    heads: int = 1
    out_size: int = 3
    out_type: str = 'param_cat'
    max_iter: int = 100
    distance: str = 'euclidean'
    fit_bias: bool = False
    alternating: bool = False
    load_proto: bool = True
    proto_path: str = '.'
    fix_proto: bool = True


@dataclass
class PANTHERConfig(PretrainedConfig):
    in_dim: int = 768
    n_classes: int = 2
    heads: int = 1
    em_iter: int = 3
    tau: float = 0.001
    embed_dim: int = 512
    ot_eps: int = 0.1
    n_fc_layers: int = 1
    dropout: float = 0.
    out_type: str = 'param_cat'
    out_size: int = 3
    load_proto: bool = True
    proto_path: str = '.'
    fix_proto: bool = True


@dataclass
class ProtoCountConfig(PretrainedConfig):
    in_dim: int = 768
    n_classes: int = 2
    out_size: int = 3
    load_proto: bool = True
    proto_path: str = '.'
    fix_proto: bool = True

@dataclass
class H2TConfig(PretrainedConfig):
    in_dim: int = 768
    n_classes: int = 2
    out_size: int = 3
    load_proto: bool = True
    proto_path: str = '.'
    fix_proto: bool = True

@dataclass
class LinearEmbConfig(PretrainedConfig):
    in_dim: int = 768
    n_classes: int = 2


@dataclass
class IndivMLPEmbConfig(PretrainedConfig):
    in_dim: int = 768
    n_classes: int = 2
    embed_dim: int = 128
    n_fc_layers: int = 2
    dropout: float = 0.25
    proto_model_type: str = 'DIEM'
    p: int = 32
    out_type: str = 'param_cat'

@dataclass
class IndivMLPEmbConfig_Shared(PretrainedConfig):
    in_dim: int = 129
    n_classes: int = 4
    shared_embed_dim: int = 64
    indiv_embed_dim: int = 32
    postcat_embed_dim: int = 512
    shared_mlp: bool = True
    indiv_mlps: bool = False
    postcat_mlp: bool = False
    n_fc_layers: int = 1
    shared_dropout: float = 0.25
    indiv_dropout: float = 0.25
    postcat_dropout: float = 0.25
    p: int = 32

@dataclass
class IndivMLPEmbConfig_Indiv(PretrainedConfig):
    in_dim: int = 129
    n_classes: int = 4
    shared_embed_dim: int = 64
    indiv_embed_dim: int = 32
    postcat_embed_dim: int = 512
    shared_mlp: bool = False
    indiv_mlps: bool = True
    postcat_mlp: bool = False
    n_fc_layers: int = 1
    shared_dropout: float = 0.25
    indiv_dropout: float = 0.25
    postcat_dropout: float = 0.25
    p: int = 32

@dataclass
class IndivMLPEmbConfig_SharedPost(PretrainedConfig):
    in_dim: int = 129
    n_classes: int = 4
    shared_embed_dim: int = 64
    indiv_embed_dim: int = 32
    postcat_embed_dim: int = 512
    shared_mlp: bool = True
    indiv_mlps: bool = False
    postcat_mlp: bool = True
    n_fc_layers: int = 1
    shared_dropout: float = 0.25
    indiv_dropout: float = 0.25
    postcat_dropout: float = 0.25
    p: int = 32

@dataclass
class IndivMLPEmbConfig_IndivPost(PretrainedConfig):
    in_dim: int = 2049
    n_classes: int = 4
    shared_embed_dim: int = 256
    indiv_embed_dim: int = 128
    postcat_embed_dim: int = 1024
    shared_mlp: bool = False
    indiv_mlps: bool = True
    postcat_mlp: bool = True
    n_fc_layers: int = 1
    shared_dropout: float = 0.25
    indiv_dropout: float = 0.25
    postcat_dropout: float = 0.25
    p: int = 16
    use_snn: bool = False

@dataclass
class IndivMLPEmbConfig_SharedIndiv(PretrainedConfig):
    in_dim: int = 2049
    n_classes: int = 4
    shared_embed_dim: int = 256
    indiv_embed_dim: int = 128
    postcat_embed_dim: int = 1024
    shared_mlp: bool = True
    indiv_mlps: bool = True
    postcat_mlp: bool = False
    n_fc_layers: int = 1
    shared_dropout: float = 0.25
    indiv_dropout: float = 0.25
    postcat_dropout: float = 0.25
    p: int = 16
    use_snn: bool = False

@dataclass
class IndivMLPEmbConfig_SharedIndivPost(PretrainedConfig):
    in_dim: int = 129
    n_classes: int = 4
    shared_embed_dim: int = 64
    indiv_embed_dim: int = 32
    postcat_embed_dim: int = 512
    shared_mlp: bool = True
    indiv_mlps: bool = True
    postcat_mlp: bool = True
    n_fc_layers: int = 1
    shared_dropout: float = 0.25
    indiv_dropout: float = 0.25
    postcat_dropout: float = 0.25
    p: int = 32
