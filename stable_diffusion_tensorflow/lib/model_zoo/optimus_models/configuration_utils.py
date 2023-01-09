# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# copyright de rune 2023
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import os
from io import open

from .file_utils import cached_path, CONFIG_NAME

logger = logging.getLogger(__name__)


class PretrainedConfig(object):
    r"""
    base class for all configuration classes.
    handles a few parameters common to al models configuration asa well as method
    for loading/download/saving configutraion

    a configuration file can be loaded and saved to disk. loading the configuration
    file and using this file to initialize model does NOT load the model weights.
    it only affects the model's configuration.

    param:
        - finetuning_task: string, default None
            name of thetask used to fine-tune the model. this can be used when converting
            from and original (Tensorflow or PyTorch) checkpoint.

        - num_labels: integer, default 2
            number of classes to use when the model is a classification model (sequences/ token)

        - output_attentions: boolean, default False
            should the model  return attentions weights

        - output_hiddend_states: string, default False
            should the model return all hidden-state

        - torchscript: string, default False
            is the model used with torchscript
    """
    pretrained_config_achieve_map = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hiddend_states = kwargs.pop("output_hiddend_states", False)
        self.torchscript = kwargs.pop("torchscript", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})

    def save_pretrained(self, save_directory):
        """
        save a configuration object to the directory `save_directory`, so that it
        can be re-loaded using the function
        '~pytorch_transformers.PretrainedConfig.from_pretrained' class method
        """
        assert os.path.isdir(
            save_directory
        ), "saving path should be directory where the model and configuration can be saved"

        # if save usingthe predefined name, load using  `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        instantiate a :class:
        '~pytorch_transformers.PretrainedConfig' (or a derived class) from
        a pretrained model configuration

        param:
            pretrained_model_name_or_path: either:
                - a string with the `shorcut name` of a pre-trained model configuration
                to load from cache or download e.g ``bert-base-uncased``
                - a path to a `directory` containing a configuration file saved using the
                function: `~pytorch_transformers.PretrainedConfig.save_pretrained` method
                ``./my_model_directory``
                - a path or url to a saved configuration JSON `file`, e.g:
                ``./my_model_directory/configuration.json``
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        if pretrained_model_name_or_path in cls.pretrained_config_achieve_map:
            config_file = cls.pretrained_config_achieve_map[
                pretrained_model_name_or_path
            ]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path

        try:
            resolved_config_file = cached_path(
                config_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
            )
        except EnvironmentError as env_error:
            if pretrained_model_name_or_path in cls.pretrained_config_achieve_map:
                logger.error(
                    "couldn't reach server at '{}' to download pretrained model configuration file".format(
                        config_file
                    )
                )
            else:
                logger.error(
                    "model name '{}' was not found in model name list ({}). "
                    "we assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url".format(
                        pretrained_model_name_or_path,
                        ", ".join(cls.pretrained_config_achieve_map.keys()),
                        config_file,
                    )
                )
            raise env_error

        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info(
                "loading configuration file {} from cache at {}".format(
                    config_file, resolved_config_file
                )
            )

        config = cls.from_json_file(resolved_config_file)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict(
                (int(key), set(value)) for key, value in config.pruned_heads.items()
            )

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("model config %s", config)
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        config = cls(vocab_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
