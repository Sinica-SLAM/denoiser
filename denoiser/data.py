# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez and adiyoss

import json
import logging
import os
import re

from .audio import Audioset, NoisyAudioset, CHiME3NoisyAudioset

logger = logging.getLogger(__name__)


def match_dns(noisy, clean):
    """match_dns.
    Match noisy and clean DNS dataset filenames.

    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    """
    logger.debug("Matching noisy and clean for dns dataset")
    noisydict = {}
    extra_noisy = []
    for path, size in noisy:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            # maybe we are mixing some other dataset in
            extra_noisy.append((path, size))
        else:
            noisydict[match.group(1)] = (path, size)
    noisy[:] = []
    extra_clean = []
    copied = list(clean)
    clean[:] = []
    for path, size in copied:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            extra_clean.append((path, size))
        else:
            noisy.append(noisydict[match.group(1)])
            clean.append((path, size))
    extra_noisy.sort()
    extra_clean.sort()
    clean += extra_clean
    noisy += extra_noisy


def match_files(noisy, clean, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "dns":
        # dns dataset filenames don't match when sorted, we have to manually match them
        match_dns(noisy, clean)
    elif matching == "sort":
        noisy.sort()
        clean.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")


class NoisyCleanSet:
    def __init__(self, json_dir, matching="sort", length=None, stride=None,
                 pad=True, sample_rate=None):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)

        match_files(noisy, clean, matching)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index], self.clean_set[index]

    def __len__(self):
        return len(self.noisy_set)


class MNoisyNoisySet:
    def __init__(self, json_dir, length=None, stride=None,
                 pad=True, sample_rate=None):
        """__init__.

        :param json_dir: directory containing both noise.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        noise_json = os.path.join(json_dir, 'noise.json')
        noisy_json = os.path.join(json_dir, 'noisy.json')
        with open(noise_json, 'r') as f:
            noise = json.load(f)
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)

        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        self.noisy_set = Audioset(noisy, **kw)
        self.m_noisy_set = NoisyAudioset(noisy, noise, **kw)

        assert len(self.noisy_set) == len(self.m_noisy_set)

    def __getitem__(self, index):
        return self.m_noisy_set[index], self.noisy_set[index]

    def __len__(self):
        return len(self.m_noisy_set)


class CHiME3MNoisyNoisySet:
    def __init__(self, json_dir, length=None, stride=None,
                 pad=True, sample_rate=None, chime3_background_path=None):
        """__init__.

        :param json_dir: directory containing both noise.json and noisy.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        noise_json = os.path.join(json_dir, 'noise.json')
        noisy_json = os.path.join(json_dir, 'noisy.json')
        with open(noise_json, 'r') as f:
            noise = json.load(f)
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)

        kw = {'length': length, 'stride': stride, 'pad': pad,
              'sample_rate': sample_rate, 'chime3_background_path': chime3_background_path}
        self.noisy_set = Audioset(noisy, **kw)
        self.m_noisy_set = CHiME3NoisyAudioset(noisy, noise, **kw)

        assert len(self.noisy_set) == len(self.m_noisy_set)

    def __getitem__(self, index):
        return self.m_noisy_set[index], self.noisy_set[index]

    def __len__(self):
        return len(self.m_noisy_set)
