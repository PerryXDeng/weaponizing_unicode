"""
file: fonts_info.py
language:python 3

extracts meta information from .ttf font files
"""
import numpy as np
import glob
import os
import pickle
from fontTools.ttLib import TTFont
from unicode_info import database as db


_FONT_ROOTDIR = ""
_NOTO = "noto_fonts/regular/*.ttf"
_OS = "os_fonts/*/*.ttf"
_WIN = "os_fonts/win_fonts/*.ttf"
_MAC = "os_fonts/mac_fonts/*.ttf"
_ALL = "*/*/*.ttf"
_TEST = "NotoSansCJKjp-Regular.ttf"

_FONT_DIRS = [os.path.join(_FONT_ROOTDIR, _NOTO), os.path.join(_FONT_ROOTDIR, _MAC), os.path.join(_FONT_ROOTDIR, _WIN)]

def map_implemented_characters_indices(fontpath:str, covered:np.array):
  """
  gets the indices of implemented characters in a font and flips the indices to true
  :param fontpath: file path to font
  :param covered: array of booleans to be modified
  """
  ttf = TTFont(fontpath)
  largest_table = None
  # picks the largest sub table
  # https://docs.microsoft.com/en-us/typography/opentype/spec/cmap
  for table in ttf["cmap"].tables:
    if largest_table is None:
      largest_table = table.cmap
    else:
      if len(table.cmap) > len(largest_table):
        largest_table = table.cmap
  boundary = covered.shape[0]
  indices = [key for key in list(largest_table.keys()) if not key > boundary] # emits private uses
  covered[indices] = True


def count_implemented_characters(fontdir:str) -> (int, int):
  """
  gets the coverage of characters by font files in a directory
  :param fontdir: directory of fonts
  :return: number of covered characters and number of total characters
  """
  ttf_filepaths = glob.glob(fontdir, recursive=True)
  print(str(len(ttf_filepaths)) + " fonts found")
  blocks, indices, n = db.map_blocks()
  covered = np.full(len(indices), False)
  for filepath in ttf_filepaths:
      map_implemented_characters_indices(filepath, covered)

  print("\nCoverage by Blocks:")
  for block in blocks.keys():
    start, finish = blocks[block]
    total = finish - start + 1
    if not finish + 1 < len(indices):
      finish = len(indices) - 2
    coverage = np.sum(covered[start:finish + 1])
    print(block + " " + str(coverage) + " " + str(total))
  coverage = np.sum(covered)
  return coverage, n


def map_character_to_single_fontpath():
  """
  maps all characters to paths of fonts that support them, if any
  :return: {codepoint:fontpath}
  """

  _, codepoint_block, _ = db.map_blocks()
  font_paths = {}

  for directory in _FONT_DIRS:
    ttf_filepaths = glob.glob(directory, recursive=True)
    for filepath in ttf_filepaths:
      ttf = TTFont(filepath)
      largest_table = None
      # picks the largest sub table
      # https://docs.microsoft.com/en-us/typography/opentype/spec/cmap
      for table in ttf["cmap"].tables:
        if largest_table is None:
          largest_table = table.cmap
        else:
          if len(table.cmap) > len(largest_table):
            largest_table = table.cmap
      boundary = len(codepoint_block)
      for index in list(largest_table.keys()):
        if index < boundary and not font_paths[index]:
          font_paths[index] = filepath
  return font_paths


def map_character_to_multiple_fontpath():
  """
  maps all characters to paths of fonts that support them, if any
  :return: {codepoint:[fontpaths]}
  """

  _, codepoint_block, _ = db.map_blocks()
  font_paths = {}

  for directory in _FONT_DIRS:
    ttf_filepaths = glob.glob(directory, recursive=True)
    for filepath in ttf_filepaths:
      ttf = TTFont(filepath)
      largest_table = None
      # picks the largest sub table
      # https://docs.microsoft.com/en-us/typography/opentype/spec/cmap
      for table in ttf["cmap"].tables:
        if largest_table is None:
          largest_table = table.cmap
        else:
          if len(table.cmap) > len(largest_table):
            largest_table = table.cmap
      boundary = len(codepoint_block)
      for index in list(largest_table.keys()):
        if index < boundary:
          if index in font_paths:
            font_paths[index].append(filepath)
          else:
            font_paths[index] = [filepath]
  return font_paths


def serialize_keys_fontpaths_mapping():
  with open('multifont_mapping.pkl', 'wb+') as f:
    pickle.dump(map_character_to_multiple_fontpath(), f)


def main():
  serialize_keys_fontpaths_mapping()


if __name__ == "__main__":
  main()
