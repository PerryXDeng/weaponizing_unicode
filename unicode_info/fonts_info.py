"""
file: fonts_info.py
language:python 3

extracts meta information from .ttf font files
"""
import numpy as np
import glob
from fontTools.ttLib import TTFont
import unicode_info.database as db


_FONT_DIR = "/home/pxd256/Workspace/project_punyslayer/fonts/"


def implemented_characters_indices(fontpath:str) -> np.array:
  """
  gets the indices of implemented characters in a font
  :param fontpath: file path to font
  :return: array of indices
  """
  try:
    ttf = TTFont(fontpath)
  except IsADirectoryError:
    print(fontpath)
    return np.asarray([32])
  return np.asarray(list(ttf["cmap"].getBestCmap().keys()))


def count_implemented_characters(fontdir:str) -> (int, int):
  """
  gets the coverage of characters by font files in a directory
  :param fontdir: directory of fonts
  :return: number of covered characters and number of total characters
  """
  ttf_filepaths = glob.glob(_FONT_DIR, recursive=True)
  _, indices, n = db.map_blocks()
  covered = np.full(len(indices), False)
  for filepath in ttf_filepaths:
      covered[implemented_characters_indices(filepath)] = True
  coverage = np.sum(covered)
  return coverage, n


def main():
  print(count_implemented_characters(_FONT_DIR + "*/*.ttf"))


if __name__ == "__main__":
  main()
