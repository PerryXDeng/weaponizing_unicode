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
_NOTO = "noto_fonts/regular/*.ttf"
_OS = "os_fonts/*/*.ttf"
_WIN = "os_fonts/win_fonts/*.ttf"
_MAC = "os_fonts/mac_fonts/*.ttf"
_ALL = "*/*/*.ttf"
_TEST = "NotoSansCJKjp-Regular.ttf"

def map_implemented_characters_indices(fontpath:str, covered:np.array) -> np.array:
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


def main():
  #print(count_implemented_characters(_FONT_DIR + _TEST))
  print(count_implemented_characters(_FONT_DIR + _ALL))


if __name__ == "__main__":
  main()
