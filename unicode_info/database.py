"""
file: database.py
language:python 3

extracts information from the unicode consortium web database
"""

__DB = "https://www.unicode.org/Public/UCD/latest/ucd/"

from typing import *
import urllib.request

__UnicodeRange = Tuple[int, int]  # inclusive decimal range of a unicode subset
__UnicodeSets = Dict[str, __UnicodeRange]  # matches subsets names with codepoint ranges
__UnicodeMap = List[str]  # maps a character to a map


def is_character_block(block_name: str) -> bool:
  """
  checks if block implements actual characters
  :param block_name: name of the block
  :return: true if are characters
  """
  keywords = ["Surrogate", "Private"]
  is_character = True
  for keyword in keywords:
    if keyword in block_name:
      is_character = False
      break
  return is_character


def map_subsets() -> (__UnicodeSets, __UnicodeMap):
  """
  In some browsers and email clients, all the characters in the domain name
  needs to be of the same subset, or "block," of unicode, in order to be
  displayed in unicode form.

  This helper function determines whether a unicode domain name will be
  displayed in "xe--*" ascii encoding format, or its unicode form.

  This function uses the definition of subset blocks specified by the latest
  unicode standard.

  :return: tuple of UnicodeSets and UnicodeMap
  """
  sets = {}
  set_map = []
  undefined_block = "undefined_block"
  with urllib.request.urlopen(__DB + "Blocks.txt") as response:
    lines = response.read().decode('utf-8').split("\n")
    for line in lines:
      if len(line) > 0 and line[0] != '\n' and line[0] != '#':
        line = line.strip()
        (hex_range, block_name) = line.split("; ")
        if is_character_block(block_name):
          (start_hex, end_hex) = hex_range.split("..")
          start = int(start_hex, 16)
          end = int(end_hex, 16)
          sets[block_name] = (start, end)
          if len(set_map) < end + 1:
            for i in range(len(set_map), end + 1):
              set_map.append(undefined_block)
          for i in range(start, end + 1):
            set_map[i] = block_name
  return sets, set_map
