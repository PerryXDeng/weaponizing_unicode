"""
file: database.py
language:python 3

extracts information from the unicode consortium web database
"""

__DB = "https://www.unicode.org/Public/UCD/latest/ucd/"

from typing import *
import urllib.request

# inclusive decimal range of a unicode subset
__UnicodeRange = Tuple[int, int]
# matches subsets names with codepoint ranges
__UnicodeBlocks = Dict[str, __UnicodeRange]
# maps potentially implemented characters to unicode
# blocks with implemented characters
__UnicodeMap = List[str]

UNDEFINED_BLOCK = "undefined" # for indicating that a character is not defined


def download_and_parse_unicode_clusters() -> list:
  """
  https://www.unicode.org/Public/security/latest/confusables.txt
  :return: {lists of lists of unicode codepoints ints corresponding to clusters}
  """
  raise NotImplementedError


def _is_character_block(block_name: str) -> bool:
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


def map_blocks() -> (__UnicodeBlocks, __UnicodeMap, int):
  """
  In some browsers and email clients, all the characters in the domain name
  needs to be of the same subset, or "block," of unicode, in order to be
  displayed in unicode form.

  This helper function determines whether a unicode domain name will be
  displayed in "xe--*" ascii encoding format, or its unicode form.

  This function uses the definition of subset blocks specified by the latest
  unicode standard.

  :return: tuple of UnicodeSets, UnicodeMap, and total # of chars
  """
  blocks = {}
  block_map = []
  with urllib.request.urlopen(__DB + "Blocks.txt") as response:
    lines = response.read().decode('utf-8').split("\n")
    for line in lines:
      if len(line) > 0 and line[0] != '\n' and line[0] != '#':
        line = line.strip()
        (hex_range, block_name) = line.split("; ")
        if _is_character_block(block_name):
          (start_hex, end_hex) = hex_range.split("..")
          start = int(start_hex, 16)
          end = int(end_hex, 16)
          blocks[block_name] = (start, end)
          if len(block_map) < end + 1:
            for i in range(len(block_map), end + 1):
              block_map.append(UNDEFINED_BLOCK)
          for i in range(start, end + 1):
            block_map[i] = block_name
  # as of unicode 12
  # block_map produces an array for the first 900k unicode code points
  # around 140k of which belong to blocks with defined code points
  n = _prune_block_map(block_map)
  return blocks, block_map, n


def _is_code_range(description:str) -> int:
  """
  determines whether an entry is a code point or the start/end of a range
  :param description: entry description, second field in line
  :return: -1 if it's a code point, 0 if it's first in a range, 1 if it's last
  """
  if len(description) > 4:
    if description[-4:] == "rst>": # first in range, inclusive
      return 0
    if description[-4:] == "ast>": # last in range, inclusive
      return 1
  return -1


def _prune_block_map(block_map:__UnicodeMap):
  """
  goes through the block map and "un-define" the blocks for characters
  that are not actually implemented
  :param block_map: unicode map of characters and blocks
  :return: total number of implemented characters
  """
  n = 0
  implemented = [False] * len(block_map)
  with urllib.request.urlopen(__DB + "UnicodeData.txt") as response:
    lines = response.read().decode('utf-8').split("\n")
    i = 0
    while i < len(lines):
      line = lines[i].strip()
      fields = line.split(";")
      if len(line) > 0 and fields[1] != "<control>"\
              and _is_character_block(fields[1]):
        index = int(fields[0], 16)
        retval = _is_code_range(fields[1])
        if retval == -1:
          implemented[index] = True
          n += 1
        elif retval == 0:
          i += 1
          line = lines[i].strip()
          fields = line.split(";")
          end = int(fields[0], 16)
          for k in range(index, end + 1):
            implemented[k] = True
            n += 1
      i += 1
  for i in range(len(implemented)):
    if not implemented[i]:
      block_map[i] = UNDEFINED_BLOCK
  # 137929 as of unicode 12
  return n
