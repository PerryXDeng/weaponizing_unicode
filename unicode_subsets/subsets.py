"""
file: subsets.py
language:python 3

In some browsers and email clients, all the characters in the domain name
needs to be of the same subset, or "block," of unicode, in order to be
displayed in unicode form.

This helper function determines whether a unicode domain name will be
displayed in "xe--*" ascii encoding format, or its unicode form.
This function uses the definition of subset blocks specified by Unicode 11.0
http://unicode.org/Public/11.0.0/ucd/Blocks.txt
"""

__FILEPATH = "./Blocks.txt"

from typing import *

__UnicodeRange = Tuple[int, int]  # decimal range of a unicode subset
__UnicodeSets = Dict[str, __UnicodeRange]  # matching subsets names with ranges
__UnicodeMap = List[str]  #


def _getUnicodeSubsets(path: str) -> (__UnicodeSets, __UnicodeMap):
  """
  outputs data structures that map the unicode blocks

  :param path: path to the file that contains a copy of the unicode 11 blocks
  :return: tuple of UnicodeSets and UnicodeMap
  """
  fd = open(path, "r")
  sets = {}
  index = []
  for line in fd:
    if line[0] != '\n' and line[0] != '#':
      line = line.strip()
      (hexRange, setName) = line.split("; ")
      (startHex, endHex) = hexRange.split("..")
      start = int(startHex, 16)
      end = int(endHex, 16)
      sets[setName] = (start, end)
      if len(index) < end:
        for i in range(len(index), end + 1):
          index.append("undefined")
      for i in range(start, end + 1):
        index[i] = setName
  fd.close()
  return sets, index

# (__unicodeSets, __unicodeMap) = _getUnicodeSubsets(__FILEPATH)
