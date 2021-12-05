#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='Extract aligned plain text from IWSLT XML data.')
parser.add_argument('-i', '--input', type=str, required=True, help='path to the input XML file')
parser.add_argument('-o', '--output', type=str, required=True, help='name of the output plain text file')

args = parser.parse_args()

tree = ET.parse(args.input)
root = tree.getroot()

with open(args.output, mode='w', encoding='utf-8') as out_file:
  for segment in root.iter('seg'):
    out_file.write('%s\n' % segment.text.strip())
