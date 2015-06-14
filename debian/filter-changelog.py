#!/usr/bin/env python

import sys
import debian.changelog

filename = sys.argv[1]
distribution = sys.argv[2]
extraversion = sys.argv[3]

fp = open(filename)
data = fp.read()

changelog = debian.changelog.Changelog(data)

for change in changelog:
	change.version = str(change.version) + distribution + extraversion
	change.distributions = distribution
	print change

