#!/usr/bin/env python

import sys
import debian.changelog

filename = sys.argv[1]
distribution = sys.argv[2]

if len(sys.argv) > 2:
    extraversion = sys.argv[3]
else:
    extraversion = ""

fp = open(filename)
data = fp.read()

changelog = debian.changelog.Changelog(data)
changelog_iter = iter(changelog)

# modify the first change to point at our distribution, but leave all others alone
change = changelog_iter.next()
change.version = str(change.version) + distribution + extraversion
change.distributions = distribution
print change

for change in changelog_iter:
	print change

