import collections

stats = collections.Counter()

line = "abcd"
stats.update(line)
print(stats["e"])
