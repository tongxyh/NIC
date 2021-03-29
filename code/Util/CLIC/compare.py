## -*- coding: utf-8 -*-
import sys
import metrics

a = {0:sys.argv[1]}
b = {0:sys.argv[2]}
results = metrics.evaluate(a,b)
print(results)