"""
Testing file
"""

import rehearsal as r
import task
ranges = task.getRanges([{'input':[0,1], 'teacher':[1]}])

r.createPsuedoItem(None, ranges=ranges)