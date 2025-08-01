# Given a queue with n items + embedder, they will all be processed eventually.

# Items removed from queue are either embedded WITH song+artist objects or failed gracefully.

# Given enough time, downloaded songs that have been embedded are always removed.

# Queue never exceeds max len.

# A queue never has duplicate items.

# Files successfully downloaded always have valid audio format and non-zero size

# Embedding processes never deadlock when accessing shared queues

# Downloaded files are never deleted while still being processed by any embedder

# System resource usage (disk space, memory) stays within bounds during processing

