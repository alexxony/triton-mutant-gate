# Additional tests targeting boundary/tail handling and various sizes
_extra_shapes = [
    (1, 1),      # 1x1 matrix
    (1, 2),      # BLOCK_SIZE=2, no tail
    (2, 1),      # single column, multiple rows
    (2, 3),      # BLOCK_SIZE=4 → tail of 1
    (4, 5),      # BLOCK_SIZE=8 → tail of 3
    (8, 15),     # BLOCK_SIZE=16 → tail of 15
    (16, 17),    # BLOCK_SIZE=32 → tail of 17
    (32, 33),    # BLOCK_SIZE=64 → tail of 33
]

for m, n in _extra_shapes:
    check(m, n, 1.0)   # each invocation launches the kernel once
