def make_synthetic_episode(
    family,
    rng,
    source=1,
    target=2,
    turns=1,
    factor=2,
    min_size=3,
    max_size=8,
):
    pairs = []

    while len(pairs) < 4:
        height = rng.randint(min_size, max_size)
        width = rng.randint(min_size, max_size)

        grid = [
            [rng.randrange(10) for _ in range(width)]
            for _ in range(height)
        ]

        if family == "copy":
            output = [row[:] for row in grid]

        elif family == "recolour":
            grid[rng.randrange(height)][rng.randrange(width)] = source

            output = [
                [
                    target if colour == source else colour
                    for colour in row
                ]
                for row in grid
            ]

        elif family == "rotate":
            output = [row[:] for row in grid]

            for _ in range(turns % 4):
                output = [
                    list(row)
                    for row in zip(*output[::-1])
                ]

        elif family == "double_canvas":
            # Expand the canvas; keep the original pixels at top-left.
            output = [
                [0] * (factor * width)
                for _ in range(factor * height)
            ]

            for row in range(height):
                output[row][:width] = grid[row][:]

        else:
            raise ValueError(f"Unknown family: {family}")

        if any(pair["input"] == grid for pair in pairs):
            continue

        pairs.append({
            "input": grid,
            "output": output,
        })

    return pairs[:3], pairs[3]