"""
Quick test to verify expand_fen and collapse_fen functions work correctly.
This doesn't require torch, so it can be run in any Python environment.
"""

def expand_fen(fen):
    """Converts standard FEN to expanded FEN format."""
    board_fen = fen.split()[0] if ' ' in fen else fen
    rows = board_fen.split('/')
    expanded_rows = []
    for row in rows:
        expanded_row = ""
        for char in row:
            if char.isdigit():
                expanded_row += '1' * int(char)
            else:
                expanded_row += char
        expanded_rows.append(expanded_row)
    return "/".join(expanded_rows)

def collapse_fen(expanded_fen):
    """Converts expanded FEN back to standard FEN notation."""
    rows = expanded_fen.split('/')
    collapsed_rows = []
    for row in rows:
        collapsed_row = ""
        count = 0
        for char in row:
            if char == '1':
                count += 1
            else:
                if count > 0:
                    collapsed_row += str(count)
                    count = 0
                collapsed_row += char
        if count > 0:
            collapsed_row += str(count)
        collapsed_rows.append(collapsed_row)
    return "/".join(collapsed_rows)

# Test cases
test_cases = [
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
    "8/8/8/8/8/8/8/8",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R"
]

print("Testing expand_fen and collapse_fen functions:\n")
for fen in test_cases:
    expanded = expand_fen(fen)
    collapsed = collapse_fen(expanded)
    match = collapsed == fen
    print(f"Original:  {fen}")
    print(f"Expanded:  {expanded}")
    print(f"Collapsed: {collapsed}")
    print(f"Match:     {match}")
    print("-" * 60)

print("\nAll tests completed!")

