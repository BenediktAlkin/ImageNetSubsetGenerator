with open("res/1_1percent") as f:
    lines = f.readlines()
lines = [line.replace("\n", "") for line in lines]

for line in sorted(lines):
    print(f"    \"{line}\",")