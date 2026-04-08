import json
import re
from tqdm import tqdm

# -----------------------
# Parse visits from plan
# -----------------------
def parse_plan(plan):
    pattern = r"\*\*Day (\d+)-(\d+):\*\*.*?visit (.*?)(?: for|\n|$)"
    matches = re.findall(pattern, plan, re.IGNORECASE)

    visits = []
    for start, end, city in matches:
        city = city.strip().split("(")[0]
        visits.append((city, int(start), int(end)))

    return visits


# -----------------------
# Extract structure
# -----------------------
def extract_structure(plan):
    visits = parse_plan(plan)

    cities = [v[0] for v in visits]
    durations = [v[2] - v[1] for v in visits]
    total_days = visits[-1][2] if visits else 0

    return cities, durations, total_days


# -----------------------
# CSA check
# -----------------------
def is_valid(pred, gold):

    try:
        p_cities, p_durations, p_total = extract_structure(pred)
        g_cities, g_durations, g_total = extract_structure(gold)

        print(p_cities, p_durations, p_total)
        print(g_cities, g_durations, g_total)
        print("=======================================================")

        checks = []
        checks.append(p_cities == g_cities)        # order + coverage
        checks.append(p_durations == g_durations)  # duration match
        checks.append(p_total == g_total)          # total days

        return all(checks)

    except:
        return False


# -----------------------
# Main evaluation
# -----------------------
def compute_csa(data):

    total = len(data)
    valid = 0

    for key, item in tqdm(data.items()):

        pred = item["our_output"]
        gold = item["golden_plan"]

        if is_valid(pred, gold):
            valid += 1
            
        

    print(f"\n✅ CSA: {valid}/{total} = {valid/total:.4f}")


with open("trip_planning_Qwen3-0.6B_new_2.json", "r") as f:
    data = json.load(f)

compute_csa(data)