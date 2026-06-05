import os
import random
import math
from sumolib.net import readNet

# =============================================================
# CONFIG
# =============================================================
NET_PATH = "Network/network_signalized.net.xml"
OUTPUT_DIR = "Demand/"

SIM_TIME = 3600

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# VEHICLE TYPES
# =============================================================
VEH_TYPES = {
    "Car":   {"def": '<vType id="Car" accel="2.6" decel="4.5" length="5" maxSpeed="16"/>', "ratio": 0.45},
    "Bike":  {"def": '<vType id="Bike" accel="2.1" decel="4.0" length="2" maxSpeed="14"/>', "ratio": 0.30},
    "Bus":   {"def": '<vType id="Bus" accel="1.0" decel="2.0" length="12" maxSpeed="12"/>', "ratio": 0.15},
    "Truck": {"def": '<vType id="Truck" accel="1.0" decel="2.0" length="10" maxSpeed="11"/>', "ratio": 0.10}
}

# =============================================================
# TRAFFIC MODES (DDS)
# =============================================================
TRAFFIC_MODES = {

    # -----------------------------
    # LIGHT - 300-500 vehicles
    # -----------------------------
    "light": {
        "routes": 15,
        "min": 20,
        "max": 40,
        "peak": 1.0,
        "desc": "LIGHT traffic (~400 vehicles)"
    },

    # -----------------------------
    # MEDIUM DDS - 800-1200 vehicles
    # -----------------------------
    "medium_low": {
        "routes": 18,
        "min": 30,
        "max": 60,
        "peak": 1.2,
        "desc": "MEDIUM LOW (~600 vehicles)"
    },

    "medium_mid": {
        "routes": 22,
        "min": 40,
        "max": 80,
        "peak": 1.5,
        "desc": "MEDIUM MID (~900 vehicles)"
    },

    "medium": {
        "routes": 25,
        "min": 50,
        "max": 100,
        "peak": 1.8,
        "desc": "MEDIUM (~1200 vehicles)"
    },

    # -----------------------------
    # HEAVY DDS - 1500-2500 vehicles
    # -----------------------------
    "heavy_low": {
        "routes": 28,
        "min": 60,
        "max": 120,
        "peak": 2.0,
        "desc": "HEAVY LOW (~1500 vehicles)"
    },

    "heavy_mid": {
        "routes": 32,
        "min": 70,
        "max": 140,
        "peak": 2.3,
        "desc": "HEAVY MID (~2000 vehicles)"
    },

    "heavy": {
        "routes": 35,
        "min": 80,
        "max": 160,
        "peak": 2.5,
        "desc": "HEAVY (~2500 vehicles)"
    }
}
# =============================================================
# PEAK TRAFFIC DISTRIBUTION
# =============================================================
def peak_factor(t, multiplier):

    morning = math.exp(-((t - 900) ** 2) / (2 * 500 ** 2))
    evening = math.exp(-((t - 1800) ** 2) / (2 * 500 ** 2))

    return 1 + multiplier * (morning + evening)


def sample_depart_time(multiplier):

    while True:

        t = random.uniform(0, SIM_TIME)

        p = peak_factor(t, multiplier) / (1 + multiplier)

        if random.random() < p:
            return t


# =============================================================
# LOAD NETWORK
# =============================================================
print("Loading network...")

net = readNet(NET_PATH)

edges = []
entry_edges = []
exit_edges = []

xmin = ymin = float("inf")
xmax = ymax = float("-inf")

# find network boundaries
for n in net.getNodes():
    x, y = n.getCoord()

    xmin = min(xmin, x)
    xmax = max(xmax, x)
    ymin = min(ymin, y)
    ymax = max(ymax, y)

# detect boundary edges
for e in net.getEdges():

    if e.getID().startswith(":"):
        continue

    edges.append(e.getID())

    shape = e.getShape()

    x, y = shape[0]

    # west boundary
    if x <= xmin + 10:
        entry_edges.append(e.getID())

    # east boundary
    if x >= xmax - 10:
        exit_edges.append(e.getID())

print("Edges:", len(edges))
print("Entry edges:", len(entry_edges))
print("Exit edges:", len(exit_edges))


# =============================================================
# TRAFFIC GENERATOR
# =============================================================
def generate_traffic(mode):

    cfg = TRAFFIC_MODES[mode]

    print("\nGenerating", mode.upper(), "traffic")
    print(cfg["desc"])

    ROUTES = cfg["routes"]
    MIN_VEH = cfg["min"]
    MAX_VEH = cfg["max"]
    PEAK = cfg["peak"]

    routes_xml = []
    vehicles = []

    veh_id = 0
    route_id = 0

    for r in range(ROUTES):

        found = False

        for attempt in range(50):

            start = random.choice(entry_edges)
            end = random.choice(exit_edges)

            if start == end:
                continue

            try:

                path = net.getShortestPath(
                    net.getEdge(start),
                    net.getEdge(end)
                )[0]

                if not path or len(path) < 2:
                    continue

                edge_list = " ".join([e.getID() for e in path])

                route_name = f"{mode}_route_{route_id}"

                routes_xml.append(
                    f'<route id="{route_name}" edges="{edge_list}"/>\n'
                )

                num_veh = random.randint(MIN_VEH, MAX_VEH)

                for v in range(num_veh):

                    depart = sample_depart_time(PEAK)

                    rnum = random.random()
                    acc = 0
                    vtype = "Car"

                    for vt, info in VEH_TYPES.items():
                        acc += info["ratio"]
                        if rnum <= acc:
                            vtype = vt
                            break

                    vehicles.append((
                        depart,
                        f'<vehicle id="{mode}_{veh_id}" '
                        f'type="{vtype}" '
                        f'depart="{depart:.2f}" '
                        f'route="{route_name}" '
                        f'departLane="best" '
                        f'departSpeed="max"/>\n'
                    ))

                    veh_id += 1

                route_id += 1
                found = True
                break

            except:
                continue

        if not found:
            print("Route generation failed")

    vehicles.sort(key=lambda x: x[0])

    output = os.path.join(OUTPUT_DIR, f"{mode}_traffic.rou.xml")

    with open(output, "w") as f:

        f.write("<routes>\n\n")

        for vt in VEH_TYPES.values():
            f.write(vt["def"] + "\n")

        f.write("\n")

        for r in routes_xml:
            f.write(r)

        f.write("\n")

        for _, v in vehicles:
            f.write(v)

        f.write("</routes>\n")

    print("Vehicles generated:", veh_id)
    print("Routes:", route_id)
    print("Saved:", output)

    return veh_id, route_id, output, mode


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":

    print("\n===============================")
    print("Traffic Generator (DDS)")
    print("===============================\n")

    for m in TRAFFIC_MODES:

        generate_traffic(m)

        print("--------------------------------")