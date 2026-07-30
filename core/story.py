"""The story spine of PROJECT R-5.

One place that knows what all five rooms are called, which learning method each
one uses, and which component R-5 recovers from it.  The navigation rail, the
inventory bar and the transition screens all read from here, so the story stays
consistent no matter which page the player is looking at.
"""

GAME_TITLE = "PROJECT R-5"
GAME_SUBTITLE = "Escape from the Learning Lab"

INTRO_STORY = (
    "Sub-level 4 of the Institute has been sealed since the incident at 02:14. "
    "R-5, an experimental navigation robot, powers up alone on a charging plate "
    "in the dark. The emergency lockdown has armed every test chamber between "
    "here and the surface. To reach the exit, R-5 must cross all five of them — "
    "and each chamber has to be solved a different way."
)

ROOMS = {
    1: {
        "number": 1,
        "name": "Laser Security Chamber",
        "sector": "LAB SECTOR A-01",
        "subtitle": "SECURITY PROTOCOL ACTIVE",
        "algorithm": "Value Iteration",
        "algorithm_short": "VALUE ITER.",
        "model_known": True,
        "briefing_warning": "Laser grid armed — contact returns R-5 to the start platform",
        "algorithm_family": "Dynamic Programming",
        "mission": "Reach the control panel and shut down the laser grid",
        "component": "Navigation Chart",
        "component_icon": "\U0001F5FA",
        "cleared_headline": "SECURITY CHAMBER CLEARED",
        "cleared_story": (
            "The laser grid is offline. R-5 downloads the facility navigation "
            "chart from the control panel and the door to the industrial sector "
            "unlocks."
        ),
        "implemented": True,
    },
    2: {
        "number": 2,
        "name": "Broken Bridge Sector",
        "sector": "LAB SECTOR B-04",
        "subtitle": "STRUCTURAL FAILURE DETECTED",
        "algorithm": "SARSA",
        "algorithm_short": "SARSA",
        "model_known": False,
        "briefing_warning": "Structure unstable — a fall into the shaft ends the run",
        "algorithm_family": "Temporal-Difference Learning",
        "mission": "Recover the security keycard and reach the exit door",
        "component": "Security Keycard",
        "component_icon": "\U0001F511",
        "cleared_headline": "BRIDGE SECTOR CLEARED",
        "cleared_story": (
            "The bridge network is behind R-5. The security keycard opens the "
            "bulkhead into the reactor level, where the power grid is still down."
        ),
        "implemented": True,
    },
    3: {
        "number": 3,
        "name": "Reactor Control Chamber",
        "sector": "LAB SECTOR C-07",
        "subtitle": "POWER GRID OFFLINE",
        "algorithm": "Q-Learning",
        "algorithm_short": "Q-LEARNING",
        "model_known": False,
        "briefing_warning": "Patrol unit active in the reactor chamber",
        "algorithm_family": "Temporal-Difference Learning",
        "mission": "Activate all three generators and avoid the patrol unit",
        "component": "Power Core",
        "component_icon": "⚡",
        "cleared_headline": "REACTOR CHAMBER CLEARED",
        "cleared_story": (
            "The reactor is back online and the blast door is open. R-5 carries a "
            "charged power core into the aerospace test hall, where the corridor "
            "ahead has collapsed."
        ),
        "implemented": True,
    },
    4: {
        "number": 4,
        "name": "Drone Wind Tunnel",
        "sector": "LAB SECTOR D-07",
        "subtitle": "FLIGHT STABILIZATION TEST",
        "algorithm": "Semi-Gradient SARSA",
        "algorithm_short": "SG SARSA",
        "model_known": False,
        "briefing_warning": "High airflow — landing speed is limited",
        "algorithm_family": "Function Approximation · Tile Coding",
        "mission": "Fly the R-5 drone link through the airflow and land softly",
        "component": "Drone Drive Unit",
        "component_icon": "\U0001F681",
        "cleared_headline": "WIND TUNNEL CLEARED",
        "cleared_story": (
            "The drone touches down on the far platform and R-5 disconnects, "
            "carrying the drive unit into the last sector: an automated warehouse "
            "that rearranges itself between missions."
        ),
        "implemented": True,
    },
    5: {
        "number": 5,
        "name": "Adaptive Storage Facility",
        "sector": "LAB SECTOR E-12",
        "subtitle": "LAYOUT RECONFIGURATION ACTIVE",
        "algorithm": "Semi-Gradient Q-Learning",
        "algorithm_short": "SG Q-LEARN",
        "model_known": False,
        "briefing_warning": "Layout unknown — radar contact only",
        "algorithm_family": "Function Approximation · Generalisation",
        "mission": "Navigate an unknown warehouse layout using radar only",
        "component": "Exit Key",
        "component_icon": "\U0001F6AA",
        "cleared_headline": "STORAGE FACILITY CLEARED",
        "cleared_story": (
            "The final gate is open. R-5 walks out of the facility with every "
            "component recovered and a navigation policy that works in warehouses "
            "it has never seen."
        ),
        "implemented": True,
    },
}

ROOM_NUMBERS = sorted(ROOMS.keys())
FIRST_ROOM = ROOM_NUMBERS[0]
LAST_ROOM = ROOM_NUMBERS[-1]


def room(number):
    """The story record for one room."""
    return ROOMS[number]


def next_room(number):
    """The number of the following room, or None after the last one."""
    if number >= LAST_ROOM:
        return None
    return number + 1
