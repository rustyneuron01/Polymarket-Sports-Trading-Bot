Two edits in the "Extending to NBA, Tennis, Other Sports" section (around line 185) and the project layout (around line 201).

EDIT 1 — in the extension table, replace the Game-data row:

OLD:
| **Game data**                     | `espn_client.py` (ESPN NHL scoreboard/summary)           | Add or swap client: ESPN NBA, ATP/WTA, etc. Same idea: score/period/clock → game state.             |

NEW:
| **Game data**                     | `espn_client.py` (ESPN NHL scoreboard/summary)           | Add or swap client: ESPN NBA, ATP/WTA, etc. Same idea: score/period/clock → game state. Tennis: `livetennisapi_client.py` (optional; live sets/games/points/server/break-point/retirement, free tier). |

EDIT 2 — in the "Project layout (main files)" code block, add a line right after the espn_client.py line:

OLD:
├── espn_client.py         # ESPN NHL scoreboard / game state

NEW:
├── espn_client.py         # ESPN NHL scoreboard / game state
├── livetennisapi_client.py # Optional Tennis client: live match state (sets/server/break-point)
