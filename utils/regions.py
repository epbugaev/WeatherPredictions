"""Геометрические константы для региональных пространственных кропов ERA5.

Кропы здесь работают как «индексные окна» по уже-нарезанному 128×256 гриду
WeatherBench, и подаются в дата-сэты в формате ``[[i0, i1], [j0, j1]]``
(lat-окно, lon-окно).
"""

from typing import Final

# Original VKR South Atlantic Ocean crop on the 128x256 WeatherBench grid.
SOUTH_ATLANTIC_CROP: Final[list[list[int]]] = [[36, 68], [125, 189]]

# USA crop used by the current reruns. Lat-window is roughly 24N..56N and
# lon-window roughly 125W..67W on the 128x256 WeatherBench grid.
USA_CROP: Final[list[list[int]]] = [[75, 107], [164, 228]]
