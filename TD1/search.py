"""Search algorithms for the Pokémon navigation exercise."""

from __future__ import annotations

import random
from collections import deque

from pokemon_game import GameMap, State, successors


def generate_random_path(
    game_map: GameMap,
    start: State,
    goal: State,
    seed: int | None = None,
    max_steps: int = 10,
    max_attempts: int = 100,
) -> list[State]:
    """Generate a random valid path from the start to the goal.

    This function is provided only to demonstrate the path visualization.
    It is not a search algorithm and is not guaranteed to return a shortest
    path.

    At each step, the trainer randomly chooses one valid neighboring state.
    Unvisited neighbors are preferred, but previously visited states may be
    selected when necessary.

    Parameters
    ----------
    game_map : list[list[str]]
        The Pokémon map.
    start : tuple[int, int]
        Initial trainer position.
    goal : tuple[int, int]
        Pokémon Center position.
    seed : int or None, default=None
        Random seed used to make the generated path reproducible.
    max_steps : int, default=10
        Maximum number of movements in one attempt.
    max_attempts : int, default=100
        Maximum number of random walks attempted.

    Returns
    -------
    list[tuple[int, int]]
        A valid path beginning at ``start`` and ending at ``goal``.

    Raises
    ------
    RuntimeError
        If no random path reaches the goal within the allowed attempts.
    """
    if start == goal:
        return [start]

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    rng = random.Random(seed)

    for _ in range(max_attempts):
        path = [start]
        visited = {start}
        current = start

        for _ in range(max_steps):
            neighboring_states = successors(game_map, current)

            if not neighboring_states:
                break

            unvisited_neighbors = [
                state for state in neighboring_states if state not in visited
            ]

            if unvisited_neighbors:
                next_state = rng.choice(unvisited_neighbors)
            else:
                next_state = rng.choice(neighboring_states)

            path.append(next_state)
            visited.add(next_state)
            current = next_state

            if current == goal:
                return path

    raise RuntimeError(
        "The random walk did not reach the goal. "
        "Try increasing max_steps or max_attempts, or use another seed."
    )


def _reconstruct_path(
    parents: dict[State, State | None],
    goal: State,
) -> list[State]:
    """Reconstruct a path from the parent dictionary."""
    path = []

    current = goal

    while current is not None:
        path.append(current)
        current = parents[current]

    path.reverse()

    return path


def breadth_first_search(
    game_map: GameMap,
    start: State,
    goal: State,
) -> list[State]:
  """Return a shortest path using Breadth-First Search.
  """
  # 1. Initialisation de la file (FIFO : First-In, First-Out) avec l'état de départ
  frontier = deque([start])

  # Dictionnaire pour stocker les parents de chaque état et suivre les états visités
  # Clé : l'état, Valeur : son état parent (None pour le point de départ)
  parents: dict[State, State | None] = {start: None}

  # Tant qu'il reste des états à explorer dans la file
  while frontier:
    # On retire le PREMIER élément entré (principe du BFS : exploration par niveau)
    current = frontier.popleft()

    # Si l'état actuel correspond à l'objectif, on arrête et on reconstruit le chemin
    if current == goal:
      return _reconstruct_path(parents, goal)

    # On parcourt tous les voisins valides (successeurs) de l'état actuel
    for next_state in successors(game_map, current):
      # Si ce voisin n'a pas encore été découvert (il n'est pas dans les parents)
      if next_state not in parents:
        # On enregistre son parent pour pouvoir remonter le chemin plus tard
        parents[next_state] = current
        # On l'ajoute à la fin de la file pour l'explorer plus tard
        frontier.append(next_state)

  # Si la file est vide et qu'on n'a pas trouvé le but, c'est qu'il n'y a pas de chemin
  raise RuntimeError("No path exists from start to goal.")


def depth_first_search(
    game_map: GameMap,
    start: State,
    goal: State,
) -> list[State]:
  """Return a path using Depth-First Search.
  """
  # 1. Initialisation de la pile (LIFO : Last-In, First-Out) sous forme de liste
  # On y stocke des tuples : (l'état actuel, son parent direct)
  frontier: list[tuple[State, State | None]] = [(start, None)]

  # Dictionnaire pour stocker les parents des états EFFECTIVEMENT explorés (dépilés)
  parents: dict[State, State | None] = {}

  # Tant qu'il y a des états dans la pile
  while frontier:
    # On retire le DERNIER élément ajouté (situé au sommet de la pile)
    current, parent = frontier.pop()

    # En DFS, un même état peut être ajouté plusieurs fois par différentes branches.
    # S'il a déjà été exploré, on l'ignore pour éviter les boucles.
    if current in parents:
      continue

    # On officialise la visite de cet état en enregistrant son parent
    parents[current] = parent

    # Si l'état actuel est l'objectif, on reconstruit et retourne le chemin
    if current == goal:
      return _reconstruct_path(parents, goal)

    # On récupère les successeurs et on les inverse (`reversed`)
    # Pourquoi ? Parce que la pile inverse l'ordre de sortie. Inverser ici
    # permet de respecter la priorité du TD (up -> right -> down -> left).
    for next_state in reversed(successors(game_map, current)):
      # Si le successeur n'a pas encore été exploré
      if next_state not in parents:
        # On l'empile au sommet avec son parent
        frontier.append((next_state, current))

  # Si la pile est vide sans avoir atteint l'objectif, aucun chemin n'existe
  raise RuntimeError("No path exists from start to goal.")