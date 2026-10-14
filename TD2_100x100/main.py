"""Compare Dijkstra and A* on a difficult random Pokémon map."""

from __future__ import annotations

from PIL import Image, ImageDraw

from pokemon_game import (
    GRASS,
    ROCK,
    TREE,
    GameMap,
    State,
    generate_hard_map,
    load_sprites,
    validate_path,
)
from search import (
    a_star,
    dijkstra,
    generate_random_path,
    path_cost,
)

# Keep a fixed seed while developing and grading.
# Use None to obtain a different random map each time.
MAP_SEED = 43
RANDOM_PATH_SEED = 10


def save_animation_as_gif(
    game_map: GameMap,
    start: State,
    goal: State,
    path: list[State],
    output_filename: str,
    duration_ms: int = 120,
    max_frames: int = 250,
) -> None:
    """Génère l'animation du parcours et l'enregistre au format GIF sans interface graphique.

    Utilise Pillow et les sprites officiels pour un rendu ultra-rapide et optimisé,
    même sur de grandes cartes (ex: 100x100).
    """
    validate_path(
        game_map,
        path,
        start=start,
        goal=goal,
    )

    rows = len(game_map)
    columns = len(game_map[0])

    # Adapter dynamiquement la taille de chaque tuile selon la résolution de la grille
    tile_size = max(6, min(32, 1000 // max(rows, columns)))

    print(f"Génération du GIF '{output_filename}' ({rows}x{columns}, tuiles de {tile_size}px)...")

    # Charger et redimensionner les sprites SVG officiels
    raw_sprites = load_sprites()
    sprites: dict[str, Image.Image] = {
        name: Image.fromarray(arr).resize(
            (tile_size, tile_size),
            Image.Resampling.LANCZOS,
        )
        for name, arr in raw_sprites.items()
    }

    # Pré-rendre le fond de la carte (herbe, arbres, rochers)
    bg = Image.new(
        "RGBA",
        (columns * tile_size, rows * tile_size),
        (255, 255, 255, 255),
    )

    for r in range(rows):
        for c in range(columns):
            sym = game_map[r][c]
            bg.paste(sprites["grass"], (c * tile_size, r * tile_size), sprites["grass"])
            if sym == TREE:
                bg.paste(sprites["tree"], (c * tile_size, r * tile_size), sprites["tree"])
            elif sym == ROCK:
                bg.paste(sprites["rock"], (c * tile_size, r * tile_size), sprites["rock"])

    # Placer le centre Pokémon à la destination
    bg.paste(sprites["clinic"], (goal[1] * tile_size, goal[0] * tile_size), sprites["clinic"])
    bg_rgb = bg.convert("RGB")

    # Si le chemin comporte beaucoup d'étapes (ex: marche aléatoire), adapter le pas
    step_stride = max(1, len(path) // max_frames)
    frame_indices = list(range(0, len(path), step_stride))
    if frame_indices[-1] != len(path) - 1:
        frame_indices.append(len(path) - 1)

    frames: list[Image.Image] = []
    trainer = sprites["trainer"]
    half_tile = tile_size // 2

    # Tracer le chemin parcouru et positionner le dresseur
    for idx in frame_indices:
        frame = bg_rgb.copy()
        draw = ImageDraw.Draw(frame)

        sub_path = path[: idx + 1]
        line_coords = [
            (c * tile_size + half_tile, r * tile_size + half_tile)
            for r, c in sub_path
        ]
        if len(line_coords) > 1:
            draw.line(line_coords, fill=(255, 0, 0), width=max(2, tile_size // 4))

        curr_r, curr_c = path[idx]
        frame.paste(trainer, (curr_c * tile_size, curr_r * tile_size), trainer)
        frames.append(frame)

    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"-> GIF '{output_filename}' généré avec succès ({len(frames)} images).")


def main() -> None:
    """Generate one map and compare Dijkstra with A*."""
    game_map, start, goal = generate_hard_map(
        rows=100,
        columns=100,
        tree_probability=0.30,
        rock_probability=0.45,
        minimum_distance=10,
        seed=MAP_SEED,
    )

    print(f"Initial state: {start}")
    print(f"Goal state:    {goal}")

    # ---------------------------------------------------------------
    # Random valid path demonstration
    # ---------------------------------------------------------------

    random_path = generate_random_path(
        game_map,
        start,
        goal,
        seed=RANDOM_PATH_SEED,
    )

    random_path_cost = path_cost(
        game_map,
        random_path,
    )

    print("\nRandom valid path")
    print("-----------------")
    print(f"Path length: {len(random_path) - 1}")
    print(f"Path cost:   {random_path_cost}")
    print(f"Path:        {random_path}")

    save_animation_as_gif(
        game_map,
        start,
        goal,
        random_path,
        "random_path.gif",
    )

    # ---------------------------------------------------------------
    # Dijkstra
    # ---------------------------------------------------------------

    dijkstra_path, dijkstra_considered = dijkstra(
        game_map,
        start,
        goal,
    )

    dijkstra_cost = path_cost(
        game_map,
        dijkstra_path,
    )

    print("\nDijkstra")
    print("---------")
    print(f"Path length:       {len(dijkstra_path) - 1}")
    print(f"Path cost:         {dijkstra_cost}")
    print(f"Considered states: {dijkstra_considered}")
    print(f"Path:              {dijkstra_path}")

    save_animation_as_gif(
        game_map,
        start,
        goal,
        dijkstra_path,
        "dijkstra_path.gif",
    )

    # ---------------------------------------------------------------
    # A*
    # ---------------------------------------------------------------

    astar_path, astar_considered = a_star(
        game_map,
        start,
        goal,
    )

    astar_cost = path_cost(
        game_map,
        astar_path,
    )

    print("\nA*")
    print("--")
    print(f"Path length:       {len(astar_path) - 1}")
    print(f"Path cost:         {astar_cost}")
    print(f"Considered states: {astar_considered}")
    print(f"Path:              {astar_path}")

    save_animation_as_gif(
        game_map,
        start,
        goal,
        astar_path,
        "astar_path.gif",
    )

    # ---------------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------------

    print("\nComparison")
    print("----------")
    print("Same optimal cost: " f"{dijkstra_cost == astar_cost}")

    print(
        "Difference in considered states: " f"{dijkstra_considered - astar_considered}"
    )


if __name__ == "__main__":
    main()
