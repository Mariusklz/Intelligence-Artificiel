"""Exercice 4 - Traffic Light Control Experiments.

Reproduit en script les expériences faites dans l'application web :
  a) entraîner + sauvegarder un contrôleur Value Iteration (Normal, paramètres par défaut)
  b) comparer à un "humain" (règles simples) sur les mêmes arrivées
  c) tester ce contrôleur en Rush Hour sans réentraînement
  d) réentraîner en Rush Hour et comparer
  e) augmenter la pénalité de changement beta

Lancement : python exercise4.py
"""
import json
import random
from math import exp

from app import app
from trafficrl.mdp import KEEP, SWITCH, MDPConfig

LAMBDA = {"Low": 0.4, "Normal": 0.8, "Rush Hour": 1.5}
N_STEPS = 2000
SEEDS = range(20)


# ---------------------------------------------------------------- entraînement
def train(arrival_rate=0.8, switch_penalty=0.0, algorithm="value_iteration"):
    """Entraîne via l'API Flask (comme le bouton Train AI) -> modèle sauvegardé."""
    client = app.test_client()
    body = {"algorithm": algorithm, "arrival_rate": arrival_rate,
            "switch_penalty": switch_penalty, "gamma": 0.95, "epsilon": 1e-3}
    start = client.post("/api/training/start", json=body).get_json()
    stream = client.get(start["stream_url"]).get_data(as_text=True)
    summary = None
    for block in stream.split("\n\n"):
        if block.startswith("event: complete"):
            summary = json.loads(block.split("data: ", 1)[1])["result"]
    model = client.get(f"/api/training/models/{start['model_id']}").get_json()
    policy = {tuple(x["state"]): x["action"] for x in model["policy"]}
    return start["model_id"], summary, policy


# ---------------------------------------------------------------- contrôleurs
# Chaque contrôleur est une fabrique : make() -> fonction state -> action
def ai(policy):
    return lambda: (lambda state: policy[state])


def human_threshold():
    """Humain 'naturel' : on bascule dès que la file au rouge est plus longue."""
    def act(state):
        q_ns, q_ew, g, c = state
        green, red = (q_ns, q_ew) if g == 0 else (q_ew, q_ns)
        return SWITCH if c == 0 and red > green else KEEP
    return act


def human_empty():
    """Humain 'patient' : on vide l'axe vert avant de basculer."""
    def act(state):
        q_ns, q_ew, g, c = state
        green, red = (q_ns, q_ew) if g == 0 else (q_ew, q_ns)
        return SWITCH if c == 0 and green == 0 and red > 0 else KEEP
    return act


def fixed_time(period=5):
    """Feu à cycle fixe (opposant par défaut de l'app : 15 s = 5 décisions de 3 s)."""
    t = 0

    def act(state):
        nonlocal t
        t += 1
        return SWITCH if state[3] == 0 and t % period == 0 else KEEP
    return act


# ---------------------------------------------------------------- simulation
def sample_arrivals(rng, lam, cap):
    """Poisson(lam) tronqué à cap (même loi que mdp.py)."""
    k, p, threshold = 0, 1.0, exp(-lam)
    while True:
        p *= rng.random()
        if p <= threshold or k >= cap:
            return k
        k += 1


def simulate(act, lam, beta=0.0, steps=N_STEPS, seed=0, cfg=MDPConfig()):
    """Dynamique exacte de l'MDP ; même seed => mêmes arrivées pour tous les contrôleurs."""
    rng = random.Random(seed)
    q_ns = q_ew = g = c = 0
    waiting = served = switches = lost = 0
    reward = 0.0
    for _ in range(steps):
        state = (q_ns, q_ew, g, c)
        a = act(state)
        if c:
            c, dep = c - 1, 0
        elif a == SWITCH:
            g, c, dep = 1 - g, cfg.switch_lost_steps - 1, 0
            switches += 1
            reward -= beta
        else:
            dep = cfg.service_capacity
        s_ns = min(q_ns, dep) if state[2] == 0 else 0
        s_ew = min(q_ew, dep) if state[2] == 1 else 0
        served += s_ns + s_ew
        n_ns = q_ns - s_ns + sample_arrivals(rng, lam, cfg.max_arrivals)
        n_ew = q_ew - s_ew + sample_arrivals(rng, lam, cfg.max_arrivals)
        lost += max(0, n_ns - cfg.max_queue) + max(0, n_ew - cfg.max_queue)
        q_ns, q_ew = min(cfg.max_queue, n_ns), min(cfg.max_queue, n_ew)
        waiting += q_ns + q_ew
        reward -= q_ns + q_ew
    return {"wait": waiting / steps, "switch": 100 * switches / steps,
            "served": served / steps, "lost": lost / steps, "reward": reward / steps}


def evaluate(make, lam, beta=0.0):
    runs = [simulate(make(), lam, beta, seed=s) for s in SEEDS]
    return {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}


def table(title, rows, lam, beta=0.0):
    print(f"\n{title}  (lambda={lam}, beta={beta}, {len(SEEDS)} x {N_STEPS} pas, arrivées identiques)")
    print(f"  {'contrôleur':32s} {'attente':>8s} {'switch%':>8s} {'servis':>7s} {'perdus':>7s} {'reward':>8s}")
    for name, make in rows:
        r = evaluate(make, lam, beta)
        print(f"  {name:32s} {r['wait']:8.2f} {r['switch']:8.1f} {r['served']:7.2f}"
              f" {r['lost']:7.3f} {r['reward']:8.2f}")


# ---------------------------------------------------------------- lecture de politique
def show_policy(policy, title, max_queue=10):
    """Carte KEEP(.)/SWITCH(S) pour feu vert = NS, cooldown 0."""
    print(f"\n{title}  [vert = NS ; lignes q_NS, colonnes q_EW ; S = SWITCH]")
    print("          q_EW " + " ".join(f"{j:2d}" for j in range(max_queue + 1)))
    for i in range(max_queue + 1):
        row = " ".join(" S" if policy[(i, j, 0, 0)] == SWITCH else " ." for j in range(max_queue + 1))
        print(f"  q_NS={i:2d}      {row}")


def switch_frontier(policy, max_queue=10):
    """Pour chaque q_NS (axe vert), plus petite file EW qui déclenche SWITCH (- si jamais)."""
    out = []
    for i in range(max_queue + 1):
        js = [j for j in range(max_queue + 1) if policy[(i, j, 0, 0)] == SWITCH]
        out.append(str(min(js)) if js else "-")
    return " ".join(f"{x:>2s}" for x in out)


if __name__ == "__main__":
    print("=" * 80, "\na) Value Iteration, Normal, paramètres par défaut")
    id_normal, summ, pi_normal = train(LAMBDA["Normal"])
    print(f"  modèle sauvegardé : models/trafficrl/{id_normal}.json -> {summ}")
    show_policy(pi_normal, "Politique VI-Normal")

    print("=" * 80, "\nb) Human vs AI (Normal)")
    controllers = [("IA VI-Normal", ai(pi_normal)),
                   ("Humain : rouge > vert", lambda: human_threshold()),
                   ("Humain : vider le vert d'abord", lambda: human_empty()),
                   ("Cycle fixe 15 s", lambda: fixed_time())]
    table("Normal", controllers, LAMBDA["Normal"])
    h = human_threshold()
    disagree = sorted(s for s in pi_normal if s[3] == 0 and pi_normal[s] != h(s))
    print(f"  états où IA et humain 'rouge > vert' divergent : {len(disagree)}/242, ex. {disagree[:6]}")

    print("=" * 80, "\nc) Rush Hour SANS réentraînement")
    table("Rush Hour", controllers, LAMBDA["Rush Hour"])

    print("=" * 80, "\nd) Réentraînement en Rush Hour")
    id_rush, summ, pi_rush = train(LAMBDA["Rush Hour"])
    print(f"  modèle sauvegardé : models/trafficrl/{id_rush}.json -> {summ}")
    show_policy(pi_rush, "Politique VI-Rush")
    print("\n  frontière de switch (q_EW min pour SWITCH, q_NS = 0..10)")
    print("    Normal :", switch_frontier(pi_normal))
    print("    Rush   :", switch_frontier(pi_rush))
    print(f"  états où les deux politiques diffèrent : {sum(pi_normal[s] != pi_rush[s] for s in pi_normal)}/484")
    table("Rush Hour", [("IA VI-Normal (non réentraînée)", ai(pi_normal)),
                        ("IA VI-Rush (réentraînée)", ai(pi_rush))], LAMBDA["Rush Hour"])
    table("Normal", [("IA VI-Normal", ai(pi_normal)),
                     ("IA VI-Rush (hors contexte)", ai(pi_rush))], LAMBDA["Normal"])

    print("=" * 80, "\ne) Pénalité de changement beta (Normal)")
    print(f"  {'beta':>4s} {'attente':>8s} {'switch%':>8s} {'reward':>8s}   frontière (q_NS = 0..10)")
    policies = {}
    for beta in (0, 2, 5, 10, 20, 50):
        _, _, policies[beta] = train(LAMBDA["Normal"], switch_penalty=beta)
        r = evaluate(ai(policies[beta]), LAMBDA["Normal"], beta)
        print(f"  {beta:4d} {r['wait']:8.2f} {r['switch']:8.1f} {r['reward']:8.2f}   {switch_frontier(policies[beta])}")
    show_policy(policies[10], "Politique VI-Normal, beta = 10")
