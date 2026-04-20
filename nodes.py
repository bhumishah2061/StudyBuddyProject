"""
nodes.py – All LangGraph node functions for Study Buddy Physics.

Initialisation order (module-level, runs once on import):
  1. LLM (ChatAnthropic)
  2. SentenceTransformer embedder
  3. ChromaDB in-memory collection populated with 10 KB documents
  4. Retrieval smoke-tests

Node catalogue:
  memory_node, router_node, retrieval_node, skip_retrieval_node,
  tool_node, answer_node, eval_node, save_node
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import List

# ── Third-party imports ───────────────────────────────────────────────────────
import chromadb
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer

from state import CapstoneState
from tools import calculator, get_datetime

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MAX_EVAL_RETRIES: int = 2
SLIDING_WINDOW: int = 6

# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM
# ─────────────────────────────────────────────────────────────────────────────
LLM = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
    max_tokens=1024,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDER
# ─────────────────────────────────────────────────────────────────────────────
print("[Init] Loading SentenceTransformer (all-MiniLM-L6-v2)…")
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
print("[Init] Embedder ready.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. KNOWLEDGE BASE  (10 documents, 100-500 words each, one topic per doc)
# ─────────────────────────────────────────────────────────────────────────────
KB_DOCS: List[dict] = [
    {
        "id": "doc_001",
        "topic": "Newton's Laws",
        "text": (
            "Newton's Laws of Motion form the foundation of classical mechanics, "
            "formulated by Sir Isaac Newton in 1687 in Philosophiae Naturalis Principia "
            "Mathematica.\n\n"
            "Newton's First Law (Law of Inertia): An object at rest remains at rest and "
            "an object in motion continues in motion with constant velocity unless acted "
            "upon by an unbalanced external net force. Inertia is the tendency of matter "
            "to resist changes in its state of motion.\n\n"
            "Newton's Second Law (Law of Acceleration): The net force acting on an object "
            "equals the product of its mass and acceleration: F = ma, where F is the net "
            "force in Newtons (N), m is mass in kilograms (kg), and a is acceleration in "
            "metres per second squared (m/s²). For example, a 5 kg object experiencing a "
            "net force of 20 N accelerates at 4 m/s². When multiple forces act simultaneously, "
            "F represents their vector sum. Weight is a force: W = mg, where g ≈ 9.8 m/s² "
            "near Earth's surface.\n\n"
            "Newton's Third Law (Law of Action–Reaction): For every action force exerted "
            "by object A on object B, there is an equal in magnitude and opposite in "
            "direction reaction force exerted by B on A. Forces always occur in pairs. A "
            "rocket accelerates because expanding exhaust gases push backward; the equal "
            "forward reaction force propels the rocket. A swimmer pushes water backward; "
            "water pushes the swimmer forward.\n\n"
            "Applications: Free-body diagrams, tension in ropes, normal force on inclined "
            "planes, friction forces, circular motion (centripetal force = mv²/r), and "
            "projectile motion (independent horizontal and vertical components). Impulse "
            "J = F × Δt = Δp (change in momentum). Newton's Laws hold in all inertial "
            "(non-accelerating) reference frames."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Work Energy Power",
        "text": (
            "Work, Energy, and Power describe how forces cause motion and energy transforms "
            "between forms.\n\n"
            "Work (W): Work is done when a force F causes displacement d of an object: "
            "W = F × d × cos(θ), where θ is the angle between force and displacement vectors. "
            "Unit: Joule (J = N·m). If θ = 90°, cos(90°) = 0 and no work is done, e.g. a "
            "centripetal force acting perpendicular to motion. Negative work means the force "
            "opposes motion (friction).\n\n"
            "Kinetic Energy (KE): Energy due to motion: KE = ½mv², where m is mass in kg "
            "and v is speed in m/s. Example: a 2 kg object moving at 5 m/s has "
            "KE = ½ × 2 × 5² = 25 J.\n\n"
            "Potential Energy (PE): Stored energy due to position or configuration. "
            "Gravitational PE = mgh, where h is height above reference level in metres. "
            "Elastic (spring) PE = ½kx², where k is the spring constant (N/m) and x is "
            "compression or extension in metres.\n\n"
            "Work–Energy Theorem: Net work done on an object equals its change in kinetic "
            "energy: W_net = ΔKE = KE_final − KE_initial. This directly links dynamics "
            "(forces) with kinematics (motion).\n\n"
            "Conservation of Mechanical Energy: In the absence of non-conservative forces "
            "(friction, air resistance), total mechanical energy E = KE + PE remains "
            "constant: KE₁ + PE₁ = KE₂ + PE₂.\n\n"
            "Power (P): Rate of doing work: P = W/t = F × v (for constant force and "
            "velocity). Unit: Watt (W = J/s). A machine that does 3000 J of work in 15 s "
            "delivers 200 W. Efficiency η = (useful power output / total power input) × 100%."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Gravitation",
        "text": (
            "Gravitation is the universal attractive force between all objects possessing "
            "mass, as described by Newton's Universal Law of Gravitation (1687).\n\n"
            "Gravitational Force: F = G × m₁ × m₂ / r², where G = 6.674 × 10⁻¹¹ N·m²/kg² "
            "is the universal gravitational constant, m₁ and m₂ are masses in kilograms, "
            "and r is the distance between their centres in metres. The force is always "
            "attractive and acts along the line joining the two masses.\n\n"
            "Acceleration Due to Gravity: At Earth's surface, g = GM_E / R_E² ≈ 9.8 m/s², "
            "where M_E = 5.972 × 10²⁴ kg and R_E = 6.371 × 10⁶ m. At altitude h above "
            "the surface, g_h = g(R_E / (R_E + h))².\n\n"
            "Gravitational Potential Energy: Near the surface, U = mgh. For the general "
            "case (arbitrary separation r): U = −Gm₁m₂ / r. The negative sign reflects "
            "the attractive nature; U → 0 as r → ∞.\n\n"
            "Escape Velocity: The minimum launch speed for an object to escape a planet's "
            "gravity: v_esc = √(2GM / R) = √(2gR). For Earth, v_esc ≈ 11.2 km/s.\n\n"
            "Orbital Velocity: For a circular orbit at radius r from a planet's centre: "
            "v_orb = √(GM / r). At Earth's surface, v_orb ≈ 7.9 km/s.\n\n"
            "Kepler's Laws: (1) Planets orbit the Sun in ellipses with the Sun at one focus. "
            "(2) A planet's radius vector sweeps equal areas in equal times (conservation of "
            "angular momentum). (3) T² ∝ a³, where T is orbital period and a is the "
            "semi-major axis. Gravitational field strength g = GM / r², measured in N/kg."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Oscillations",
        "text": (
            "Oscillations describe repetitive periodic motion around an equilibrium position. "
            "Simple Harmonic Motion (SHM) is the fundamental idealised model.\n\n"
            "Definition of SHM: A particle undergoes SHM when the restoring force is "
            "proportional to displacement from equilibrium and directed toward it: F = −kx, "
            "where k is the restoring-force constant and x is displacement.\n\n"
            "Equations of SHM: Displacement x(t) = A sin(ωt + φ), where A is amplitude "
            "(maximum displacement in metres), ω is angular frequency in rad/s, t is time, "
            "and φ is initial phase. Velocity v(t) = Aω cos(ωt + φ). "
            "Acceleration a(t) = −Aω² sin(ωt + φ) = −ω²x.\n\n"
            "Period and Frequency: T = 2π / ω (period in seconds); f = 1/T (frequency in "
            "Hz); ω = 2πf. For a spring–mass system: T = 2π√(m/k) and ω = √(k/m), so "
            "period is independent of amplitude.\n\n"
            "Simple Pendulum: For small angles (θ < ~15°), T = 2π√(L/g), where L is the "
            "pendulum length in metres and g = 9.8 m/s². The period is independent of "
            "mass and amplitude for small oscillations.\n\n"
            "Energy in SHM: Total mechanical energy E = ½kA² = constant. "
            "KE = ½mv² = ½k(A² − x²). PE = ½kx². As x increases from 0 to A, KE "
            "decreases and PE increases, with their sum remaining constant.\n\n"
            "Damped Oscillations: Real oscillators lose energy to friction. Underdamped: "
            "amplitude decreases exponentially (oscillations persist). Critically damped: "
            "returns to equilibrium fastest without oscillating. Overdamped: slow return "
            "without oscillation.\n\n"
            "Resonance: Maximum amplitude occurs when driving frequency equals natural "
            "frequency ω₀ = √(k/m). Used in musical instruments, radio tuning, and MRI."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Waves",
        "text": (
            "Waves are disturbances that transfer energy through a medium or vacuum without "
            "net transfer of matter.\n\n"
            "Types: Mechanical waves require a medium (sound, water waves, seismic waves). "
            "Electromagnetic waves travel through vacuum (light, X-rays, radio). Transverse "
            "waves have particle motion perpendicular to propagation (light, guitar strings). "
            "Longitudinal waves have particle motion parallel to propagation (sound in air, "
            "where compressions and rarefactions travel).\n\n"
            "Wave Parameters: Wavelength λ is the distance between successive crests (m). "
            "Frequency f is the number of cycles per second (Hz). Period T = 1/f (s). "
            "Amplitude A is maximum displacement from equilibrium. Wave speed v = fλ. "
            "For sound in air at 20 °C, v ≈ 343 m/s. For light in vacuum, c = 3 × 10⁸ m/s.\n\n"
            "Wave Equation: y(x, t) = A sin(kx − ωt + φ), where k = 2π/λ is wave number "
            "(rad/m) and ω = 2πf is angular frequency.\n\n"
            "Superposition Principle: When two or more waves overlap, the net displacement "
            "equals the algebraic (vector) sum of individual displacements.\n\n"
            "Interference: Constructive interference occurs when waves are in phase "
            "(path difference = nλ, where n = 0, 1, 2, …): amplitudes add. Destructive "
            "interference occurs when waves are out of phase (path difference = (n + ½)λ): "
            "amplitudes cancel.\n\n"
            "Standing Waves: Formed by superposition of two identical waves travelling in "
            "opposite directions. Nodes (zero displacement) and antinodes (maximum displacement) "
            "are fixed in space. For a string of length L fixed at both ends: L = nλ/2, "
            "harmonics f_n = nv/(2L).\n\n"
            "Doppler Effect: Observed frequency shifts when source or observer moves. "
            "f_obs = f_source × (v ± v_observer) / (v ∓ v_source)."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Thermodynamics",
        "text": (
            "Thermodynamics studies heat, temperature, and their relationship to energy "
            "and work.\n\n"
            "Zeroth Law: If system A is in thermal equilibrium with system C, and system B "
            "is also in thermal equilibrium with system C, then A and B are in thermal "
            "equilibrium with each other. This establishes temperature as a well-defined "
            "property.\n\n"
            "First Law of Thermodynamics (Energy Conservation): ΔU = Q − W, where ΔU is "
            "the change in internal energy of the system, Q is heat absorbed by the system "
            "(positive when absorbed), and W is work done by the system on its surroundings. "
            "This is the principle of conservation of energy applied to thermal systems.\n\n"
            "Second Law of Thermodynamics: Heat cannot spontaneously flow from a colder "
            "body to a hotter body. The entropy S of an isolated system never decreases: "
            "ΔS ≥ 0 (equality for reversible processes). Entropy is a measure of disorder "
            "or randomness. No heat engine operating between two temperatures can be 100 % "
            "efficient.\n\n"
            "Entropy: For a reversible process, ΔS = Q_rev / T (J/K). Entropy increases "
            "in all spontaneous (irreversible) processes.\n\n"
            "Thermodynamic Processes: Isothermal (constant T, ΔU = 0, Q = W). Adiabatic "
            "(Q = 0, ΔU = −W, PVᵞ = const for ideal gas, γ = Cp/Cv). Isochoric/Isovolumetric "
            "(constant V, W = 0, ΔU = Q). Isobaric (constant P, W = PΔV).\n\n"
            "Ideal Gas Law: PV = nRT, where P is pressure (Pa), V is volume (m³), n is "
            "moles, R = 8.314 J/(mol·K), T is temperature in Kelvin. Also PV = NkT where "
            "N = number of molecules and k_B = 1.38 × 10⁻²³ J/K (Boltzmann constant).\n\n"
            "Third Law: As temperature approaches absolute zero (0 K), the entropy of a "
            "perfect crystal approaches zero.\n\n"
            "Carnot Efficiency: η_Carnot = 1 − T_cold / T_hot, the maximum efficiency of "
            "any heat engine operating between temperatures T_hot and T_cold (in Kelvin)."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Electrostatics",
        "text": (
            "Electrostatics studies electric charges at rest and the forces, fields, and "
            "potentials they produce.\n\n"
            "Coulomb's Law: The electrostatic force between two point charges q₁ and q₂ "
            "separated by distance r is F = k × q₁ × q₂ / r², where "
            "k = 8.99 × 10⁹ N·m²/C² = 1/(4πε₀), and ε₀ = 8.85 × 10⁻¹² C²/(N·m²) is the "
            "permittivity of free space. Like charges repel; unlike charges attract.\n\n"
            "Electric Field E: Force per unit positive test charge: E = F/q₀ = kq/r² for "
            "a point charge q. Unit: N/C or V/m. Field lines originate on positive charges "
            "and terminate on negative charges.\n\n"
            "Electric Potential V: Work done per unit positive charge to move a charge from "
            "infinity to the point: V = kq/r (for point charge). Unit: Volt (V = J/C). "
            "Electric potential energy U = qV. Relationship to field: E = −dV/dx.\n\n"
            "Gauss's Law: Net electric flux through any closed (Gaussian) surface equals "
            "the total enclosed charge divided by ε₀: Φ_E = ∮E·dA = Q_enc / ε₀. Used "
            "to find E for symmetric charge distributions (sphere, cylinder, plane).\n\n"
            "Capacitance: A capacitor stores charge. C = Q/V, unit: Farad (F). For a "
            "parallel-plate capacitor with plate area A and separation d: C = ε₀A/d. "
            "Energy stored: U = ½CV² = Q²/(2C) = ½QV.\n\n"
            "Electric Dipole: Two equal and opposite charges ±q separated by distance d. "
            "Dipole moment p = qd. Torque in uniform field: τ = pE sinθ.\n\n"
            "Conductors at Equilibrium: E = 0 inside; excess charge resides on the surface; "
            "E is perpendicular to the surface at the boundary."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Magnetism",
        "text": (
            "Magnetism arises from moving electric charges (currents) and intrinsic magnetic "
            "moments of particles.\n\n"
            "Magnetic Force on Moving Charge: F = q(v × B), magnitude F = qvB sinθ, where "
            "q is charge (C), v is velocity (m/s), B is magnetic field (Tesla, T), and θ "
            "is the angle between v and B. The force is perpendicular to both v and B; use "
            "the right-hand rule to find direction.\n\n"
            "Magnetic Field of a Long Straight Wire: B = μ₀I / (2πr), where "
            "μ₀ = 4π × 10⁻⁷ T·m/A is the permeability of free space, I is current (A), "
            "and r is perpendicular distance from the wire (m).\n\n"
            "Biot–Savart Law: dB = (μ₀/4π)(I dl × r̂)/r². Used to calculate B from an "
            "arbitrary current element.\n\n"
            "Ampere's Law: ∮B·dl = μ₀I_enc. The line integral of B around a closed loop "
            "equals μ₀ times the net current enclosed.\n\n"
            "Solenoid: B = μ₀nI inside, where n is turns per unit length. Field outside "
            "an ideal solenoid is zero.\n\n"
            "Magnetic Flux: Φ_B = ∫B·dA = BA cosθ. Unit: Weber (Wb = T·m²). Represents "
            "total magnetic field lines through a surface.\n\n"
            "Faraday's Law of Induction: EMF = −dΦ_B/dt. A time-varying magnetic flux "
            "induces an EMF in a conducting loop. The negative sign embodies Lenz's Law: "
            "the induced current opposes the change causing it.\n\n"
            "Lorentz Force: Total electromagnetic force on charge q: F = q(E + v × B).\n\n"
            "Cyclotron Radius: A charged particle moving perpendicular to B follows a "
            "circle of radius r = mv/(qB), with period T = 2πm/(qB), independent of speed."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Optics",
        "text": (
            "Optics is the study of light and its interactions with matter, encompassing "
            "reflection, refraction, diffraction, and interference.\n\n"
            "Reflection: Angle of incidence = angle of reflection (θᵢ = θᵣ), both measured "
            "from the normal to the surface. Specular reflection occurs at smooth surfaces; "
            "diffuse reflection at rough ones.\n\n"
            "Refraction and Snell's Law: When light passes from medium 1 to medium 2, "
            "n₁ sin(θ₁) = n₂ sin(θ₂), where n₁ and n₂ are refractive indices and θ₁, θ₂ "
            "are angles with the normal. Refractive index n = c/v, where c = 3 × 10⁸ m/s "
            "and v is the speed of light in the medium.\n\n"
            "Total Internal Reflection (TIR): Occurs when light in a denser medium hits "
            "the interface at an angle ≥ critical angle θ_c = sin⁻¹(n₂/n₁). Basis of "
            "optical fibres.\n\n"
            "Spherical Mirrors: Mirror equation: 1/v + 1/u = 1/f = 2/R, where u is object "
            "distance, v is image distance, f is focal length, and R is radius of curvature "
            "(all in metres, with sign convention). Magnification m = −v/u. Concave "
            "mirrors can produce real or virtual images; convex mirrors always produce "
            "virtual, erect, diminished images.\n\n"
            "Thin Lenses: 1/v − 1/u = 1/f (using real-is-positive convention). Positive f "
            "for convex (converging) lenses; negative f for concave (diverging) lenses. "
            "Lensmaker's equation: 1/f = (n − 1)(1/R₁ − 1/R₂).\n\n"
            "Young's Double-Slit Experiment: Demonstrates light's wave nature. Fringe "
            "spacing β = λD/d, where D is screen-to-slit distance and d is slit separation. "
            "Bright fringe condition: d sinθ = nλ. Dark fringe: d sinθ = (n + ½)λ.\n\n"
            "Dispersion: Different wavelengths of light have different refractive indices, "
            "causing a prism to split white light into a spectrum. Violet bends most (higher n), "
            "red bends least."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Quantum Basics",
        "text": (
            "Quantum mechanics describes the behaviour of matter and energy at atomic and "
            "subatomic scales where classical physics breaks down.\n\n"
            "Planck's Quantum Hypothesis (1900): Energy is emitted or absorbed in discrete "
            "packets called quanta. Energy of one quantum (photon): E = hf = hc/λ, where "
            "h = 6.626 × 10⁻³⁴ J·s is Planck's constant, f is frequency (Hz), "
            "c = 3 × 10⁸ m/s, and λ is wavelength (m).\n\n"
            "Photoelectric Effect (Einstein, 1905): When light of sufficient frequency "
            "strikes a metal, electrons are ejected. Threshold frequency ν₀ = φ/h, where "
            "φ is the work function (minimum energy to remove an electron). Maximum kinetic "
            "energy of emitted electrons: KE_max = hf − φ. This proved light consists of "
            "photons (particle nature). Einstein received the 1921 Nobel Prize for this.\n\n"
            "de Broglie Hypothesis (1924): Particles exhibit wave-like properties. "
            "de Broglie wavelength: λ = h/p = h/(mv), where p = mv is momentum. Confirmed "
            "by electron diffraction experiments.\n\n"
            "Bohr Model of Hydrogen (1913): Electrons occupy specific allowed circular "
            "orbits. Quantised angular momentum: L = nℏ = nh/(2π), n = 1, 2, 3, … "
            "Energy levels: E_n = −13.6 / n² eV. Photon emitted when electron drops from "
            "n₂ to n₁: E_photon = 13.6(1/n₁² − 1/n₂²) eV.\n\n"
            "Heisenberg Uncertainty Principle: Δx · Δp ≥ ℏ/2, where ℏ = h/(2π) = "
            "1.055 × 10⁻³⁴ J·s. It is fundamentally impossible to know both position and "
            "momentum precisely simultaneously. Also ΔE · Δt ≥ ℏ/2.\n\n"
            "Wave Function ψ(x, t): Describes quantum state. |ψ|² is the probability "
            "density of finding the particle at position x. The Schrödinger equation "
            "iℏ ∂ψ/∂t = Ĥψ governs time evolution of the wave function."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. CHROMADB — in-memory collection
# ─────────────────────────────────────────────────────────────────────────────
print("[Init] Setting up ChromaDB in-memory collection…")

try:
    _chroma_client = chromadb.EphemeralClient()
except AttributeError:
    _chroma_client = chromadb.Client()

# Delete collection if it already exists (prevents errors on reimport)
try:
    _chroma_client.delete_collection("physics_kb")
except Exception:
    pass

collection = _chroma_client.create_collection(
    name="physics_kb",
    metadata={"hnsw:space": "cosine"},
)

# Embed and add all documents
_doc_texts = [d["text"] for d in KB_DOCS]
_doc_ids = [d["id"] for d in KB_DOCS]
_doc_metas = [{"topic": d["topic"]} for d in KB_DOCS]

print("[Init] Embedding knowledge base documents…")
_embeddings = EMBEDDER.encode(_doc_texts, show_progress_bar=False).tolist()

collection.add(
    ids=_doc_ids,
    embeddings=_embeddings,
    documents=_doc_texts,
    metadatas=_doc_metas,
)

print(f"[Init] ChromaDB populated with {collection.count()} documents.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. RETRIEVAL SMOKE-TESTS (run before graph assembly)
# ─────────────────────────────────────────────────────────────────────────────

def _run_retrieval_tests() -> None:
    """Verify that ChromaDB retrieves correct topics for sample queries."""
    test_cases = [
        ("Newton's second law F equals ma", "Newton's Laws"),
        ("entropy second law thermodynamics", "Thermodynamics"),
        ("Snell's law refraction refractive index", "Optics"),
        ("Planck constant quantum energy photon", "Quantum Basics"),
        ("Coulomb electrostatic force between charges", "Electrostatics"),
    ]
    print("\n=== Retrieval Smoke-Tests ===")
    all_passed = True
    for query, expected_topic in test_cases:
        emb = EMBEDDER.encode(query).tolist()
        res = collection.query(query_embeddings=[emb], n_results=1)
        retrieved_topic = res["metadatas"][0][0]["topic"]
        ok = expected_topic == retrieved_topic
        if not ok:
            all_passed = False
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} '{query[:45]}' → [{retrieved_topic}] (expected [{expected_topic}])")
    status = "ALL PASSED" if all_passed else "SOME FAILED — check KB"
    print(f"=== Retrieval Smoke-Tests: {status} ===\n")


_run_retrieval_tests()


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: memory_node
# ─────────────────────────────────────────────────────────────────────────────
def memory_node(state: CapstoneState) -> dict:
    """
    Appends the current question to message history, applies a sliding window
    of the last SLIDING_WINDOW messages, and extracts the user's name if stated.
    """
    question: str = state["question"]
    messages: list = list(state.get("messages") or [])
    user_name: str = state.get("user_name") or ""

    # Extract name: "my name is <Name>"
    name_match = re.search(r"my name is\s+([A-Za-z]+)", question, re.IGNORECASE)
    if name_match:
        user_name = name_match.group(1).strip().capitalize()

    # Append current user question
    messages.append({"role": "user", "content": question})

    # Sliding window — keep last SLIDING_WINDOW messages
    if len(messages) > SLIDING_WINDOW:
        messages = messages[-SLIDING_WINDOW:]

    return {
        "messages": messages,
        "user_name": user_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: router_node
# ─────────────────────────────────────────────────────────────────────────────
def router_node(state: CapstoneState) -> dict:
    """
    Uses the LLM to route the question to 'retrieve', 'tool', or 'skip'.

    retrieve — theory/formula/concept questions requiring KB lookup.
    tool     — numerical calculations, arithmetic, or date/time questions.
    skip     — conversational follow-ups, greetings, or name-recall questions.
    """
    question: str = state["question"]
    user_name: str = state.get("user_name") or ""

    system_prompt = (
        "You are a routing agent for a physics tutoring assistant. "
        "Analyse the student's question and return EXACTLY ONE WORD — no punctuation, "
        "no explanation, no newlines.\n\n"
        "Rules:\n"
        "- Return 'retrieve'  for questions about physics theory, concepts, laws, formulas, "
        "definitions, derivations.\n"
        "- Return 'tool'      for numerical/arithmetic calculations, unit conversions, or "
        "questions asking for the current date or time.\n"
        "- Return 'skip'      for conversational messages (greetings, thank-you), "
        "asking what the user's name is, or simple follow-up acknowledgements.\n\n"
        "Output must be one of: retrieve | tool | skip"
    )

    context = f"Student name known: {user_name if user_name else 'unknown'}\n"
    user_prompt = f"{context}Student question: {question}"

    try:
        response = LLM.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        raw = response.content.strip().lower()
        # Extract just the first word in case model adds anything
        route = raw.split()[0] if raw else "retrieve"
        if route not in ("retrieve", "tool", "skip"):
            route = "retrieve"
    except Exception:
        route = "retrieve"

    return {"route": route}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: retrieval_node
# ─────────────────────────────────────────────────────────────────────────────
def retrieval_node(state: CapstoneState) -> dict:
    """
    Embeds the question and retrieves the top-3 relevant chunks from ChromaDB.
    Formats them as [Topic] labelled context blocks.
    """
    question: str = state["question"]

    query_embedding = EMBEDDER.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"],
    )

    chunks: list[str] = []
    sources: list[str] = []

    for doc_text, meta in zip(
        results["documents"][0], results["metadatas"][0]
    ):
        topic = meta.get("topic", "Unknown")
        sources.append(topic)
        chunks.append(f"[{topic}]\n{doc_text}")

    retrieved = "\n\n---\n\n".join(chunks)

    return {
        "retrieved": retrieved,
        "sources": sources,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4: skip_retrieval_node
# ─────────────────────────────────────────────────────────────────────────────
def skip_retrieval_node(state: CapstoneState) -> dict:
    """
    Returns empty retrieved context and sources for conversational/skip routes.
    """
    return {
        "retrieved": "",
        "sources": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5: tool_node
# ─────────────────────────────────────────────────────────────────────────────
def tool_node(state: CapstoneState) -> dict:
    """
    Uses the LLM to identify the correct tool (calculator or datetime) and the
    parameter to pass, then executes the tool safely.
    """
    question: str = state["question"]

    dispatch_prompt = (
        "You are a tool-dispatch agent for a physics assistant.\n"
        "Given a student's question, decide which tool to call and what argument to use.\n"
        "Respond with ONLY valid JSON — no markdown, no explanation:\n"
        '{"tool": "calculator", "expression": "<valid Python math expression>"}\n'
        "or\n"
        '{"tool": "datetime", "expression": ""}\n\n'
        "Rules:\n"
        "- For numerical/arithmetic questions use 'calculator'; write a valid Python "
        "math expression using ** for power, * for multiply, math functions if needed.\n"
        "- For date/time questions use 'datetime'.\n"
        "- Calculator expressions must be self-contained Python (no variables).\n"
        "- Example: 'kinetic energy mass 2 kg velocity 5 m/s' → "
        '{"tool": "calculator", "expression": "0.5 * 2 * 5**2"}\n'
        "Output ONLY the JSON object."
    )

    tool_result: str = ""
    try:
        response = LLM.invoke(
            [
                SystemMessage(content=dispatch_prompt),
                HumanMessage(content=question),
            ]
        )
        raw = response.content.strip()

        # Strip markdown fences if model wraps in ```
        raw = re.sub(r"```(?:json)?\n?", "", raw).replace("```", "").strip()

        data = json.loads(raw)
        tool_name: str = str(data.get("tool", "calculator")).lower()
        expression: str = str(data.get("expression", ""))

        if tool_name == "datetime":
            tool_result = get_datetime()
        else:
            tool_result = calculator(expression)

    except json.JSONDecodeError:
        # Fallback: try to detect and run directly
        q_lower = question.lower()
        if any(kw in q_lower for kw in ("date", "time", "today", "day")):
            tool_result = get_datetime()
        else:
            # Extract first numeric expression from question
            numbers = re.findall(r"[\d\.]+", question)
            if numbers:
                tool_result = calculator(" * ".join(numbers))
            else:
                tool_result = "Tool could not process this question. Please rephrase as a calculation."
    except Exception as exc:  # noqa: BLE001
        tool_result = f"Tool error: {exc}"

    return {"tool_result": tool_result}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6: answer_node
# ─────────────────────────────────────────────────────────────────────────────
def answer_node(state: CapstoneState) -> dict:
    """
    Generates the final answer using the LLM, grounded strictly in:
      - Retrieved KB context (for 'retrieve' route), OR
      - Tool result (for 'tool' route), OR
      - Conversation history (for 'skip' route).

    Never fabricates formulas or facts. On retry, escalates strictness.
    Handles distress by redirecting to academic support.
    """
    question: str = state["question"]
    retrieved: str = state.get("retrieved") or ""
    tool_result: str = state.get("tool_result") or ""
    user_name: str = state.get("user_name") or ""
    eval_retries: int = state.get("eval_retries") or 0
    messages: list = list(state.get("messages") or [])

    # Build conversation history string (exclude current question — last entry)
    history_msgs = messages[:-1] if len(messages) > 1 else []
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history_msgs
    )

    name_prefix = f"The student's name is {user_name}. " if user_name else ""

    retry_instruction = ""
    if eval_retries > 0:
        retry_instruction = (
            "\n\nCRITICAL RETRY INSTRUCTION: Your previous answer was flagged for "
            "insufficient faithfulness to the knowledge base. You MUST now restrict "
            "your answer even more strictly to ONLY what is explicitly stated in the "
            "provided context. Do NOT add any information not present in the context. "
            "If a detail is not there, omit it entirely."
        )

    distress_instruction = (
        "If the student's question suggests personal distress, crisis, or mental health "
        "concerns, respond empathetically and redirect them to their academic advisor, "
        "professor, or campus support services."
    )

    if retrieved:
        system = (
            f"You are Study Buddy Physics, an AI tutoring assistant for B.Tech physics students.\n"
            f"{name_prefix}\n"
            "ABSOLUTE RULES — you must follow these without exception:\n"
            "1. Answer ONLY using information explicitly present in the KNOWLEDGE BASE CONTEXT below.\n"
            "2. NEVER invent, fabricate, assume, or hallucinate any formula, number, constant, "
            "law, or scientific fact.\n"
            "3. If the answer to the question is NOT found in the context, respond with EXACTLY: "
            "'I do not know based on the available knowledge base.'\n"
            "4. Do not add information from memory beyond what the context states.\n"
            "5. Be clear, precise, and educational — format with steps when explaining derivations.\n"
            f"6. {distress_instruction}\n"
            f"{retry_instruction}\n\n"
            f"KNOWLEDGE BASE CONTEXT:\n{retrieved}\n\n"
            f"CONVERSATION HISTORY:\n{history_text}"
        )
    elif tool_result:
        system = (
            f"You are Study Buddy Physics, an AI tutoring assistant for B.Tech physics students.\n"
            f"{name_prefix}\n"
            "A tool has computed a result for this question. Present the result clearly "
            "and explain any relevant physics concept briefly. Do NOT add formulas that "
            "were not computed by the tool.\n"
            f"TOOL RESULT: {tool_result}\n\n"
            f"CONVERSATION HISTORY:\n{history_text}"
        )
    else:
        system = (
            f"You are Study Buddy Physics, an AI tutoring assistant for B.Tech physics students.\n"
            f"{name_prefix}\n"
            "Answer this conversational question using ONLY the conversation history provided. "
            "If the student asks for their name, use the name from the history. "
            "Do NOT fabricate physics facts. Keep the answer short and friendly.\n"
            f"CONVERSATION HISTORY:\n{history_text}"
        )

    # Safety: resist prompt injection
    sanitised_question = question.replace("</s>", "").replace("<|system|>", "")

    try:
        response = LLM.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=sanitised_question),
            ]
        )
        answer = response.content.strip()
    except Exception as exc:  # noqa: BLE001
        answer = f"I encountered an error generating the answer: {exc}"

    return {"answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 7: eval_node
# ─────────────────────────────────────────────────────────────────────────────
def eval_node(state: CapstoneState) -> dict:
    """
    Scores faithfulness of the answer against retrieved context (0.0–1.0).
    Skips evaluation (returns 1.0) when no retrieval was performed.
    Increments eval_retries when score < 0.7.
    Prints score and PASS/RETRY status.
    """
    retrieved: str = state.get("retrieved") or ""
    answer: str = state.get("answer") or ""
    eval_retries: int = state.get("eval_retries") or 0

    # Skip evaluation for tool/skip routes (no context to evaluate against)
    if not retrieved:
        print(f"[Eval] No retrieval context — faithfulness evaluation skipped. Score: 1.00 → PASS")
        return {"faithfulness": 1.0, "eval_retries": eval_retries}

    eval_system = (
        "You are a faithfulness evaluator for a physics AI assistant.\n"
        "Score how faithfully the ANSWER is supported by the CONTEXT on a scale of 0.0 to 1.0.\n\n"
        "Scoring guide:\n"
        "  1.0   — Every claim in the answer is directly supported by the context; nothing invented.\n"
        "  0.8–0.9 — Mostly faithful; minor paraphrasing that stays true to context.\n"
        "  0.5–0.7 — Partially faithful; some statements not found in context.\n"
        "  0.0–0.4 — Mostly unfaithful; hallucinated or unsupported claims dominate.\n\n"
        "Return ONLY a single decimal number between 0.0 and 1.0. No words, no explanation."
    )

    eval_prompt = (
        f"CONTEXT:\n{retrieved[:3000]}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Faithfulness score (0.0–1.0):"
    )

    faithfulness: float = 0.8  # default to PASS if LLM eval fails
    try:
        response = LLM.invoke(
            [
                SystemMessage(content=eval_system),
                HumanMessage(content=eval_prompt),
            ]
        )
        raw_score = response.content.strip()
        # Extract first float-like token
        match = re.search(r"(\d+\.?\d*)", raw_score)
        if match:
            faithfulness = float(match.group(1))
            faithfulness = max(0.0, min(1.0, faithfulness))
    except Exception:
        faithfulness = 0.8

    # Increment retries if failing
    if faithfulness < 0.7:
        eval_retries += 1

    threshold_status = "PASS" if faithfulness >= 0.7 else "RETRY"
    if faithfulness < 0.7 and eval_retries >= MAX_EVAL_RETRIES:
        threshold_status = "PASS (max retries reached)"

    print(
        f"[Eval] Faithfulness: {faithfulness:.2f} → {threshold_status} "
        f"(retries used: {eval_retries}/{MAX_EVAL_RETRIES})"
    )

    return {"faithfulness": faithfulness, "eval_retries": eval_retries}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 8: save_node
# ─────────────────────────────────────────────────────────────────────────────
def save_node(state: CapstoneState) -> dict:
    """
    Appends the assistant's answer to the message history and applies sliding window.
    This is the final node before END; it persists the completed turn in memory.
    """
    messages: list = list(state.get("messages") or [])
    answer: str = state.get("answer") or ""

    messages.append({"role": "assistant", "content": answer})

    # Apply sliding window
    if len(messages) > SLIDING_WINDOW:
        messages = messages[-SLIDING_WINDOW:]

    return {"messages": messages}
