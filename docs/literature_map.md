        # Literature Map

        Paper: 63 embodied_model_repair_memory

        Field box: adaptive world models for long-horizon robot tasks

        Thesis: Embodied Model Repair Memory turns the seed bet into a mechanism: Store repairs to model beliefs as reusable state, not transient corrective prompts.

        ## Landscape Sweep Summary
        The selector ranked records from the shared 500,000-record pool. The closest visible clusters are:
        - Scene Memory Transformer for Embodied Agents in Long-Horizon Tasks (2019)
- Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective (2026)
- To imitate or not to imitate: Boosting reinforcement learning-based construction robotic control for long-horizon tasks using virtual demonstrations (2022)
- ERRA: An Embodied Representation and Reasoning Architecture for Long-Horizon Language-Conditioned Manipulation Tasks (2023)
- Generative World Models of Tasks: LLM-Driven Hierarchical Scaffolding for Embodied Agents (2025)
- Dreaming when Necessary: Advancing World Action Models with Adaptive Multi-Modal Reasoning (2026)
- Learning 3D Persistent Embodied World Models (2025)
- IEI-TIA: Industrial Embodied Intelligence Trustworthy Interpretable Agent for Robotic Long-Horizon and Repetitive Tasks (2026)
- AHA-WAM:Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing (2026)
- What-If World: A Causal Benchmark for General World Models in Embodied Scenarios (2026)
- PI-VLA: Adaptive Symmetry-Aware Decision-Making for Long-Horizon Vision–Language–Action Manipulation (2026)
- FurnitureBench: Reproducible real-world benchmark for long-horizon complex manipulation (2025)

        ## Hidden Assumptions
        - The executed trajectory is a sufficient training target.
- Unobserved physical alternatives can be averaged into uncertainty.
- Task labels capture the mechanism that caused failure.
- A planner only needs nominal feasibility.
- Embodiment-specific contact effects are nuisance variation.

        ## Boundary
        The project avoids weak moves such as bigger models, generic uncertainty, or a benchmark-only contribution. It centers a mechanism-level change: Embodied model repair memory keeps action-critical alternatives explicit until a physical observation collapses them.
