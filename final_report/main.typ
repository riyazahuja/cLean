#import "@preview/arkheion:0.1.1": arkheion, arkheion-appendices

#let abs = [
    We present *cLean*, a domain-specific language (DSL) embedding the Lean 4 interactive theorem prover for natively writing CUDA-style GPU kernels, efficiently executing them on-device, and formally verifying their safety and correctness properties by leveraging Lean's powerful dependent type theory. cLean provides high-level abstractions for imperative SIMT programming within a monadic, functional framework, enabling annotation-free, higher-order, and interactive verification of GPU kernels. Our system includes a full parser and transpiler for the core cLean DSL, as well as a formal semantics and verification framework for proving safety properties about these kernels. Moreover, we implement an execution engine for running these kernels on both GPUs and CPU simulators, and extend prior work to prove safety and functional correctness interactively, without the need of an external SMT solver, and within higher-order logic. We demonstrate cLean's expressiveness and performance through a suite of benchmark kernels, showing significant speedup against CPU baselines, as well as competitive performance against equivalent hand-written CUDA code. We additionally show the robustness of our system by showcasing its superior theorem proving abilities, as well as introduce a case study of the end-to-end implementation and verification of a novel, formally verified Gröbner basis computation algorithm, showcasing its ability to produce efficient GPU code while ensuring rigorous correctness guarantees. *TODO CLEANUP*
]

#show: arkheion.with(
    title: "cLean: Formally Verified GPU Programming in Lean 4",
    authors: (
        (name: "Riyaz Ahuja", email: "riyaza@andrew.cmu.edu", affiliation: "Carnegie Mellon University"),
    ),
    abstract: abs,
    keywords: ("Lean 4", "Formal Verification", "CUDA", "Domain-Specific Languages", "Thread Safety", "Grobner Bases"),
)

#set heading(numbering: none)

= Background

GPU kernels are critical for modern high-performance computing, and as these kernels run at large scale and often manipulate shared memory across thousands of threads, safety (e.g., absence of data races, correct synchronization) and correctness (the kernel actually implements the intended algorithm) are essential. Race conditions or incorrect thread interaction can silently corrupt results at scale, and debugging these behaviors on a GPU is notoriously difficult. Thus, a framework for writing GPU kernels that can be formally and statically verified before execution has enormous practical value.

Prior work in GPU kernel verification includes tools such as GPUVerify and GKLEE. GPUVerify in particular pioneered the "two-thread abstraction," reducing a parallel GPU kernel to a sequential program whose properties can be checked by an SMT solver. This technique successfully proves race-freedom and barrier-safety for many small kernels, and follow-up work enhanced invariant generation to increase automation. However, these systems show clear limitations: as kernels become more complex, especially when they include loops with non-trivial inductive structure, irregular memory accesses, or deeper arithmetic reasoning, the underlying SMT solver often fails to discharge the verification conditions. This highlights a fundamental limitation of purely SMT-based approaches to verification: they lack the expressive power and interactive proof capabilities needed to reason about more sophisticated GPU programs. Moreover, these prior systems focus almost exclusively on safety; they do not attempt to verify functional correctness of the kernel relative to a high-level specification.

Lean 4, a modern interactive theorem prover designed around a small trusted kernel and a high-performance native compiler, offers the capabilities that SMT-only tools lack. Lean's metaprogramming framework allows us to define custom DSLs, manipulate syntactic objects, and generate code programmatically. Its proof system provides powerful interactive tactics (simp, auto, lean-smt, etc.) that combine automation of proof search in higher-order logic and dependent type theories with human interactivity. Lean also supports reasoning about inductive invariants, algebraic specifications, Hoare-style pre/post-conditions, and correctness statements that SMT solvers alone cannot robustly handle. This combination makes Lean an excellent foundation for GPU kernel verification in a way that allows the complex underlying parallelism to be abstracted away from the user, as well as enabling easy integration of GPU acceleration with computer algebra in formalized mathematics. *TODO MAKE BETTER AND BETTER MENTION RELATED WORK*

= Methodology

Our approach is to implement a verified GPU DSL extending GPUVerify's approach directly inside Lean 4, giving it a fully formal semantics. Using Lean's metaprogramming, we transpile this DSL to CUDA C++, allowing for live execution via Lean's Infoview. Additionally, in the transpilation process, our system produces a corresponding Lean expression representing the kernel's semantics, allowing us to apply Lean's tactic framework to automatically or interactively prove properties of the kernel, including race-freedom, barrier-safety, and optionally full functional correctness relative to a user-specified contract. Loop invariants, pre/post-conditions, and algebraic correctness proofs can all be carried out inside Lean, enabling verification far beyond what SMT-only systems can achieve.

With this system, we verify small benchmark kernels and, as a stretch goal, apply it to the acceleration of a critical compute-intensive task inside Lean itself: an efficient Gröbner-basis-based ideal-membership algorithm. As Lean is primarily used by mathematicians for formalization, there is large demand for efficient algebraic reasoning systems, and the current tactics for testing ideal membership in commutative algebra and algebraic geometry are primarily CPU-bound code called via API from external CAS systems. This algorithm is highly parallelizable, as each monomial reduction step can be run concurrently on device.

*TODO MAKE BETTER AND BE DETAILED AND FIGURES*

= Experiments and Results

*PARAGRAPH INTRODUCING THE SECTION*

== Experimental Setup

*dataset, hardware, software; how do we measure performance; what baselines do we compare against; what metrics do we use for verification success; do we measure full time for execution?*

== Performance Results

*Big comparison of CPU vs GPU vs Handwritten CUDA; tables and graphs; discuss speedups and overheads; discuss limitations*

*Also discuss performance breakdown on different kernels in our dataset: show the breakdown in total runtime, GPU/CPU/memory utilization/throughput, etc.*

== Verification Results

*Big comparison of which kernels we could verify safety on; compare against prior work (GPUVerify, GKLEE); discuss which proofs were automatic vs interactive; discuss time taken for verification; discuss limitations*

*Also disucss correctness results for certain kernels, mention proof size, automation, etc.*

== Qualitative Analysis

*Give examples of kernels, launching scripts, safety proofs, correctness proofs; discuss the expressiveness of our DSL; discuss the usability of our system; discuss the power of Lean's proof tactics in this context*

= Grobner Basis Case Study

*Motivate why Grobner basis is important; discuss prior work on Grobner basis in Lean and other systems; discuss the algorithm we implement; discuss how we parallelize it; discuss how we verify it; show performance results; show verification results; discuss limitations and future work*

== Algorithm

*Discuss the standard Grobner basis algorithm; discuss the ideal membership problem; discuss how we implement it in Lean; discuss how we parallelize it for GPU execution*

*Outline parallelization strategy; discuss challenges in parallelizing; discuss how we overcome them*

*Give the explicit algorithm we implement and where parallelization and edge cases arise*

== Performance Results

*Show speedups and performance results; compare against CPU baseline, and best-in-class SoTA F4 performance; discuss limitations; note what examples we're feeding in*

== Verification Results

*Give proof of safety informally; discuss proof of correctness; discuss challenges in verification; discuss how we overcame them; discuss limitations and future work*


= Conclusion

*Summary of contributions; summary of results; future work directions*


#show: arkheion-appendices
=

== Additional Qualitative Results

=== Safety Proofs

=== Correctness Proofs

