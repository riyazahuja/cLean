# WP Core Slice Coverage

The current proof target is the scalar, memory, and CFG slice of PTX-like kernels.
This is intentionally smaller than the full IR. Ordinary kernels using arithmetic,
predicates, loads/stores, branches, and loops should be representable and provable
end-to-end with `KernelSpec.Valid`; barriers, warp collectives, atomics, and MMA are
outside this proof target for now.

This document is coverage tracking only. The codebase does not currently define an
`Instr.IsCore`, `Block.IsCore`, or `KernelEnv.IsCore` predicate for this slice.
Scheduler control for the checked examples is expressed relative to the active
invariant: examples derive `StepMachineSelects` from an entailment into
`OnlyRunnableWarp 0 0`, instead of assuming every machine step globally selects
warp `(0, 0)`.

| IR operation | Executable semantics | WP/proof support | Example coverage |
| --- | --- | --- | --- |
| `assignReg` | Yes | WP rules, single-lane and lane-list framed rules | Basic WP examples, SAXPY scalar/vector, matmul scalar |
| `assignPred` | Yes | WP rules, single-lane and lane-list framed rules | Basic WP examples and branch setup |
| `assignPredValue` | Yes | WP rules, single-lane and lane-list framed rules | Basic WP examples |
| `load` | Yes | WP rules for global/shared/local/param/const, including lane-list global/shared/local/param/const rules | SAXPY scalar/vector, matmul scalar, global-memory loop, matmul inner loop, memory-rule examples |
| `store` | Yes | WP rules for global/shared/local | SAXPY scalar/vector, matmul scalar, global-memory loop, matmul inner loop, memory-rule examples |
| `cvta` | Yes | WP rules | Addressing examples |
| `isspacep` | Yes | WP rules | Addressing examples |
| `br` | Yes | WP rules and CFG preservation rules | Straight-line CFG examples |
| `cbr` | Yes | WP rules and branch-sensitive CFG VCs with local branch-control evidence for loop proofs | Branch-sensitive loop smoke test, counted loops, matmul inner loop |
| `terminate` | Yes | WP rules and finalization support | Completed scalar, vector-lift, and loop examples |
| `barrierCTA` | Yes | Low-level/outcome WP support only; not in the core proof target | Barrier-specific examples only |
| `warp` | Constructor exists | Executable semantics currently returns `none`; out of scope | None |
| `atomic` | Constructor exists | Executable semantics currently returns `none`; out of scope | None |
| `mma` | Constructor exists | Executable semantics currently returns `none`; out of scope | None |

Completed examples should not hide `KernelSpec.Valid` behind external `.Valid`
assumptions. The current scalar SAXPY proof is a per-lane scalar proof, and the
older matmul scalar proof is a single-contribution cell demo. The checked
`matmul_cell_loop_kernel_valid` theorem covers a real fixed-`n = 3` inner loop.

The current loop coverage includes a branch-sensitive CFG/WP smoke test with local
cbr target-control evidence, a checked `n = 3` arithmetic counted-loop validity
proof, a checked `n = 3` global-memory counted-loop validity proof with explicit
read-decode, output-encode, and input/output disjointness assumptions, and a checked
`n = 3` matmul-cell inner-loop validity proof, and a vector-level SAXPY theorem
that lifts per-lane scalar validity into a `∀ i, i < n` postcondition for
single-warp launches with `n ≤ 32`.
