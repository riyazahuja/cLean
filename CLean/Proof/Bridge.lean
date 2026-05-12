import CLean.PTX.Bridge

namespace CLean
namespace PTX

/-- Checked kernel lowering is the current trusted boundary into executable IR. -/
def LoweredKernelWf (kernel : Kernel) (env : KernelEnv) : Prop :=
  lowerKernelEnvChecked? kernel = .ok env ∧ KernelEnv.wf env

/-- Checked module lowering means every checked kernel has a well-formed executable environment. -/
def LoweredModuleWf (checked : CheckedModule) : Prop :=
  ∀ ck ∈ checked.kernels.toList, LoweredKernelWf ck.source ck.env

theorem lowerKernelEnvChecked_wf {kernel : Kernel} {env : KernelEnv}
    (h : lowerKernelEnvChecked? kernel = .ok env) :
    KernelEnv.wf env := by
  -- The checked lowering construction inserts blocks keyed by their labels and rejects unknown entry labels.
  sorry

theorem lowerKernelEnvChecked_loweredKernelWf {kernel : Kernel} {env : KernelEnv}
    (h : lowerKernelEnvChecked? kernel = .ok env) :
    LoweredKernelWf kernel env := by
  exact ⟨h, lowerKernelEnvChecked_wf h⟩

theorem parseAndLowerKernel?_sound {input : String} {checked : CheckedKernel}
    (h : parseAndLowerKernel? input = .ok checked) :
    LoweredKernelWf checked.source checked.env := by
  -- This packages parser success and checked lowering success as the proof-facing bridge theorem.
  sorry

theorem parseAndLowerModule?_sound {input : String} {checked : CheckedModule}
    (h : parseAndLowerModule? input = .ok checked) :
    LoweredModuleWf checked := by
  -- This is the module-level lift of `parseAndLowerKernel?_sound`; keep as a theorem surface for now.
  sorry

end PTX
end CLean
