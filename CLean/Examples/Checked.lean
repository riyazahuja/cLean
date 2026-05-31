import CLean.PTX.Bridge
import CLean.Examples.Loop
import CLean.Examples.MatmulLoop
import CLean.Examples.Saxpy

namespace CLean
namespace Examples

def checkedMovKernel : PTX.Kernel :=
  { entry := "entry"
    regs := #[
      { name := "r0", ty := .u32 },
      { name := "r1", ty := .u32 }
    ]
    blocks := #[
      { label := "entry"
        body := #[{ instr := .mov .u32 "r1" (.reg "r0") }]
        term := .exit }
    ] }

def checkedUnsupportedKernel : PTX.Kernel :=
  { entry := "entry"
    blocks := #[
      { label := "entry"
        body := #[{ instr := .unsupported "volatile.load" #["volatile"] #[] }]
        term := .exit }
    ] }

example :
    PTX.lowerKernelSupported? checkedMovKernel = true := by
  native_decide

example :
    PTX.lowerKernelSupported? checkedUnsupportedKernel = false := by
  native_decide

end Examples
end CLean
