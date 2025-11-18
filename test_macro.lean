import CLean.KernelMacro

open CLean.KernelMacro

-- Test the basic macro
test_gpu_kernel foo : Nat := 42

#check foo
#check foo_ir
#eval foo
#eval foo_ir
