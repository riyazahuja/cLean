import CLean.GPU

open GpuDSL

kernelArgs CopyArgs(N: Nat)
  global[input output: Array Float]

#print CopyArgs
#check CopyArgs
