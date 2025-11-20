import CLean.GPU

open GpuDSL

kernelArgs saxpyArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

#print saxpyArgs
