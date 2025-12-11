/-
  Generate PTX files for all 9 cLean kernels

  This file extracts the CUDA code from each kernel and compiles it to PTX
  for use in the C++ benchmark runner.
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import Benchmarks.Dataset.VectorAdd
import Benchmarks.Dataset.VectorSquare
import Benchmarks.Dataset.Saxpy
import Benchmarks.Dataset.ScalarMul
import Benchmarks.Dataset.MatrixTranspose
import Benchmarks.Dataset.MatrixMulTiled
import Benchmarks.Dataset.TemplateScale
import Benchmarks.Dataset.ExclusiveScan
import Benchmarks.Dataset.DotProduct

open CLean.DeviceCodeGen
open System

def nvccPath : String := "/usr/local/cuda/bin/nvcc"
def ptxDir : String := "Benchmarks/ptx"

def compileToPTX (name : String) (cudaCode : String) : IO Unit := do
  -- Ensure directory exists
  IO.FS.createDirAll ptxDir

  let cuFile := s!"{ptxDir}/{name}.cu"
  let ptxFile := s!"{ptxDir}/{name}.ptx"

  -- Write CUDA source
  IO.FS.writeFile cuFile cudaCode

  -- Compile to PTX
  let result ← IO.Process.output {
    cmd := nvccPath
    args := #["-ptx", "-arch=sm_75", "-o", ptxFile, cuFile]
  }

  if result.exitCode != 0 then
    IO.println s!"Error compiling {name}:"
    IO.println result.stderr
  else
    IO.println s!"Generated {ptxFile}"

def main : IO Unit := do
  IO.println "Generating PTX files for all cLean kernels..."
  IO.println ""

  -- 1. VectorAdd
  let vectorAddCuda := kernelToCuda CLean.Benchmarks.Dataset.VectorAdd.vectorAddKernelIR
  compileToPTX "vectoradd" vectorAddCuda

  -- 2. VectorSquare
  let vectorSquareCuda := kernelToCuda CLean.Benchmarks.Dataset.VectorSquare.vectorSquareKernelIR
  compileToPTX "vectorsquare" vectorSquareCuda

  -- 3. SAXPY
  let saxpyCuda := kernelToCuda CLean.Benchmarks.Dataset.Saxpy.saxpyKernelIR
  compileToPTX "saxpy" saxpyCuda

  -- 4. ScalarMul
  let scalarMulCuda := kernelToCuda CLean.Benchmarks.Dataset.ScalarMul.scalarMulKernelIR
  compileToPTX "scalarmul" scalarMulCuda

  -- 5. MatrixTranspose
  let transposeCuda := kernelToCuda CLean.Benchmarks.Dataset.MatrixTranspose.transposeKernelIR
  compileToPTX "matrixtranspose" transposeCuda

  -- 6. MatrixMul
  let matMulCuda := kernelToCuda CLean.Benchmarks.Dataset.MatrixMulTiled.matMulKernelIR
  compileToPTX "matrixmul" matMulCuda

  -- 7. TemplateScale
  let templateScaleCuda := kernelToCuda CLean.Benchmarks.Dataset.TemplateScale.templateScaleKernelIR
  compileToPTX "templatescale" templateScaleCuda

  -- 8. ExclusiveScan
  let scanCuda := kernelToCuda CLean.Benchmarks.Dataset.ExclusiveScan.scanKernelIR
  compileToPTX "exclusivescan" scanCuda

  -- 9. DotProduct
  let dotProductCuda := kernelToCuda CLean.Benchmarks.Dataset.DotProduct.dotProductKernelIR
  compileToPTX "dotproduct" dotProductCuda

  IO.println ""
  IO.println "PTX generation complete!"
