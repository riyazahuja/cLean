/-
  Main Benchmark Runner

  Orchestrates all kernel benchmarks and generates reports
-/

import Benchmarks.Harness
import Benchmarks.Dataset.VectorAdd
import Benchmarks.Dataset.VectorSquare
import Benchmarks.Dataset.Saxpy
import Benchmarks.Dataset.ScalarMul
import Benchmarks.Dataset.MatrixTranspose
import Benchmarks.Dataset.MatrixMulTiled
import Benchmarks.Dataset.TemplateScale
import Benchmarks.Dataset.ExclusiveScan
import Benchmarks.Dataset.DotProduct

namespace CLean.Benchmarks

open CLean.Benchmarks

/-! ## Main Benchmark Suite Runner -/

def runAllKernelBenchmarks : IO (Array KernelBenchmarkSuite) := do
  IO.println "╔══════════════════════════════════════════════════════════════════╗"
  IO.println "║              cLean GPU Benchmark Suite                           ║"
  IO.println "║    Comparing: CPU Reference vs cLean GPU vs CUDA Reference       ║"
  IO.println "╚══════════════════════════════════════════════════════════════════╝"
  IO.println ""

  let mut allResults : Array KernelBenchmarkSuite := #[]

  -- Run each kernel's benchmark suite
  let vectorAddSuite ← Dataset.VectorAdd.runAllBenchmarks
  allResults := allResults.push vectorAddSuite

  let vectorSquareSuite ← Dataset.VectorSquare.runAllBenchmarks
  allResults := allResults.push vectorSquareSuite

  let saxpySuite ← Dataset.Saxpy.runAllBenchmarks
  allResults := allResults.push saxpySuite

  let scalarMulSuite ← Dataset.ScalarMul.runAllBenchmarks
  allResults := allResults.push scalarMulSuite

  let transposeSuite ← Dataset.MatrixTranspose.runAllBenchmarks
  allResults := allResults.push transposeSuite

  let matMulSuite ← Dataset.MatrixMulTiled.runAllBenchmarks
  allResults := allResults.push matMulSuite

  let templateScaleSuite ← Dataset.TemplateScale.runAllBenchmarks
  allResults := allResults.push templateScaleSuite

  let scanSuite ← Dataset.ExclusiveScan.runAllBenchmarks
  allResults := allResults.push scanSuite

  let dotProductSuite ← Dataset.DotProduct.runAllBenchmarks
  allResults := allResults.push dotProductSuite

  return allResults

/-! ## Report Generation -/

-- Simple float formatting
def fmtFloat (f : Float) : String :=
  let s := f.toString
  -- Truncate to reasonable length
  if s.length > 10 then s.take 10 else s

def generateMarkdownReport (suites : Array KernelBenchmarkSuite) : String := Id.run do
  let mut s := "# cLean GPU Benchmark Results\n\n"
  s := s ++ "## Overview\n\n"
  s := s ++ "This report compares:\n"
  s := s ++ "- **CPU Reference**: Sequential Lean implementation\n"
  s := s ++ "- **cLean GPU**: GPU kernel generated from Lean DSL\n"
  s := s ++ "- **CUDA Reference**: Hand-written CUDA (when available)\n\n"

  -- Summary table
  s := s ++ "## Summary\n\n"
  s := s ++ "| Kernel | Total Tests | Correct | Max Speedup vs CPU |\n"
  s := s ++ "|--------|-------------|---------|--------------------|\n"

  for suite in suites do
    let correct := suite.results.filter (·.correct) |>.size
    let total := suite.results.size
    let maxSpeedup := suite.results.foldl (fun acc r => max acc (r.cpuTimeMs / r.gpuTotalTimeMs)) 0.0
    s := s ++ s!"| {suite.kernelName} | {total} | {correct}/{total} | {fmtFloat maxSpeedup}x |\n"

  s := s ++ "\n"

  -- Detailed results per kernel
  for suite in suites do
    s := s ++ s!"## {suite.kernelName}\n\n"
    s := s ++ s!"{suite.description}\n\n"

    s := s ++ "| Input Size | CPU (ms) | GPU Total (ms) | GPU Kernel (ms) | Speedup | Correct |\n"
    s := s ++ "|------------|----------|----------------|-----------------|---------|----------|\n"

    for result in suite.results do
      let speedup := result.cpuTimeMs / result.gpuTotalTimeMs
      let checkmark := if result.correct then "✓" else "✗"
      s := s ++ s!"| {result.inputSize} | {fmtFloat result.cpuTimeMs} | {fmtFloat result.gpuTotalTimeMs} | {fmtFloat result.gpuKernelOnlyMs} | {fmtFloat speedup}x | {checkmark} |\n"

    s := s ++ "\n"

  return s

def generateCsvReport (suites : Array KernelBenchmarkSuite) : String := Id.run do
  let mut lines : Array String := #[]
  lines := lines.push "kernel,input_size,cpu_ms,gpu_total_ms,gpu_kernel_ms,cuda_ms,speedup_vs_cpu,correct"

  for suite in suites do
    for result in suite.results do
      let speedupCpu := result.cpuTimeMs / result.gpuTotalTimeMs
      let cudaMs := match result.cudaReferenceMs with
        | some cuda => fmtFloat cuda
        | none => "N/A"
      lines := lines.push s!"{result.kernelName},{result.inputSize},{fmtFloat result.cpuTimeMs},{fmtFloat result.gpuTotalTimeMs},{fmtFloat result.gpuKernelOnlyMs},{cudaMs},{fmtFloat speedupCpu},{result.correct}"

  return String.intercalate "\n" lines.toList

def generateLatexTable (suites : Array KernelBenchmarkSuite) : String := Id.run do
  let mut s := "\\begin{table}[h]\n"
  s := s ++ "\\centering\n"
  s := s ++ "\\caption{cLean GPU Benchmark Results}\n"
  s := s ++ "\\label{tab:benchmark-results}\n"
  s := s ++ "\\begin{tabular}{lrrrrr}\n"
  s := s ++ "\\toprule\n"
  s := s ++ "Kernel & Input Size & CPU (ms) & GPU (ms) & Speedup & Correct \\\\\n"
  s := s ++ "\\midrule\n"

  for suite in suites do
    for result in suite.results do
      let speedup := result.cpuTimeMs / result.gpuTotalTimeMs
      let check := if result.correct then "Y" else "N"
      s := s ++ s!"{suite.kernelName} & {result.inputSize} & {fmtFloat result.cpuTimeMs} & {fmtFloat result.gpuTotalTimeMs} & {fmtFloat speedup}x & {check} \\\\\n"
    s := s ++ "\\midrule\n"

  s := s ++ "\\bottomrule\n"
  s := s ++ "\\end{tabular}\n"
  s := s ++ "\\end{table}\n"

  return s

/-! ## Main Entry Point -/

def ensureResultsDir : IO Unit := do
  let dir : System.FilePath := "Benchmarks/results"
  let plotsDir : System.FilePath := "Benchmarks/results/plots"
  let dirExists ← dir.pathExists
  unless dirExists do
    IO.FS.createDirAll dir
  let plotsDirExists ← plotsDir.pathExists
  unless plotsDirExists do
    IO.FS.createDirAll plotsDir

def printGpuInfo : IO Unit := do
  IO.println "\n=== GPU Information ==="
  let gpuInfo ← IO.Process.output {
    cmd := "nvidia-smi"
    args := #["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
  }
  if gpuInfo.exitCode == 0 then
    IO.println s!"GPU: {gpuInfo.stdout.trim}"
  else
    IO.println "GPU info not available"

  let (memUsed, memTotal) ← getGpuMemoryInfo
  IO.println s!"Memory: {memUsed} MB / {memTotal} MB"
  IO.println ""

def main : IO Unit := do
  -- Ensure output directories exist
  ensureResultsDir

  -- Print GPU info
  printGpuInfo

  -- Run all benchmarks
  let suites ← runAllKernelBenchmarks

  -- Generate reports
  separator
  IO.println "Generating reports..."

  -- Write JSON report (with detailed timing breakdown)
  exportResultsToJson suites "Benchmarks/results/benchmark_data.json"
  IO.println "  ✓ Benchmarks/results/benchmark_data.json"

  -- Write CSV report (with detailed timing breakdown)
  exportResultsToCsv suites "Benchmarks/results/benchmark_data.csv"
  IO.println "  ✓ Benchmarks/results/benchmark_data.csv"

  -- Write Markdown report
  let mdReport := generateMarkdownReport suites
  IO.FS.writeFile "Benchmarks/results/benchmark_report.md" mdReport
  IO.println "  ✓ Benchmarks/results/benchmark_report.md"

  -- Write LaTeX table
  let latexTable := generateLatexTable suites
  IO.FS.writeFile "Benchmarks/results/benchmark_table.tex" latexTable
  IO.println "  ✓ Benchmarks/results/benchmark_table.tex"

  separator
  IO.println "Benchmark suite complete!"
  IO.println ""
  IO.println "Run 'python3 Benchmarks/visualize.py' to generate plots."

end CLean.Benchmarks

-- Top-level main for executable
def main : IO Unit := CLean.Benchmarks.main
