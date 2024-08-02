using CSV, DataFrames, HDF5
using LinearAlgebra
using SparseArrays
using Statistics
using CairoMakie

include("TFIexact.jl")
include("gen_samples.jl")

let
  L = 12
  J = -1

  spin_basis = vec(collect(Iterators.product(fill([1,0],L)...)));

  gs = [-0.5,-1.0,-2.0]
  ts = collect(0:0.001:1)

  name = "/Users/wladi/Projects/MutualInformation/wavefunctions/psi_L=$(L).h5"
  h5open(name, "w") do file
    for g in gs
      psis = analytical_timeEv((L,J,g), ts)
      for (t,psi) in zip(ts,psis)
        file["J=$(J)_g=$(g)/t=$(t)"] = psi
      end
    end
  end
end
