using DataFrames, HDF5
using LinearAlgebra

function read_wavefunction(L, J, g, t)
  # path to the wavefunction files
  path = "./data/" 
  name = path * "psi_L=$(L).h5"

  file = h5open(name, "r")
  psi  = read(file, "J=$(J)_g=$(g)/t=$(t)")
  return psi
end
