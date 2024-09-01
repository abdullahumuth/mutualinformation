using DataFrames, HDF5
using LinearAlgebra

function read_hdf5_wavefunction(L, J, g, t)
  path = "data/inputs/" 
  name = path * "psi_L=$(L).h5"

  file = h5open(name, "r")
  psi  = read(file, "J=$(J)_g=$(g)/t=$(t)")
  close(file)
  return psi
end
