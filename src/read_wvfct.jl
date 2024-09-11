using DataFrames, HDF5
using LinearAlgebra
using NPZ
using Random

function read_hdf5_wavefunction(L, J, g, t)
  path = "data/inputs/" 
  name = path * "psi_L=$(L).h5"

  file = h5open(name, "r")
  psi  = read(file, "J=$(J)_g=$(g)/t=$(t)")
  close(file)
  return psi
end


function read_wavefunction(L, J, g, t)
  path = "data/inputs/" 
  name = path * "ising_critical.npy"
  psi_all = npzread(name)
  return psi_all[Int(t*20)+1, :]
end

