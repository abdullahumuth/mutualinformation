using NPZ

function read_wavefunction(L, J, g, t)
  path = "data/inputs/" 
  name = path * "ising_critical.npy"
  psi_all = npzread(name)
  return psi_all[Int(t*20), :]
end

