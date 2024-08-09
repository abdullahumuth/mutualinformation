using NPZ
using Random
function read_wavefunction(L, J, g, t)
  path = "data/inputs/" 
  name = path * "ising_critical.npy"
  psi_all = npzread(name)
  return psi_all[Int(t*20)+1, :]
end


function fake_y(n; unit=1, offset=0, partition=1)
  f(x) = [div((x-1),sqrt(2^n))*unit,((x-1) % sqrt(2^n))*unit] .+ unit*(-2^(n/2-1) .+ 1/2)
  shuffled = Random.randperm(MersenneTwister(111), 2^n)
  if partition == 1
    straight = stack(map(f, 1:2^n))
    g = straight[:,shuffled]
  elseif partition == 2
    h1 = stack(map(f, 1:2^(n-1))) .+ [offset, 0]
    h2 = stack(map(f, 1:2^(n-1))) .+ [-offset, 0]
    g = hcat(h1, h2)[:, shuffled]
  elseif partition == 4
    h1 = stack(map(f, 1:2^(n-2))) .+ [offset, offset]
    h2 = stack(map(f, 1:2^(n-2))) .+ [-offset, offset]
    h3 = stack(map(f, 1:2^(n-2))) .+ [offset, -offset]
    h4 = stack(map(f, 1:2^(n-2))) .+ [-offset, -offset]
    g = hcat(h1, h2, h3, h4)[:, shuffled]
  end

  g = g./sqrt(sum((x)->x^2,g))
  return g
end