using CSV, DataFrames, HDF5
using LinearAlgebra
using SparseArrays
using Statistics
using CairoMakie
using Integrals

include("TFIexact.jl")
include("gen_samples.jl")

function gaussian_kernel(x, mu; sigma)
  return exp(-sum(abs2, x-mu)/(2*sigma^2)) / sqrt(2*pi*sigma^2)
end

function p_spin(spin_sample, psi)
  return abs2(psi[sample_to_int(spin_sample)])
end

function p_psi_cond_s(psi_sample, spin_sample, psi, spin_basis; sigma)
  return gaussian_kernel(psi_sample, psi[sample_to_int(spin_sample)]; sigma)
end

function p_spin_psi(psi_sample, spin_sample, psi, spin_basis; sigma)
  return p_psi_cond_s(psi_sample, spin_sample, psi, spin_basis; sigma) * p_spin(spin_sample, psi)
end

function shannonEntropy(psi, spin_basis)
  return -mapreduce(+, psi) do p
    return p * log2(p)
  end
end

function shannonEntropy_spin(spin_basis, psi; sigma)

  return mapreduce(+, spin_basis) do spin
    isapprox(0.0, p_spin(spin, psi); atol=1e-10) && return 0

    res = p_spin(spin, psi) * log2(p_spin(spin, psi))

    return -res
  end
end
           
function mutualInformation(spin_basis, psi; sigma, abstol = 1e-6, reltol = 1e-6)
  domain = ((-1,-1), (1,1))

  norm_p_spin_psi = mapreduce(+, spin_basis) do spin
    psiIntegral = IntegralProblem((x,n) -> p_spin_psi(x[1]+1im*x[2], spin, psi, spin_basis; sigma), domain)
    res = solve(psiIntegral, HCubatureJL(); reltol, abstol)
    return res.u
  end
 
  return mapreduce(+, spin_basis) do spin
    function func(x,n)
      p = x[1]+1im*x[2]
      if isapprox(0.0, p_spin_psi(p, spin, psi, spin_basis; sigma); atol=1e-10)
        return 0
      end
      return p_spin_psi(p, spin, psi, spin_basis; sigma)/norm_p_spin_psi * log2(p_spin_psi(p, spin, psi, spin_basis; sigma)/norm_p_spin_psi / (p_spin(spin, psi) * sum([p_spin_psi(p, s, psi, spin_basis; sigma)/norm_p_spin_psi for s in spin_basis])))
    end

    psiIntegral = IntegralProblem((x,n) -> func(x,n), domain)
    res = solve(psiIntegral, HCubatureJL(); reltol, abstol)
    # res = p_spin_psi(p, spin, psi, spin_basis; sigma)/norm_p_spin_psi * log(p_spin_psi(p, spin, psi, spin_basis; sigma)/norm_p_spin_psi / (p_spin(spin, psi) * sum([p_spin_psi(p, s, psi, spin_basis; sigma)/norm_p_spin_psi for s in spin_basis])))

    return res.u
  end
end

function mutualInformationRadial(spin_basis, psi; sigma, abstol = 1e-6, reltol = 1e-6)
  domain = ((0,0), (1,2*pi))

  norm_p_spin_psi = mapreduce(+, spin_basis) do spin
    psiIntegral = IntegralProblem((x,n) -> x[1] * p_spin_psi(x[1]*exp(1im*x[2]), spin, psi, spin_basis; sigma), domain)
    res = solve(psiIntegral, HCubatureJL(); reltol, abstol)
    return res.u
  end
 
  return mapreduce(+, spin_basis) do spin
    function func(x,n)
      p = x[1]*exp(1im*x[2])
      if isapprox(0.0, p_spin_psi(p, spin, psi, spin_basis; sigma); atol=1e-10)
        return 0
      end
      return x[1] * p_spin_psi(p, spin, psi, spin_basis; sigma)/norm_p_spin_psi * log2(p_spin_psi(p, spin, psi, spin_basis; sigma)/norm_p_spin_psi / (p_spin(spin, psi) * sum([p_spin_psi(p, s, psi, spin_basis; sigma)/norm_p_spin_psi for s in spin_basis])))
    end

    psiIntegral = IntegralProblem((x,n) -> func(x,n), domain)
    res = solve(psiIntegral, HCubatureJL(); reltol, abstol)
    # res = p_spin_psi(p, spin, psi, spin_basis; sigma)/norm_p_spin_psi * log(p_spin_psi(p, spin, psi, spin_basis; sigma)/norm_p_spin_psi / (p_spin(spin, psi) * sum([p_spin_psi(p, s, psi, spin_basis; sigma)/norm_p_spin_psi for s in spin_basis])))

    return res.u
  end
end


let
  L = 5
  J = -1.0
  g = -1.0

  fig = Figure()
  ax = Axis(fig[1,1]; title = "L = $L, J = $J, g=$g", xlabel = "time", ylabel = L"mutual Information $I$",
  )

  spin_basis = vec(collect(Iterators.product(fill([1,0],L)...)));

  ts = collect(0:0.01:1)
  psis = analytical_timeEv((L,J,g), ts)

  sigmas = [1e-1, 5e-2]

  for sigma in sigmas
    data = map(psis) do psi
      # return shannonEntropy_spin(spin_basis, psi; sigma)
      return mutualInformationRadial(spin_basis, psi; sigma, abstol = 1e-6, reltol = 1e-6)
    end

    lines!(ax, ts, abs.(data); label = "sigma = $sigma")
  end
  # hlines!(ax, [L*log2(2)], label="H = L log(2) = $(round(L*log2(2), digits=1))")
  Legend(fig[1,2], ax)
  fig
end

