## spinflips
function spinflip(spin::Tuple{Vararg{Int64}}, pos::Tuple{Vararg{Int64}})
    res =  map(enumerate(spin)) do (n,s)
        return n ∈ pos ? (s+1)%2 : s
    end
    return Tuple(res)
end

function bits_to_int(conf)
    pos = mapreduce(+, enumerate(conf)) do (l,s)
        return s*2^(l-1)
    end
    return pos + 1
end

### Transverse field Ising model ###
function build_H(spin_basis, param)
    (L,J,g) = param

    rows = Vector{Int}(); 
    columns = Vector{Int}(); 
    values = Vector{Float64}();
    
    ## diagonal part of TFI
for spin in spin_basis
        n1 = bits_to_int(spin)
        
        #zz_term
        # zzTerm = mapreduce(+, zip(collect(1:L), vcat(collect(2:L), 1))) do (i,j) ##periodic
        zzTerm = mapreduce(+, zip(collect(1:L-1), collect(2:L))) do (i,j)
            return J*(2*spin[i]-1)*(2*spin[j]-1)
        end

        append!(rows, [n1])
        append!(columns, [n1])
        append!(values, [zzTerm])

        for pos in 1:prod(L)
            newSpin = spinflip(spin, (pos,))
            n2 = bits_to_int(newSpin)

            append!(rows, [n1])
            append!(columns, [n2])
            append!(values, [g])
        end
    end
    
    return dropzeros(sparse(rows,columns,values))
end;

function build_xPol(spin_basis, param)
    (L,_,_) = param

    rows = Vector{Int}(); 
    columns = Vector{Int}(); 
    values = Vector{Float64}();
    
    ## diagonal part of TFI
    for spin in spin_basis
        n1 = bits_to_int(spin)

        for pos in 1:prod(L)
            newSpin = spinflip(spin, (pos,))
            n2 = bits_to_int(newSpin)

            append!(rows, [n1])
            append!(columns, [n2])
            append!(values, [1/L])
        end
    end
    
    return dropzeros(sparse(rows,columns,values))
end;

function analytical_timeEv(params, ts)
    (L,J,g) = params

    spin_basis = vec(collect(Iterators.product(fill([1,0],L)...)));

    zPol_precalc = map(spin_basis) do spin
        return mean(2 .* spin .- 1)
    end

    xPolM = build_xPol(spin_basis, (L,0,0))

    H = build_H(spin_basis, (L,J,g))

    ## determine starting vector
    H_x = build_H(spin_basis, (L,0,g))
    valsx, vecsx = eigen(Matrix(H_x))

    psi = vecsx[:,1]
    psi_init = psi
    vals, vecs = eigen(Matrix(H))
    data0 = [0, dot(psi, xPolM*psi)]

    psi = Transpose(vecs) * psi

    psis = map(zip(ts[1:end-1], ts[2:end])) do (t, tf)
        dt = tf-t

        # Propagate state
        U = exp.(-1im*dt .* vals)
        psi = U .* psi

        return vecs*psi
    end
    # data = map(zip(ts[1:end-1], ts[2:end])) do (t, tf)
    #     dt = tf-t
    #
    #     # Propagate state
    #     U = exp.(-1im*dt .* vals)
    #     psi = U .* psi
    #
    #     psi_prime = vecs*psi
    #     push!(psis, psi_prime)
    #     xPol = dot(psi_prime, xPolM*psi_prime) 
    #     return [real(t), real(xPol)]
    # end

    return pushfirst!(psis, psi_init)
end;

