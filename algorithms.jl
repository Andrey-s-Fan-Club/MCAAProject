using StatsBase
using Graphs
using ProgressBars

include("helper.jl")

@inline function overlap(x::Vector{Int8}, x_start::Vector{Int8})
    return abs(mean(x .* x_star))
end


function hamiltonian(x::Vector{Int64}, h::Matrix{Float64})
    
    N = length(x)
    sum = 0.0
    @simd for i = 1:(N-1)
        @simd for j = (i+1):N
            @inbounds sum += h[i, j] * x[i] * x[j]
        end
    end
    return -sum
end


function compute_h(adj::BitMatrix, a::Float64, b::Float64, n::Int64)
    return 0.5 .* (log(a / b) .* adj .+ log((n - a) / (n - b)) .* (1 .- adj))
end


@fastmath @inline function get_h_row(adj::BitMatrix, a::Float64, b::Float64, n::Int64, v::Int64)
    # Broadcast operations with dot
    return 0.5 .* (log(a / b) .* adj[v, :] .+ log((n - a) / (n - b)) .* (1 .- adj[v, :]))
end


# Use views to avoid copies of array[mask] in the return expression
@fastmath @views function compute_acceptance_proba(x::Vector{Int8}, v::Int64, h::Matrix{Float64}, factor::Bool)
    mask = trues(length(x))
    mask[v] = false
    
    # Don't need to do the min with 1 with our implementation
    if factor
        return exp(- 2 * x[v] * sum(x[mask] .* h[v, mask]))
    else
        return exp(- x[v] * sum(x[mask] .* h[v, mask]))
    end
end


# Use views to avoid copies of array[mask] in the return expression
@fastmath @views function compute_acceptance_proba_factor(x::Vector{Int8}, v::Int64, h::Matrix{Float64}, factor::Float64)
    mask = trues(length(x))
    mask[v] = false
    
    # Don't need to do the min with 1 with our implementation
    return exp(- factor * x[v] * sum(x[mask] .* h[v, mask]))
end


@inline function metropolis_step!(h::Matrix{Float64}, cur_x::Vector{Int8}, nb::Int64, factor::Bool)
    
    comp = sample(1:nb)
    
    acc = compute_acceptance_proba(cur_x, comp, h, factor)
    
    if rand(Uniform(0, 1)) <= acc
        cur_x[comp] = -cur_x[comp]
    end
end


@inline function metropolis_step_factor!(h::Matrix{Float64}, cur_x::Vector{Int8}, nb::Int64, factor::Float64)
    
    comp = sample(1:nb)
    
    acc = compute_acceptance_proba_factor(cur_x, comp, h, factor)
    
    if rand(Uniform(0, 1)) <= acc
        cur_x[comp] = -cur_x[comp]
    end
end


function metropolis(h::Matrix{Float64}, nb::Int64, nb_iter::Int64, x_star::Vector{Int8}, a::Float64=nothing, b::Float64=nothing, adj::BitMatrix=nothing, n0::Int64=nothing)
    cur_x = generate_x(nb)
    overlap_vector = Vector{Float64}(undef, nb_iter)
    
    for i = 1:nb_iter
        metropolis_step!(h, cur_x, nb)
        if x_star != nothing
            overlap_vector[i] = overlap(cur_x, x_star)
        end
    end
    
    return cur_x, overlap_vector
end


function metropolis_comp(h::Matrix{Float64}, nb::Int64, nb_iter::Int64, adj::BitMatrix=nothing, a::Float64=nothing, b::Float64=nothing, n0::Int64=nothing)
    cur_x = generate_x(nb)
    for i = 1:nb_iter
        @inbounds metropolis_step!(h, cur_x, nb)
    end
    
    return cur_x
end


@inline function houdayer_step!(cur_x1::Vector{Int8}, cur_x2::Vector{Int8}, adj::BitMatrix, N::Int64)
    y = cur_x1 .* cur_x2
    
    diff_index = findall(y .== -1)
    same_index = findall(y .== 1)
    
    rand_comp = sample(diff_index, 1)
    
    # Set entire row to 0 for all nodes with y = 1
    adj_copy = copy(adj)
    mask = zeros(length(same_index), N)
    adj_copy[same_index, :] = mask
    adj_copy[:, same_index] = transpose(mask)
    
    # Find connected components
    label = zeros(N)
    Graphs.connected_components!(label, Graphs.Graph(adj_copy))
    
    cluster = findall(label .== @view label[rand_comp])
    cur_x1[cluster] = (-1) .* @view cur_x1[cluster]
    cur_x2[cluster] = (-1) .* @view cur_x2[cluster]
end


@inline function houdayer(h::Matrix{Float64}, nb::Int64, nb_iter::Int64, x_star::Vector{Int8}, a::Float64, b::Float64, adj::BitMatrix, arg::Int64=nothing)
    cur_x1 = generate_x(nb)
    cur_x2 = generate_x(nb)
    
    overlap_vector = Vector{Float64}(undef, nb_iter)
    
    for i = 1:nb_iter
        if !all(cur_x1 .== cur_x2)
            houdayer_step!(cur_x1, cur_x2, adj, nb)
        end
        
        metropolis_step!(h, cur_x1, nb)
        metropolis_step!(h, cur_x2, nb)
        
        if x_star != nothing
            overlap_vector[i] = overlap(cur_x1, x_star)
        end
    end
    
    return hamiltonian(cur_x1, h) < hamiltonian(cur_x2, h) ? cur_x1 : cur_x2, overlap_vector
end


@inline function houdayer_mixed(h::Matrix{Float64}, nb::Int64, nb_iter::Int64, x_star::Vector{Int8}, a::Float64, b::Float64, adj::BitMatrix, n0::Int64)
    cur_x1 = generate_x(nb)
    cur_x2 = generate_x(nb)
    
    overlap_vector = Vector{Float64}(undef, nb_iter)
    
    for i = 1:nb_iter
        if mod(i, n0) == 0 && (!all(cur_x1 .== cur_x2))
            houdayer_step!(cur_x1, cur_x2, adj, N)
        end
        
        metropolis_step!(h, cur_x1, nb)
        metropolis_step!(h, cur_x2, nb)
        
        if x_star != nothing
            overlap_vector[i] = overlap(cur_x1, x_star)
        end
    end
    
    return hamiltonian(cur_x1, h) < hamiltonian(cur_x2, h) ? cur_x1 : cur_x2, overlap_vector
end


@inline function houdayer_mixed_comp(h::Matrix{Float64}, nb::Int64, nb_iter::Int64, adj::BitMatrix, a::Float64, b::Float64, n0::Int64)
    
    cur_x1 = generate_x(nb)
    cur_x2 = generate_x(nb)
    
    nb_votes = 25000
    nb_factor = 50000
    
    for i = 1:(nb_iter - nb_factor)
        if mod(i, n0) == 0 && (!all(cur_x1 .== cur_x2))
            houdayer_step!(cur_x1, cur_x2, adj, N)
        end
        
        metropolis_step_factor!(h, cur_x1, nb, 1.0)
        metropolis_step_factor!(h, cur_x2, nb, 1.0)
        
    end
    
    for i = (nb_iter - nb_factor + 1):(nb_iter - nb_votes)
        if mod(i, n0) == 0 && (!all(cur_x1 .== cur_x2))
            houdayer_step!(cur_x1, cur_x2, adj, N)
        end
        
        metropolis_step_factor!(h, cur_x1, nb, 2.0)
        metropolis_step_factor!(h, cur_x2, nb, 2.0)
        
    end
    
    votes1 = Matrix{Int8}(undef, nb, nb_votes)
    votes2 = Matrix{Int8}(undef, nb, nb_votes)
    
    for (idx, i) = enumerate((nb_iter-nb_votes+1):nb_iter)
        if mod(i, n0) == 0 && (!all(cur_x1 .== cur_x2))
            houdayer_step!(cur_x1, cur_x2, adj, N)
        end
        
        metropolis_step_factor!(h, cur_x1, nb, 2.0)
        metropolis_step_factor!(h, cur_x2, nb, 2.0)
        
        votes1[:, idx] = cur_x1
        votes2[:, idx] = cur_x2
    end
    
    cur_x1 = majority_vote(votes1)
    cur_x2 = majority_vote(votes2)
    
    return hamiltonian(cur_x1, h) < hamiltonian(cur_x2, h) ? cur_x1 : cur_x2
    
end


function run_experiment(nb::Int64, a::Float64, b::Float64, x_star::Vector{Int8}, algorithm::Function, nb_iter::Int64=1000, nb_exp::Int64=100, n0::Int64=0)
    overlaps = zeros(nb_iter, nb_exp)

    Threads.@threads for j = 1:nb_exp
        if x_star != nothing
            adj = generate_graph(x_star, a, b)
        end
            
        h = compute_h(adj, a, b, nb)        
        new_x, overlap_list = algorithm(h, nb, nb_iter, x_star, a, b, adj, n0)
        
        overlaps[:, j] = overlap_list
    end
    
    return mean(overlaps, dims=2)
end


function competition(adj::BitMatrix, a::Float64, b::Float64, nb_iter::Int64, nb_exp::Int64, nb::Int64, n0::Int64)
    
    x_hat = Matrix{Int8}(undef, nb, nb_exp)
    hamiltonians = Vector{Float64}(undef, nb_exp)
    h = compute_h(adj, a, b, nb)
    
    limit = ceil(Int64, ratio_factor*nb_iter)
    
    Threads.@threads for i = eachindex(x_hat[1, :])
        @inbounds x_hat[:, i] = houdayer_mixed_comp(h, nb, nb_iter, adj, a, b, n0, limit)
        hamiltonians[i] = hamiltonian(x_hat[:, i], h)
    end
    
    # Return x estimate with the lowest energy
    min_ham_idx = argmin(hamiltonians)
    print("Minimum Hamiltonian among $(nb_exp) experiments : $(hamiltonians[min_ham_idx])")
    return x_hat[:, min_ham_idx]
end


function overlap_r(x_star::Vector{Int8}, algorithm::Function, nb::Int64, nb_iter::Int64, nb_exp::Int64, d::Int64=3, n0::Int64=0, nb_r::Int64=10)
    range_r = exp10.(range(-3, 0, nb_r))
    
    overlap_r = zeros(nb_r)
    
    Threads.@threads for i = 1:nb_r
        a = 2 * d / (range_r[i] + 1)
        b = range_r[i] * a
        overlap = run_experiment(nb, a, b, x_star, algorithm, nb_iter, nb_exp, n0)
        overlap_r[i] = last(overlap[1, :])
    end
    
    return overlap_r, range_r
end

    