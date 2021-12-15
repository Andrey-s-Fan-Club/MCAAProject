using StatsBase
using Graphs

include("helper.jl")

@inline function overlap(x::Vector{Int8}, x_start::Vector{Int8})
    return abs(mean(x .* x_star))
end


@fastmath @inline function get_h_row(adj::BitMatrix, a::Float64, b::Float64, n::Int64, v::Int64)
    # Broadcast operations with dot
    return 0.5 .* (log(a / b) .* adj[v, :] .+ log((n - a) / (n - b)) .* (1 .- adj[v, :]))
end


# Use views to avoid copies of array[mask] in the return expression
@fastmath @views function compute_acceptance_proba(x::Vector{Int8}, v::Int64, n::Int64, adj::BitMatrix, a::Float64, b::Float64)
    mask = trues(length(x))
    mask[v] = false
    
    return min(1, exp(- x[v] * sum(x[mask] .* get_h_row(adj, a, b, n, v)[mask])))
end


@inline function metropolis_step!(adj::BitMatrix, a::Float64, b::Float64, cur_x::Vector{Int8}, nb::Int64)
    
    comp = sample(1:nb)
    
    acc = compute_acceptance_proba(cur_x, comp, nb, adj, a, b)
    
    if rand(Uniform(0, 1)) <= acc
        cur_x[comp] = -cur_x[comp]
    end
end


function metropolis(adj::BitMatrix, a::Float64, b::Float64, nb::Int64, nb_iter::Int64, x_star::Vector{Int8}, arg=nothing)
    cur_x = generate_x(nb)
    overlap_vector = Vector{Float64}(undef, nb_iter)
    
    for i = 1:nb_iter
        metropolis_step!(adj, a, b, cur_x, nb)
        if x_star != nothing
            overlap_vector[i] = overlap(cur_x, x_star)
        end
    end
    
    return cur_x, overlap_vector
end


function metropolis_comp(adj::BitMatrix, a::Float64, b::Float64, nb::Int64, nb_iter::Int64)
    cur_x = generate_x(nb)
    for i = 1:nb_iter
        metropolis_step!(adj, a, b, cur_x, nb)
    end
    
    return cur_x
end


@inline function houdayer_step!(cur_x1::Vector{Int8}, cur_x2::Vector{Int8}, adj::BitMatrix)
    y = cur_x1 .* cur_x2
    
    diff_index = findall(y .== -1)
    same_index = findall(y .== 1)
    
    rand_comp = sample(diff_index, 1)
    
    # Set entire row to 0 for all nodes with y = 1
    adj_copy = copy(adj)
    N = length(cur_x1)
    mask = zeros(length(same_index), N)
    adj_copy[same_index, :] = mask
    adj_copy[:, same_index] = transpose(mask)
    
    observed_graph = Graphs.Graph(adj_copy)
    
    # Find connected components
    label = zeros(N)
    label = Graphs.connected_components!(label, observed_graph)
    label_rand_comp = @view label[rand_comp]
    
    cluster = findall(label .== label_rand_comp)
    cur_x1[cluster] = (-1) .* @view cur_x1[cluster]
    cur_x2[cluster] = (-1) .* @view cur_x2[cluster]
end


@inline function houdayer(adj::BitMatrix, a::Float64, b::Float64, nb::Int64, nb_iter::Int64, x_star::Vector{Int8}, arg=nothing)
    cur_x1 = generate_x(nb)
    cur_x2 = generate_x(nb)
    
    overlap_vector = Vector{Float64}(undef, nb_iter)
    
    for i = 1:nb_iter
        if !all(cur_x1 .== cur_x2)
            houdayer_step!(cur_x1, cur_x2, adj)
        end
        
        metropolis_step!(adj, a, b, cur_x1, nb)
        metropolis_step!(adj, a, b, cur_x2, nb)
        
        if x_star != nothing
            overlap_vector[i] = overlap(cur_x1, x_star)
        end
    end
    
    return cur_x1, overlap_vector
end


@inline function houdayer_mixed(adj::BitMatrix, a::Float64, b::Float64, nb::Int64, nb_iter::Int64, x_star::Vector{Int8}, n0::Int64)
    cur_x1 = generate_x(nb)
    cur_x2 = generate_x(nb)
    
    overlap_vector = Vector{Float64}(undef, nb_iter)
    
    for i = 1:nb_iter
        if mod(i, n0) == 0 && (!all(cur_x1 .== cur_x2))
            houdayer_step!(cur_x1, cur_x2, adj)
        end
        
        metropolis_step!(adj, a, b, cur_x1, nb)
        metropolis_step!(adj, a, b, cur_x2, nb)
        
        if x_star != nothing
            overlap_vector[i] = overlap(cur_x1, x_star)
        end
    end
    
    return cur_x1, overlap_vector
end


@inline function houdayer_custom(adj::BitMatrix, a::Float64, b::Float64, nb::Int64, nb_iter::Int64, x_star::Vector{Int8}, steps::Dict{Float64, Int64})
    # Decompose into different for loops for the different n0
    
    return 1
    
end


@inline function houdayer_mixed_comp(adj::BitMatrix, a::Float64, b::Float64, nb::Int64, nb_iter::Int64, n0::Int64)
    
    cur_x1 = generate_x(nb)
    cur_x2 = generate_x(nb)
       
    for i = 1:nb_iter
        if mod(i, n0) == 0 && (!all(cur_x1 .== cur_x2))
            houdayer_step!(cur_x1, cur_x2, adj)
        end
        
        metropolis_step!(adj, a, b, cur_x1, nb)
        metropolis_step!(adj, a, b, cur_x2, nb)
        
    end
    
    return cur_x1
    
end


function run_experiment(nb::Int64, a::Float64, b::Float64, x_star::Vector{Int8}, algorithm::Function, nb_iter::Int64=1000, nb_exp::Int64=100, n0::Int64=0)
    overlaps = zeros(nb_exp, nb_iter)

    Threads.@threads for j = eachindex(overlaps[1, :])
        if x_star != nothing
            adj = generate_graph(x_star, a, b)
        end
        
        new_x, overlap_list = algorithm(adj, a, b, nb, nb_iter, x_star, n0)
        
        overlaps[j, :] = overlap_list
    end
    
    return mean(overlaps, dims=1)
end


function competition(adj::BitMatrix, a::Float64, b::Float64, nb_iter::Int64, nb_exp::Int64, nb::Int64)
    x_hat = Matrix{Int8}(undef, nb, nb_exp)
    
    Threads.@threads for i = eachindex(x_hat[1, :])
        @inbounds x_hat[:, i] = metropolis_comp(adj, a, b, nb, nb_iter)
    end
    
    return x_hat
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

    