using StatsBase
using Combinatorics

include("helper.jl")

function overlap(x::Vector{Int8}, x_start::Vector{Int8})
    return abs(mean(x .* x_star))
end


function get_h_row(adj::BitMatrix, a::Integer, b::Integer, n::Integer, v::Integer)
    # Broadcast operations with dot
    return 0.5 .* (log(a / b) .* adj[v, :] .+ log((n - a) / (n - b)) .* (1 .- adj[v, :]))
end


function compute_acceptance_proba(x::Vector{Int8}, v::Integer, n::Integer, adj::BitMatrix, a::Integer, b::Integer)
    mask = trues(length(x))
    mask[v] = false
    
    h_row_masked = get_h_row(adj, a, b, n, v)
    return - x[v] * sum(x[mask] .* h_row_masked[mask])
end


function metropolis_step(adj::BitMatrix, a::Integer, b::Integer, cur_x::Vector{Int8}, nb::Integer)
    comp = sample(1:nb)
    proposed_move = copy(cur_x)
    proposed_move[comp] = -proposed_move[comp]
    
    acc = compute_acceptance_proba(cur_x, comp, nb, adj, a, b)
    
    if rand(Uniform(0, 1)) <= acc
        return proposed_move
    else
        return cur_x
    end
end


function metropolis(adj::BitMatrix, a::Integer, b::Integer, nb::Integer, nb_iter::Integer, x_star::Vector{Int8}, arg=nothing)
    cur_x = generate_x(nb)
    overlap_vector = Vector{Float64}(undef, nb_iter)
    
    for i = 1:nb_iter
        cur_x = metropolis_step(adj, a, b, cur_x, nb)
        if x_star != nothing
            overlap_vector[i] = overlap(cur_x, x_star)
        end
    end
    
    return cur_x, overlap_vector
end


function houdayer_step(cur_x1::Vector{Int8}, cur_x2::Vector{Int8})
    y = cur_x1 .* cur_x2
    
    diff_index = findall(y .== -1)
    all_pairs = combinations(diff_index, 2)
    
    
    

function run_experiment(nb::Integer, a::Integer, b::Integer, x_star::Vector{Int8}, algorithm::Function, nb_iter::Integer=1000, nb_exp::Integer=100, n0::Integer=0)
    overlaps = zeros(nb_exp, nb_iter)

    Threads.@threads for j = 1:nb_exp
        if x_star != nothing
            adj = generate_graph(x_star, a, b)
        end
        
        new_x, overlap_list = algorithm(adj, a, b, nb, nb_iter, x_star, n0)
        
        overlaps[j, :] = overlap_list
    end
    
    return mean(overlaps, dims=1)
end
    