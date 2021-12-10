using StatsBase
using Distributions
using Plots

function generate_x(size::Integer)
    return sample(Vector{Int8}([-1, 1]), size)
end

function generate_graph(x::Vector{Int8}, a::Real, b::Real)
    n = length(x)
    adjacency_mat = falses(n, n)
    
    b_n = b / n
    a_n = a / n
        
    # Indexing start at 1 and n is included
    Threads.@threads for i = 1:n
        for j = (i+1):n
            prod = x[i] * x[j]
            
            if prod == -1
                value = rand(Uniform(0, 1)) < b_n
            else
                value = rand(Uniform(0, 1)) < a_n
            end
             
            adjacency_mat[i, j] = value
            adjacency_mat[j, i] = value
            
        end
    end
        
    return adjacency_mat
end
            
function visualize_overlap(overlap_array::Vector{Float64}, nb_exp::Integer)
    plot(overlap_array,
        title="Average overlap over $(nb_exp) experiments",
        xlabel="Iterations of the MC",
        ylabel="Avg overlap")
end

                