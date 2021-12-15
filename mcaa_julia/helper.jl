using StatsBase
using Distributions
using Plots
using Clustering


function generate_x(size::Int64)
    return sample(Vector{Int8}([-1, 1]), size)
end


function generate_graph(x::Vector{Int8}, a::Float64, b::Float64)
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


function visualize_overlap(
        overlap_array_100::Vector{Float64},
        overlap_array_500::Vector{Float64},
        overlap_array_1000::Vector{Float64},
        overlap_array_5000::Vector{Float64},
        nb_exp::Integer, x_vlines::Vector{Int64}, a::Real, b::Real, n0::Integer=0)
    
    if n0 !== 0
        title = "Average overlap over $(nb_exp) experiments\na=$(a), b=$(b), n0=$(n0)"
    else
        title = "Average overlap over $(nb_exp) experiments\na=$(a), b=$(b)"
    end
    
    plot(overlap_array_100,
        title=title,
        xlabel="Iterations of the MC",
        ylabel="Avg overlap",
        label="N=100"
    )
    
    plot!(overlap_array_500,
        label="N=500"
    )
    
    plot!(overlap_array_1000,
        label="N=1000"
    )
    
    plot!(overlap_array_5000,
        label="N=5000"
    )
    
    # Plots vertical lines at different values
    vline!(x_vlines, label="Thresholds")
    plot!(size=(1000, 600), legend=:bottomright, margin=5Plots.mm)
end


function plot_overlap_r(
        overlap_r_100::Vector{Float64},
        overlap_r_500::Vector{Float64},
        overlap_r_1000::Vector{Float64},
        overlap_r_5000::Vector{Float64},
        range_r::Vector{Float64},
        d::Real)
    
    # Theoretical phase transition (see computations)
    r_c = (sqrt(d) - 1) / (sqrt(d) + 1)
    
    plot(range_r, overlap_r_100, xaxis=:log,
        title="Average overlap over $(nb_exp) experiments",
        xlabel="r",
        ylabel="Avg overlap",
        label="N=100"
    )
    
    plot!(range_r, overlap_r_500, xaxis=:log,
        title="Average overlap over $(nb_exp) experiments",
        xlabel="r",
        ylabel="Avg overlap",
        label="N=500"
    )
    
    plot!(range_r, overlap_r_1000, xaxis=:log,
        title="Average overlap over $(nb_exp) experiments",
        xlabel="r",
        ylabel="Avg overlap",
        label="N=1000"
    )
    
    plot!(range_r, overlap_r_5000, xaxis=:log,
        title="Average overlap over $(nb_exp) experiments",
        xlabel="r",
        ylabel="Avg overlap",
        label="N=5000"
    )
    
    vline!([r_c], xaxis=:log, label="Theoretical transition")
    plot!(size=(1000, 600), legend=:bottomleft, margin=5Plots.mm)
end


function visualize_n0(
        overlap_array_100::Vector{Float64},
        overlap_array_1000::Vector{Float64},
        overlap_array_5000::Vector{Float64},
        overlap_array_10000::Vector{Float64},
        overlap_array_20000::Vector{Float64},
        nb_exp::Integer, x_vlines::Vector{Int64}, a::Real, b::Real)
    
    plot(overlap_array_100,
        title="Average overlap over $(nb_exp) experiments\na=$(a), b=$(b)",
        xlabel="Iterations of the MC",
        ylabel="Avg overlap",
        label="n0=100"
    )
    
    plot!(overlap_array_1000,
        label="n0=1000"
    )
    
    plot!(overlap_array_5000,
        label="n0=5000"
    )
    
    plot!(overlap_array_10000,
        label="n0=10'000"
    )
    
    plot!(overlap_array_20000,
        label="n0=20'000"
    )
    
    # Plots vertical lines at different values
    vline!(x_vlines, label="Threshold")
    plot!(size=(1000, 600), legend=:bottomright, margin=5Plots.mm)
    
end


function kmeans_clustering(x_hat_matrix::Matrix{Int8})
    # Group into 2 cluters
    res = kmeans(x_hat_matrix, 2)
    
    # Get index of points assigned to second cluster
    cluster_index = findall(assignments(res) .== 2)
    
    # Flip the sign of these clusters
    x_hat_matrix[:, cluster_index] = - x_hat_matrix[:, cluster_index]
    
    return x_hat_matrix
end


function majority_vote(x_hat_matrix::Matrix{Int8})
    means = mean(x_hat_matrix, dims=2)
    for i = 1:length(means)
        if means[i] >= 0
            means[i] = 1
        else
            means[i] = -1
        end
    end
    
    return means
end
    
