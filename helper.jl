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


function choose_x(cur_x1::Vector{Int64}, cur_x2::Vector{Int64})
    
    mean1 = abs(sum(cur_x1))
    mean2 = abs(sum(cur_x2))
    
    if mean1 < mean2
        return cur_x1
    else
        return cur_x2
    end    
end


function visualize_overlap(
        overlap_array_1::Vector{Float64},
        overlap_array_2::Vector{Float64},
        overlap_array_3::Vector{Float64},
        overlap_array_4::Vector{Float64},
        nb_exp::Integer, x_vlines::Vector{Int64}, a::Real, b::Real, names::Vector{String}, algorithm::String, n0::Integer=0)
    
    if n0 !== 0
        title = "Average overlap over $(nb_exp) experiments\na=$(a), b=$(b), n0=$(n0) "
    else
        title = "Average overlap over $(nb_exp) experiments\na=$(a), b=$(b)"
    end
    
    if length(algorithm)>0
        title*="("*algorithm*")"
    end
    
    plot(overlap_array_1,
        title=title,
        xlabel="Iterations of the MC",
        ylabel="Avg overlap",
        label="N="*names[1]
    )
    if length(overlap_array_2)>0
        plot!(overlap_array_2,
            label="N="*names[2]
        )
    end
    if length(overlap_array_3)>0
        plot!(overlap_array_3,
            label="N="*names[3]
        )
    end
    if length(overlap_array_4)>0
        plot!(overlap_array_4,
            label="N="*names[4]
        )
    end
    
    # Plots vertical lines at different values
    if length(x_vlines)>0
        vline!(x_vlines, label="Thresholds")
    end
    plot!(size=(1000, 600), legend=:bottomright, margin=5Plots.mm)
end


function plot_overlap_r(
        overlap_r_1::Vector{Float64},
        overlap_r_2::Vector{Float64},
        overlap_r_3::Vector{Float64},
        overlap_r_4::Vector{Float64},
        range_r::Vector{Float64},
        d::Real, 
        names::Vector{String}, 
        algorithm::String)
    
    # Theoretical phase transition (see computations)
    r_c = (sqrt(d) - 1) / (sqrt(d) + 1)
    
    sup_title=""
    if length(algorithm)>0
        sup_title="("*algorithm*")"
    end
    
    plot(range_r, overlap_r_1, xaxis=:log,
        title="Average overlap over $(nb_exp) experiments "*sup_title,
        xlabel="r",
        ylabel="Avg overlap",
        label="N="*names[1]
    )
    
    if length(overlap_r_2)>0
        plot!(range_r, overlap_r_2, xaxis=:log,
        label="N="*names[2]
    )
    end
    
    if length(overlap_r_3)>0
        plot!(range_r, overlap_r_3, xaxis=:log,
        label="N="*names[3]
        )
    end
    
    if length(overlap_r_4)>0
        plot!(range_r, overlap_r_4, xaxis=:log,
        label="N="*names[4]
    )
    end
    
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


function plot_comparisons(
        overlap_metropolis_500::Vector{Float64},
        overlap_houdayer_500::Vector{Float64},
        overlap_houdayer_mixed_500::Vector{Float64},
        nb_exp::Int64)
    
    plot(overlap_houdayer_500,
        title="Average overlap over $(nb_exp) experiments, N=500",
        xlabel="Iterations of the MC",
        ylabel="Avg overlap",
        label="Houdayer",
        color=:dodgerblue
    )
    
    plot!(overlap_metropolis_500,
        label="Metropolis",
        color=:red
    )
    
    plot!(overlap_houdayer_mixed_500,
        label="Houdayer Mixed",
        color=:purple
    )
    
    plot!(size=(1000, 600), legend=:bottomright, margin=5Plots.mm)
end


function majority_vote(x_hat_matrix::Matrix{Int8})
    sums = sum(x_hat_matrix, dims=2)
    for i = 1:length(sums)
        if sums[i] >= 0
            sums[i] = 1
        else
            sums[i] = -1
        end
    end
    
    return sums[:, 1]
end
    
