function plot_θ_comparison(timesθ, θ, θ_approx, n)

  layout = Layout(title=attr(text="ε = 10⁻⁵",x=0.035,y=0.975,font_size=30),
    xaxis_title="time [s]", legend=attr(x=0.375,y=1.05,yanchor="top",orientation="h"))

  θ_plt = hcat(θ, θ_approx)
  sel_colors = ["black","blue"]
  sel_dashes = ["","dash"]
  names = ["θᵃ", "θᵃ approx"]
  traces = [scatter(x=timesθ,y=θ_plt[:,i],name=names[i],
    line=attr(width=4,color=sel_colors[i],dash=sel_dashes[i])) for i=1:size(θ_plt)[2]]
  p = plot(traces,layout)
  display(p)
  #savefig(p, joinpath("/home/user1/git_repos/NicholasPhD/Mabla_results/cube20x20x20.json/plots/θᵃ$n.eps"))

end

function make_θ_plot(n::Int)
  timesθ = get_timesθ(FEMInfo)
  pp = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/case3/cube20x20x20.json/ST-GRB_st_-$n/ROM_structures/θᵃ.csv"
  θᵃapp = load_CSV(Vector{Float}(undef,0),pp)
  θᵃapp = reshape(θᵃapp,:,50)
  pp = "/home/user1/git_repos/Mabla.jl/tests/unsteady/poisson/case3/cube20x20x20.json/ST-GRB_-$n/ROM_structures/θᵃ.csv"
  θᵃ = load_CSV(Vector{Float}(undef,0),pp)
  θᵃ = reshape(θᵃ,:,50)
  plot_θ_comparison(timesθ, θᵃ[1,:], θᵃapp[1,:], n)
end

function interpolated_θ_BSpline(
  RBVars::PoissonUnsteady{T},
  Mat_μ_sparse::SparseMatrixCSC{T, Int},
  timesθ::Vector{T},
  MDEIMᵢ::Matrix{T},
  MDEIM_idx::Vector{Int},
  MDEIM_idx_time::Vector{Int},
  Q::Int) where T

  red_timesθ = timesθ[MDEIM_idx_time]
  discarded_idx_time = setdiff(collect(1:RBVars.Nₜ), MDEIM_idx_time)
  θ = zeros(T, Q, RBVars.Nₜ)

  red_θ = (MDEIMᵢ \
    Matrix{T}(reshape(Mat_μ_sparse, :, length(red_timesθ))[MDEIM_idx, :]))
  θ[:, MDEIM_idx_time] = red_θ

  for q = 1:Q
    etp = BSplineInterpolation(red_θ[q,:], red_timesθ, 2, :Uniform, :Uniform)

    for iₜ = discarded_idx_time
      θ[q, iₜ] = etp(timesθ[iₜ])
    end
  end

  θ::Matrix{T}

end

function modify_timesθ_and_MDEIM_idx(
  MDEIM_idx::Vector{Int},
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady)

  timesθ = get_timesθ(RBInfo)
  idx_space, idx_time = from_vec_to_mat_idx(MDEIM_idx, RBVars.Nₛᵘ^2)
  idx_time_mod = label_sorted_elems(idx_time)
  timesθ_mod = timesθ[unique(sort(idx_time))]
  MDEIM_idx_mod = (idx_time_mod .- 1) * RBVars.Nₛᵘ^2 + idx_space
  timesθ_mod, MDEIM_idx_mod
end

function my_unique(v::Vector)
  u = unique(v)
  idx = Int.(indexin(u,v))
  return u,idx
end

function my_sort(v::Vector)
  s = sort(v)
  idx = Int.(indexin(s,v))
  return s,idx
end

function find_idx_duplicates(v::Vector)
  u,idx = my_unique(v)
  v_dupl = unique(v[setdiff(collect(1:length(v)),idx)])
  idx_dupl = [findall(x->x.==vi,v) for vi=v_dupl]
  return v_dupl,idx_dupl
end

function subtract_idx_in_blocks(idx::Vector)

  idx_new = copy(idx)

  count_loc = zeros(Int,length(idx))
  for i=1:length(idx)
    if idx[i] != idx[1]+i-1
      count_loc[i] = 1
    end
  end

  count_glob = cumsum(count_loc)
  for i=1:length(idx)
    if count_loc[i] == 0
      global val = count_glob[i]
    end
    count_glob[i] = val
  end

  idx_new -= count_glob
  return idx_new

end

function from_spacetime_to_space_time_idx_mat(idx::Vector, Nᵤ::Int)

  idx_time = 1 .+ floor.(Int,(idx.-1)/Nᵤ^2)
  idx_space = idx - (idx_time.-1)*Nᵤ^2

  idx_space, idx_time

end

function invert_sparse_to_full_idx(sparse_to_full_idx::Vector,Nₛ::Int)
  r_idx, _ = from_vec_to_mat_idx(sparse_to_full_idx, Nₛ)
  full_to_sparse_idx = Int[]
  for i = 1:Nₛ
    append!(full_to_sparse_idx, findall(x -> x == i, r_idx))
  end
  return full_to_sparse_idx
end

function chebyshev_polynomial(x::Float, n::Int)

  if n == 0
    return 1
  elseif n == 1
    return 2*x
  else
    return 2*x*chebyshev_polynomial(x,n-1) - chebyshev_polynomial(x,n-2)
  end

end

function chebyschev_multipliers(x::Vector, order::Int, dim=3)

  Ξ = Matrix{Float}[]
  for d = 1:dim
    for n = 1:order
      for k = 1:n
        ωₖ = k*pi/(order+1)
        Pⁿₖ = chebyshev_polynomial(x[1]*cos(ωₖ*x[1]) + x[2]*sin(ωₖ*x[2]), n)/sqrt(pi)
        append!(Ξ, Pⁿₖ*I(dim)[:,d])
      end
    end
  end

  return Ξ

end


function index_mapping_inverse(i::Int,RBVars::PoissonUnsteady) ::Tuple
  iₛ = 1+floor(Int,(i-1)/RBVars.nₜᵘ)
  iₜ = i-(iₛ-1)*RBVars.nₜᵘ
  iₛ,iₜ
end
