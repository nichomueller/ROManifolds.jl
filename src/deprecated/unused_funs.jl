function plot_θ_comparison(timesθ, θ, θ_approx)

  layout = Layout(title="θᵃ vs approximated θᵃ, ε = 10⁻⁵ ",xaxis_title="time [s]",yaxis_title="θᵃ(t)")
  #= selected_color=["black"]
  for _ = 1:size(θ_approx)[2]
    append!(selected_color, ["blue"])
  end =#
  θ_plt = hcat(θ, θ_approx)
  #traces = [scatter(x=timesθ,y=θ_plt[:,i],line=attr(width=4,color=selected_color[i])) for i=1:size(θ_plt)[2]]
  sel_colors = ["black","blue"]
  sel_dashes = ["","dash"]
  names = ["θᵃ", "θᵃ approx"]
  traces = [scatter(x=timesθ,y=θ_plt[:,i],name=names[i],
    line=attr(width=4,color=sel_colors[i],dash=sel_dashes[i])) for i=1:size(θ_plt)[2]]
  p = plot(traces,layout)
  savefig(p, joinpath("/home/nmueller/scratch/tests/unsteady/poisson/case3/cube20x20x20.json/ST-GRB_st_fun_-5/θᵃ.eps"))

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
