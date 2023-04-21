struct DistributedAssembler
  assembler::Function
  allocated_snap::Matrix{Float}
  findnz_idx::Vector{Int}
end

function DistributedAssembler(op::ParamOperator,args...)
  initial_assembler = setup_assembler(op,args...)
  findnz_idx = get_findnz_idx(initial_assembler(1))
  assembler = setup_assembler(op,args...;findnz_idx)
  allocated_snap = assembler(1)
  DistributedAssembler(assembler,allocated_snap,findnz_idx)
end

get_assembler(::ParamOperator,args...) = assemble_vectors

get_assembler(::ParamBilinOperator,::Nothing) = assemble_matrices

get_assembler(da::DistributedAssembler) = da.assembler

get_allocated_snap(da::DistributedAssembler) = da.allocated_snap

get_findnz_idx(da::DistributedAssembler) = da.findnz_idx

function setup_assembler(
  op::RBVariable{Nonaffine,Ttr},
  μ::Vector{Param},
  args...;
  findnz_idx=nothing) where Ttr

  assembler = get_assembler(op,findnz_idx)
  k -> assembler(op,findnz_idx;μ=μ[k])
end

function setup_assembler(
  op::RBVariable{Nonlinear,Ttr},
  μ::Vector{Param},
  uh::Snapshots;
  findnz_idx=nothing) where Ttr

  assembler = get_assembler(op,findnz_idx)
  u_fun(k) = FEFunction(op,uh[k],μ[k])
  k -> assembler(op,findnz_idx;μ=μ[k],u=u_fun(k))
end

function setup_assembler(
  op::RBBilinVariable{Nonaffine,Ttr},
  μ::Vector{Param},
  fun::Function;
  findnz_idx=nothing) where Ttr

  assembler = get_assembler(op,findnz_idx)
  k -> assembler(op,findnz_idx;μ=μ[k],u=fun(k),t=first(get_timesθ(op)))
end

function setup_assembler(
  op::RBBilinVariable{Nonlinear,Ttr},
  μ::Vector{Param},
  fun::Function;
  findnz_idx=nothing) where Ttr

  assembler = get_assembler(op,findnz_idx)
  k -> assembler(op,findnz_idx;μ=first(μ),u=fun(k),t=first(get_timesθ(op)))
end

function Gridap.evaluate(da::DistributedAssembler,k::Int)
  assembler = get_assembler(da)
  snap = get_allocated_snap(da)
  findnz_idx = get_findnz_idx(da)
  copyto!(snap,assembler(k))
  dist_snap = DistMatrix(dist_snap)
  dist_snap,findnz_idx
end

function Gridap.evaluate(da::DistributedAssembler,k::UnitRange{Int})
  println("TO DO...")
end

function unpack_for_assembly(op::ParamLinOperator)
  get_param_fefunction(op),get_test(op)
end

function unpack_for_assembly(op::ParamBilinOperator)
  get_param_fefunction(op),get_test(op),get_trial(op)
end

function unpack_for_assembly(op::ParamLiftOperator)
  get_param_fefunction(op),get_test(op),get_dirichlet_function(op)
end

function assemble_vectors(op::ParamBilinOperator,::Nothing;kwargs...)
  error("Something is wrong")
end

function assemble_vectors(
  op::ParamLinOperator,
  ::Nothing;
  μ=realization(op),t=get_timesθ(op),u=nothing)::Matrix{Float}

  assemble_vectors(unpack_for_assembly(op)...,μ,t,u)
end

function assemble_vectors(
  op::ParamLinOperator,
  findnz_idx::Vector{Int};
  μ=realization(op),t=get_timesθ(op),u=nothing)::Matrix{Float}

  vecs = assemble_vectors(unpack_for_assembly(op)...,μ,t,u)
  get_findnz_vals(vecs,findnz_idx)
end

function assemble_vectors(
  fefun::Function,
  test::FESpace,
  μ::Param,t,args...)

  if isnothing(t)
    mat = Matrix(assemble_vector(fefun(μ),test))
  else
    mat = Matrix{Float}(undef,get_Ns(test),length(t))
    @inbounds for (n,tn) = enumerate(t)
      copyto!(view(mat,:,n),assemble_vector(fefun(μ,tn),test))
    end
  end

  mat
end

function assemble_vectors(
  fefun::Function,
  test::FESpace,
  dir::Function,
  μ::Param,t,u)

  if isnothing(t)

    if isnothing(u)
      mat = Matrix(assemble_vector(v->fefun(μ,dir(μ),v),test))
    else
      mat = Matrix(assemble_vector(v->fefun(u,dir(μ),v),test))
    end

  else

    mat = Matrix{Float}(undef,get_Ns(test),length(t))

    function vec(tn::Real)
      if isnothing(u)
        assemble_vector(v->fefun(μ,tn,dir(μ,tn),v),test)
      elseif typeof(u) == Function
        assemble_vector(v->fefun(u(tn),dir(μ,tn),v),test)
      else typeof(u) == FEFunction
        assemble_vector(v->fefun(u,dir(μ,tn),v),test)
      end
    end

    @inbounds for (n,tn) = enumerate(t)
      copyto!(view(mat,:,n),vec(tn))
    end

  end

  mat

end

function assemble_vectors(
  op::ParamBilinOperator,
  findnz_idx::Vector{Int};
  μ=realization(op),t=get_timesθ(op),u=nothing)

  mats = assemble_matrices(unpack_for_assembly(op)...,μ,t,u)
  get_findnz_vals(mats,findnz_idx)
end

function assemble_matrices(
  op::ParamBilinOperator,
  args...;
  μ=realization(op),t=get_timesθ(op),u=nothing)::Vector{SparseMatrixCSC{Float,Int}}

  assemble_vectors(unpack_for_assembly(op)...,μ,t,u)
end

function assemble_matrices(
  fefun::Function,
  test::FESpace,
  trial::Ttr,
  μ::Param,t,u)::Vector{SparseMatrixCSC{Float,Int}} where Ttr

  if isnothing(t)
    if isnothing(u)
      [assemble_matrix(fefun(μ),trial(μ),test)]
    else
      [assemble_matrix(fefun(u),trial(μ),test)]
    end
  else
    if isnothing(u)
      [assemble_matrix(fefun(μ,tθ),trial(μ,tθ),test) for tθ = t]
    elseif typeof(u) == Function
      [assemble_matrix(fefun(u(tθ)),trial(μ,tθ),test) for tθ = t]
    else typeof(u) == FEFunction
      [assemble_matrix(fefun(u),trial(μ,tθ),test) for tθ = t]
    end
  end
end

function get_findnz_vals(arr::Matrix{Float},findnz_idx::Vector{Int})
  Matrix(selectdim(arr,1,findnz_idx))
end

function get_findnz_vals(mat::SparseMatrixCSC{Float,Int},findnz_idx::Vector{Int})
  Matrix(mat[:][findnz_idx])
end

function get_findnz_vals(
  vec::Vector{SparseMatrixCSC{Float,Int}},
  findnz_idx::Vector{Int})

  nz,nvec = length(findnz_idx),length(vec)
  mat = Matrix{Float}(undef,nz,nvec)
  @inbounds @simd for k = eachindex(vec)
    copyto!(view(mat[:,k]),get_findnz_vals(vec[k],findnz_idx))
  end
  mat
end

function get_findnz_idx(mat::Matrix{Float})
  sum_cols = sum(mat,dims=2)[:]
  findall(x -> abs(x) ≥ eps(),sum_cols)
end

function get_findnz_idx(mat::SparseMatrixCSC{Float,Int})
  findnz_idx, = findnz(mat[:])
  findnz_idx
end

function get_findnz_idx(vecs::Vector{SparseMatrixCSC{Float,Int}})
  get_findnz_idx(first(vecs))
end
