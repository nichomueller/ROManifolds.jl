struct DistributedAssembler
  op::ParamOperator
  assembler::Function
  findnz_idx::Vector{Int}
end

function DistributedAssembler(op::ParamOperator,args...)
  initial_assembler = setup_assembler(op,args...)
  findnz_idx = get_findnz_idx(initial_assembler(1))
  assembler = setup_assembler(op,args...;findnz_idx)
  DistributedAssembler(op,assembler,findnz_idx)
end

get_op(da::DistributedAssembler) = da.op

get_assembler(da::DistributedAssembler) = da.assembler

get_findnz_idx(da::DistributedAssembler) = da.findnz_idx

# function setup_assembler(
#   op::ParamLinOperator{Nonaffine},
#   μ::Vector{Param},
#   args...)

#   findnz_idx = get_findnz_idx(assemble_vectors(op,μ[1]))
#   snaps = assemble_vectors(op,findnz_idx,μ)
#   snaps,findnz_idx
# end

# function setup_assembler(
#   op::ParamOperator{Nonlinear,Ttr},
#   μ::Vector{Param},
#   uh::Snapshots;
#   findnz_idx=nothing) where Ttr

#   assembler = get_assembler(op,findnz_idx)
#   u_fun(k) = FEFunction(op,uh[k],μ[k])
#   k -> assembler(op,findnz_idx;μ=μ[k],u=u_fun(k))
# end

# function setup_assembler(
#   op::ParamBilinOperator{Nonaffine,Ttr},
#   μ::Vector{Param},
#   fun::Function;
#   findnz_idx=nothing) where Ttr

#   assembler = get_assembler(op,findnz_idx)
#   k -> assembler(op,findnz_idx;μ=μ[k],u=fun(k),t=first(get_times(op)))
# end

# function setup_assembler(
#   op::ParamBilinOperator{Nonlinear,Ttr},
#   μ::Vector{Param},
#   fun::Function;
#   findnz_idx=nothing) where Ttr

#   assembler = get_assembler(op,findnz_idx)
#   k -> assembler(op,findnz_idx;μ=first(μ),u=fun(k),t=first(get_times(op)))
# end

function assemble(da::DistributedAssembler,idx::Base.OneTo{Int})
  op = get_op(da)
  id = get_id(op)
  nsnap = length(idx)
  printstyled("MDEIM: generating $nsnap snapshots for $id \n";color=:blue)

  assembler = get_assembler(da)
  findnz_idx = get_findnz_idx(da)
  Nz,Nt,ns = length(findnz_idx),get_Nt(op),length(idx)
  snaps = Elemental.zeros(EMatrix{Float},Nz,Nt*ns)
  @sync @distributed for k = idx
    copyto!(view(snaps,:,(k-1)*Nt+1:k*Nt),assembler(k))
  end
  s = Snapshots(id,snaps,nsnap)

  s,findnz_idx
end

function assemble(da::DistributedAssembler,μ::Vector{Param})
  assemble(da,eachindex(μ))
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

# function assemble_vectors(op::ParamBilinOperator,::Nothing;kwargs...)
#   error("Something is wrong")
# end

# function assemble_vectors(
#   op::ParamLinOperator,
#   args...;
#   μ=realization(op),t=get_times(op),u=nothing)::EMatrix{Float}

#   assemble_vectors(unpack_for_assembly(op)...,μ,t,u)
# end

# function assemble_vectors(
#   op::ParamLinOperator,
#   findnz_idx::Vector{Int};
#   μ=realization(op),t=get_times(op),u=nothing)::EMatrix{Float}

#   vecs = assemble_vectors(unpack_for_assembly(op)...,μ,t,u)
#   get_findnz_vals(vecs,findnz_idx)
# end

# function assemble_vectors(
#   fefun::Function,
#   test::FESpace,
#   μ::Param,t,args...)

#   if isnothing(t)
#     mat = EMatrix(assemble_vector(fefun(μ),test))
#   else
#     mat = EMatrix{Float}(undef,get_Ns(test),length(t))
#     @inbounds for (n,tn) = enumerate(t)
#       copyto!(view(mat,:,n),assemble_vector(fefun(μ,tn),test))
#     end
#   end

#   mat
# end

# function assemble_vectors(
#   fefun::Function,
#   test::FESpace,
#   dir::Function,
#   μ::Param,t,u)

#   if isnothing(t)

#     if isnothing(u)
#       mat = EMatrix(assemble_vector(v->fefun(μ,dir(μ),v),test))
#     else
#       mat = EMatrix(assemble_vector(v->fefun(u,dir(μ),v),test))
#     end

#   else

#     function vec(tn::Real)
#       if isnothing(u)
#         assemble_vector(v->fefun(μ,tn,dir(μ,tn),v),test)
#       elseif typeof(u) == Function
#         assemble_vector(v->fefun(u(tn),dir(μ,tn),v),test)
#       else typeof(u) == FEFunction
#         assemble_vector(v->fefun(u,dir(μ,tn),v),test)
#       end
#     end

#     mat = EMatrix{Float}(undef,get_Ns(test),length(t))
#     @inbounds for (n,tn) = enumerate(t)
#       copyto!(view(mat,:,n),vec(tn))
#     end

#   end

#   mat

# end

# function assemble_vectors(
#   op::ParamBilinOperator,
#   findnz_idx::Vector{Int};
#   μ=realization(op),t=get_times(op),u=nothing)

#   mats = assemble_matrices(unpack_for_assembly(op)...,μ,t,u)
#   get_findnz_vals(mats,findnz_idx)
# end

# function assemble_matrices(
#   op::ParamBilinOperator,
#   args...;
#   μ=realization(op),t=get_times(op),u=nothing)::SparseMatrixCSC{Float,Int}

#   assemble_matrices(unpack_for_assembly(op)...,μ,first(t),u)
# end

# function assemble_matrices(
#   fefun::Function,
#   test::FESpace,
#   trial::Ttr,
#   μ::Param,t::Real,u)::SparseMatrixCSC{Float,Int} where Ttr

#   if isnothing(t)
#     if isnothing(u)
#       assemble_matrix(fefun(μ),trial(μ),test)
#     else
#       assemble_matrix(fefun(u),trial(μ),test)
#     end
#   else
#     if isnothing(u)
#       assemble_matrix(fefun(μ,t),trial(μ,t),test)
#     elseif typeof(u) == Function
#       assemble_matrix(fefun(u(t)),trial(μ,t),test)
#     else typeof(u) == FEFunction
#       assemble_matrix(fefun(u),trial(μ,t),test)
#     end
#   end
# end

function assemble_matrices(
  f::Function,
  U::ParamTransientTrialFESpace,
  V::FESpace,
  μvec::Vector{Param},
  tvec::Vector{<:Real})

  a,get_matdata = assembler_setup(f,U,V,μvec,tvec)
  m1,findnz_idx = assembler_invariables(a,get_matdata,μvec,tvec)
  assembler_loop(a,get_matdata,m1,findnz_idx,μvec,tvec)
end

function assembler_setup(
  f::Function,
  U::ParamTransientTrialFESpace,
  V::FESpace,
  μvec::Vector{Param},
  tvec::Vector{<:Real})

  U1 = U(first(μvec),first(tvec))
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U1)
  a = SparseMatrixAssembler(U1,V)

  function get_matdata(μ::Param,t::Real)
    collect_cell_matrix(U(μ,t),V,f(μ,t,u,v))
  end

  a,get_matdata
end

function assembler_invariables(
  a::SparseMatrixAssembler,
  get_matdata::Function,
  μvec::Vector{Param},
  tvec::Vector{<:Real})

  matdata1 = get_matdata(first(μvec),first(tvec))
  m1 = Gridap.Algebra.nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  symbolic_loop_matrix!(m1,a,matdata1)
  m2 = Gridap.Algebra.nz_allocation(m1)
  numeric_loop_matrix!(m2,a,matdata1)
  m3 = Gridap.Algebra.create_from_nz(m2)
  findnz_idx = get_findnz_idx(m3)

  m1,findnz_idx
end

function assembler_loop(
  a::SparseMatrixAssembler,
  get_matdata::Function,
  m1::Gridap.Algebra.CounterCSC{Float,Int,Gridap.Algebra.Loop},
  findnz_idx::Vector{Int},
  μvec::Vector{Param},
  tvec::Vector{<:Real})

  Nz,Nt,np = length(findnz_idx),length(tvec),length(μvec)
  mat = Elemental.zeros(EMatrix{Float},Nz,Nt*np)
  @distributed for k = eachindex(μvec)
    for n = eachindex(tvec)
      matdata_kn = get_matdata(μvec[k],tvec[n])
      m2 = Gridap.Algebra.nz_allocation(m1)
      numeric_loop_matrix!(m2,a,matdata_kn)
      copyto!(view(mat,:,(k-1)*Nt+n),my_create_from_nz(m2))
    end
  end

  mat
end

function my_create_from_nz(a::Gridap.Algebra.InserterCSC{Tv,Ti}) where {Tv,Ti}
  k = 1
  for j in 1:a.ncols
    pini = Int(a.colptr[j])
    pend = pini + Int(a.colnnz[j]) - 1
    for p in pini:pend
      a.nzval[k] = a.nzval[p]
      k += 1
    end
  end
  @inbounds for j in 1:a.ncols
    a.colptr[j+1] = a.colnnz[j]
  end
  Gridap.FESpaces.length_to_ptrs!(a.colptr)
  nnz = a.colptr[end]-1
  resize!(a.nzval,nnz)

  my_nnz = findall(v -> abs.(v) .>= eps(), a.nzval)
  a.nzval[my_nnz]
end

function get_findnz_vals(
  ::EMatrix,
  arr::AbstractMatrix{Float},
  findnz_idx::Vector{Int})

  EMatrix(get_findnz_vals(arr,findnz_idx))
end

function get_findnz_vals(
  vec::Vector{SparseMatrixCSC{Float,Int}},
  findnz_idx::Vector{Int})

  nz,nvec = length(findnz_idx),length(vec)
  mat = Elemental.zeros(EMatrix{Float},nz,nvec)
  @inbounds @simd for k = eachindex(vec)
    copyto!(view(mat,:,k),get_findnz_vals(vec[k],findnz_idx))
  end
  mat
end

function get_findnz_idx(mat::EMatrix{Float})
  sum_cols = sum(mat,dims=2)[:]
  findall(x -> abs(x) ≥ eps(),sum_cols)
end

function get_findnz_idx(mat::SparseMatrixCSC{Float,Int})
  findnz_idx, = findnz(mat[:])
  findnz_idx
end
